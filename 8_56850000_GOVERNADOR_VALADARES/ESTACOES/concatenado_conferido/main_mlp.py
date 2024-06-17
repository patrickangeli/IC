import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
from imblearn.over_sampling import SMOTE
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, confusion_matrix
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import Adam

def main(
    nome_bacia: str,
    file_input_cota: str, 
    file_input_chuva: str,
    tempo_antecedencia: int,
    num_steps: int,
    nome_rede: str,
    rodar_cross_validation: bool,
    utilizar_percentil_corte: bool,
    incluir_balanceamento: bool,
         ):

    # (01) Definir a grade de parâmetros para o GridSearch
    param_grid = {
        'model__neurons': [32, 64, 128],
        'model__learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [32, 64, 128],
        'epochs': [100]
    }
    
    # (02) Carregar o conjunto de dados
    DIR_OUTPUT_BASE = f"Saida/{nome_bacia}/{nome_rede}"
    Path(DIR_OUTPUT_BASE).mkdir(exist_ok=True, parents=True)
    df_cota = pd.read_csv(file_input_cota, delimiter=';')
    df_chuva = pd.read_csv(file_input_chuva, delimiter=';')
    lst_names_colunms_cota = list(df_cota.columns)
    lst_names_colunms_cota = [f"cota_{coluna}"for coluna in lst_names_colunms_cota]
    lst_names_colunms_chuva = list(df_chuva.columns)
    lst_names_colunms_chuva = [f"chuva_{coluna}"for coluna in lst_names_colunms_chuva]
    
    porc_registro_por_row = 0.5
    df_cota.columns = lst_names_colunms_cota
    df_chuva.columns = lst_names_colunms_chuva
    df_concatenado = pd.concat([df_chuva, df_cota], axis=1)
    df_concatenado = df_concatenado.drop('cota_data', axis=1)
    df_concatenado = df_concatenado.rename(columns={'chuva_data': 'data'})
    df_concatenado.replace('-999.99', -999.99, inplace=True)
    lst_cod_estacoes = list(df_concatenado.columns)
    num_necessario_registros_por_row = int(porc_registro_por_row * len(lst_cod_estacoes))
    estacao_interesse = lst_cod_estacoes[-1]
    df_concatenado_filter = df_concatenado.loc[df_concatenado[estacao_interesse] != -999.99]
   
    colunas_interesse = df_concatenado_filter.columns[~df_concatenado_filter.columns.isin(['data', estacao_interesse])]
    condicao = (df_concatenado_filter[colunas_interesse] != -999.99).sum(axis=1) >= num_necessario_registros_por_row
    df_concatenado_filter = df_concatenado_filter.loc[condicao]
    df_concatenado_filter = df_concatenado_filter.reset_index(drop=True)
    
    df_cota_estacao_interesse = df_concatenado[estacao_interesse]
    valores_niveis = list(df_cota_estacao_interesse)
    
    if utilizar_percentil_corte:
        percentil_corte = [85]
        limite_corte = list(np.percentile(valores_niveis, percentil_corte))[0]
        
        # Condição para eliminar linhas onde a coluna 'estacao_interesse' é menor que limite_corte
        condition = df_concatenado[estacao_interesse] < limite_corte
        df_concatenado_filter = df_concatenado.drop(df_concatenado[condition].index)

    percentis_corte = [90, 95]
    limites = list(np.percentile(valores_niveis, percentis_corte))
    df_concatenado_filter['classe'] = df_concatenado_filter[estacao_interesse].apply(rotular_cota, p1=limites[0], p2=limites[1])
    df_concatenado_filter['classe'] = df_concatenado_filter['classe'].astype(int)
    
    lst_datas = list(df_concatenado_filter['data'])
    formato_string = '%d/%m/%Y %H:%M'
    lst_datetimes = []
    for data_iter in lst_datas:
        data_iter_datetime = datetime.strptime(data_iter, formato_string)
        lst_datetimes.append(data_iter_datetime)

    # (03) Criando sequencias e padronizando os dados
    scaler = MinMaxScaler(feature_range=(0,1))
    data_X = scaler.fit_transform(df_concatenado_filter[lst_cod_estacoes[1:-1]].values)
    data_Y = np.array(df_concatenado_filter[['classe']].values)
    data_Y = data_Y.astype(int)
    X, y = create_sequences(data_X, data_Y, tempo_antecedencia, lst_datetimes, num_steps)
    
    train_size = 0.70
    val_size = 0.15
    test_size = 0.15
    
    # (04) Separar conjunto de dados em treino, validação e teste
    # Primeiro, divida em treino+validação e teste
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Em seguida, divida o conjunto treino+validação em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size/(train_size+val_size), random_state=42)
    X_train = np.expand_dims(X_train, axis=1)
    X_train = np.concatenate(X_train, axis=0)
    
    if incluir_balanceamento:
        smote = SMOTE(random_state=42)
        
        # BALANCEANDO OS DADOS DE TREINO
        ##########################################################
        print('Distribuição original de treino:', Counter(y_train.ravel()))
        y_train = y_train.ravel()

        # Achata X_train para 2D
        num_samples = X_train.shape[0]
        X_train_2d = X_train.reshape(num_samples, -1)
        
        # Aplica o SMOTE aos dados
        original_shape_X = X_train.shape
        X_resampled, y_resampled = smote.fit_resample(X_train_2d, y_train)
        
        # Reformatar X_resampled de volta para a forma original
        new_shape = (X_resampled.shape[0],) + original_shape_X[1:]
        X_train = X_resampled.reshape(new_shape)
        y_train = to_categorical(y_resampled, num_classes=3)
        ###########################################################
        # BALANCEANDO OS DADOS DE VALIDAÇÃO
        print('Distribuição original de validação:', Counter(y_val.ravel()))
        y_val = y_val.ravel()

        # Achata X_train para 2D
        num_samples = X_val.shape[0]
        X_val_2d = X_val.reshape(num_samples, -1)
        
        # Aplica o SMOTE aos dados
        original_shape_X = X_val.shape
        X_resampled, y_resampled = smote.fit_resample(X_val_2d, y_val)
        
        # Reformatar X_resampled de volta para a forma original
        new_shape = (X_resampled.shape[0],) + original_shape_X[1:]
        X_val = X_resampled.reshape(new_shape)
        y_val = to_categorical(y_resampled, num_classes=3)
        ################################################################
    else:
        y_train = to_categorical(y_train, num_classes=3)
        y_val = y_val.ravel()
        y_val = to_categorical(y_val, num_classes=3)
    
    # (05) Definir o KerasClassifier com os parâmetros padrão
    num_atributos = X_train.shape[2]
    model = KerasClassifier(model=create_model_mlp, verbose=1, num_steps=num_steps, num_atributos=num_atributos)

    # (06) Definir a estratégia de validação cruzada
    if rodar_cross_validation:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, error_score='raise')
    else:
        cv = 2
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, error_score='raise')
    grid_result = grid.fit(X_train, y_train, validation_data=(X_val, y_val))

    dir_output_result_train = f"{DIR_OUTPUT_BASE}/train/antecedencia_{tempo_antecedencia}hours/{rodar_cross_validation}_cross/{utilizar_percentil_corte}_percentil/{incluir_balanceamento}_balanceamento"
    Path(dir_output_result_train).mkdir(exist_ok=True, parents=True)
    with open(f'{dir_output_result_train}/resultTrain_{num_steps}steps_croos.txt', 'w') as arquivo:
        arquivo.write(f'melhor_config={grid_result.best_params_}\n')
        arquivo.write(f'accuracy={grid_result.best_score_}\n')

    # Imprimir o melhor estimador
    print('=' * 20)
    print(f"Melhor pontuação: {grid_result.best_score_} usando {grid_result.best_params_}")
    print('=' * 20)

   # (07) Testando modelo
    X_test = np.expand_dims(X_test, axis=1)
    X_test = np.concatenate(X_test, axis=0)  
    Y_test = to_categorical(y_test, num_classes=3)
    best_model = grid_result.best_estimator_
    predict_test = best_model.predict(X_test)
    predict_test_normalized = predict_test / np.sum(predict_test)
    classe_predita = np.argmax(predict_test_normalized, axis=1)
    data_Y_predict = classe_predita.ravel().tolist()
    data_Y_org = np.argmax(Y_test, axis=1).tolist()
    
    dir_output_result_test = f"{DIR_OUTPUT_BASE}/teste/antecedencia_{tempo_antecedencia}hours/{rodar_cross_validation}_cross/{utilizar_percentil_corte}_percentil/{incluir_balanceamento}_balanceamento"
    Path(dir_output_result_test).mkdir(exist_ok=True, parents=True)
    file_info_test = f'{dir_output_result_test}/resultTest_{num_steps}steps.txt'
    gerar_csv_teste(data_Y_predict, data_Y_org, file_info_test)

    # (08) Plot confusion_matrix
    labels = sorted(set(data_Y_org))  # Assumindo que os rótulos são inteiros
    conf_matrix_path = f'{dir_output_result_test}/confusion_matrix_{num_steps}steps.png'
    title = f"{nome_rede}({str(tempo_antecedencia).zfill(2)}hours) - {num_steps} steps"
    plt_confusion_matrix(data_Y_org, data_Y_predict, labels, title, save_path=conf_matrix_path)

    # (08) Plot das métricas estatísticas calculadas
    acuracia = accuracy_score(data_Y_org, data_Y_predict)
    precisao = precision_score(data_Y_org, data_Y_predict, average='macro')
    recall = recall_score(data_Y_org, data_Y_predict, average='macro')
    f1 = f1_score(data_Y_org, data_Y_predict, average='macro')
    lst_labels_metricas = ['accuracy_score', 'precision_score', 'recall_score', 'f1_score']
    lst_values_metricas = [acuracia, precisao, recall, f1]
    
    plt.figure(figsize=(12, 8))
    plt.bar(lst_labels_metricas, lst_values_metricas, color=['blue', 'green', 'red', 'orange'])
    
    title = f"Performance Metrics: {nome_rede}({str(tempo_antecedencia).zfill(2)}hours) - {num_steps} steps"
    plt.title(title)
    plt.ylim(0.2, 1.0)
    plt.xticks(lst_labels_metricas, lst_labels_metricas, rotation=45)
    plt.tight_layout()
    fielename_fig = f'{dir_output_result_test}/metrics_{num_steps}steps.png'
    plt.savefig(fielename_fig)
    plt.close()
    
    # (09) Plot importancia das variáveis
    plot_importancia_vars_preditoras()
    
    
def plot_importancia_vars_preditoras():
    pass

def create_model_mlp(num_steps, num_atributos, neurons, learning_rate):
    model = Sequential()
    model.add(SimpleRNN(units=neurons, activation='relu', input_shape=(num_steps, num_atributos)))
    model.add(Dense(units=3, activation='sigmoid'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_sequences(data_x, data_y, tempo_antecedencia, lst_datas, num_steps):
    X, Y = [], []
    len_data = len(data_x)
    for i in range(len_data):
        time_init = lst_datas[i]
        try:
            time_interesse = lst_datas[i+tempo_antecedencia]
        except:
            break
        diferenca = time_interesse - time_init
        diferenca_horas = diferenca.total_seconds() / 3600
        if diferenca_horas != tempo_antecedencia:
            continue
        time_inicial_capturado =  time_init - timedelta(hours=num_steps-1)
        datetimes_no_intervalo = [dt for dt in lst_datas if time_inicial_capturado <= dt <= time_init]
        if len(datetimes_no_intervalo) != num_steps:
            continue
        
        id_X_input_inicial = i - num_steps
        if id_X_input_inicial < 0:
            continue
        id_X_input_final = i
        try:
            id_Y_input = id_X_input_final + tempo_antecedencia
            y_value = data_y[id_Y_input]
            Y.append(y_value)
        except:
            break
        
        x_sequence = data_x[id_X_input_inicial:id_X_input_final]
        X.append(x_sequence)
    return np.array(X), np.array(Y)

def rotular_cota(valor, p1, p2):
    if valor >= 0 and valor <= p1:
        return 0
    elif valor > p1 and valor < p2:
        return 1
    elif valor >= p2:
        return 2
    else:
        return -999.99
    
def gerar_csv_teste(y_pred, y_true, file_info_output):
    coluna_pred = [elem for elem in y_pred]
    coluna_true = [elem for elem in y_true]
    df = pd.DataFrame({'y_true': coluna_true, 'y_pred': coluna_pred})
    
    dir_output = f"{Path(file_info_output).parent}"
    Path(dir_output).mkdir(exist_ok=True, parents=True)
    file_csv = f"{dir_output}/{Path(file_info_output).name.replace('.txt', '.csv')}"
    df.to_csv(file_csv, index=False)
    
def plt_confusion_matrix(y_true, y_pred, labels, title, save_path):
    confusionMatrix = confusion_matrix(y_true, y_pred, labels=labels)
    normalizedMatrix = confusionMatrix.astype('float') / (confusionMatrix.sum(axis=1)[:, np.newaxis] + 1e-10)

    plt.imshow(normalizedMatrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - ' + title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    threshold = normalizedMatrix.max() / 2.0
    for i in range(normalizedMatrix.shape[0]):
        for j in range(normalizedMatrix.shape[1]):
            plt.text(j, i, '{:.2%}'.format(normalizedMatrix[i, j]), horizontalalignment="center",
                     color="white" if normalizedMatrix[i, j] > threshold else "black")

    plt.savefig(save_path)
    plt.close()
    
if __name__ == "__main__":
    time_init = datetime.now()
    lst_tempos_antecedencia = [6,12,24]
    lst_num_steps = [4, 6, 8]
    for tempo_antecedencia in lst_tempos_antecedencia:
        for num_steps in lst_num_steps:
            main(nome_bacia="bacia_valadares",
                file_input_cota='cota.csv',
                file_input_chuva='chuva.csv',
                tempo_antecedencia=tempo_antecedencia,
                num_steps=num_steps,
                nome_rede="MLP",
                rodar_cross_validation=True,
                utilizar_percentil_corte=False,
                incluir_balanceamento=False,
                )
    time_fim = datetime.now()
    print_statement = f"Inicio: {time_init} / Fim: {time_fim}"

    # Save the print statement to a text file
    file_path = "print_statement.txt"
    with open(file_path, "w") as file:
        file.write(print_statement)

    # Optionally, you can also print it to the console
    print(print_statement)
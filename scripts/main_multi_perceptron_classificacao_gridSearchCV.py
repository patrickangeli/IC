
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# 0-NORMAL, 1-ATENÇÃO, 2-INUNDAÇÃO

def main(file_input_cota: str,
         file_input_chuva: str,
         tempo_antecedencia: int,
         num_steps: int,
    #    num_neurons: int,
    #     num_epochs: int,
    #     func_camada_oculta: str,
    #     func_camada_saida: str,
    #     learning_rate: float,
    #     batch_size: int,
         param_grid: dict,
         porc_registro_por_row: float,
         dir_output: str,

         ):
    df_cota = pd.read_csv(file_input_cota, delimiter=';')
    df_chuva = pd.read_csv(file_input_chuva, delimiter=';')
    lst_names_colunms_cota = list(df_cota.columns)
    lst_names_colunms_cota = [f"cota_{coluna}"for coluna in lst_names_colunms_cota]
    lst_names_colunms_chuva = list(df_chuva.columns)
    lst_names_colunms_chuva = [f"chuva_{coluna}"for coluna in lst_names_colunms_chuva]

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
    percentis_corte = [80, 85, 90]
    limites = list(np.percentile(valores_niveis, percentis_corte))
    df_concatenado_filter['classe'] = df_concatenado_filter[estacao_interesse].apply(rotular_cota, p1=limites[0], p2=limites[1])
    df_concatenado_filter['classe'] = df_concatenado_filter['classe'].astype(int)
    contagem_eventos_inundação = df_concatenado_filter['classe'].value_counts().get(2, 0)
    lst_datas = list(df_concatenado_filter['data'])

    formato_string = '%d/%m/%Y %H:%M'
    lst_datetimes = []
    for data_iter in lst_datas:
        data_iter_datetime = datetime.strptime(data_iter, formato_string)
        lst_datetimes.append(data_iter_datetime)

    scaler = MinMaxScaler(feature_range=(0,1))
    data_X = scaler.fit_transform(df_concatenado_filter[lst_cod_estacoes[1:-1]].values)
    data_Y = np.array(df_concatenado_filter[['classe']].values)
    data_Y = data_Y.astype(int)

    
    X, Y = create_sequences(data_X, data_Y, tempo_antecedencia, lst_datetimes, num_steps)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    result_train, model = create_model()
    GrindSearch(X_train, Y_train, model)
    accuracy = result_train.history['accuracy']
    loss = result_train.history['loss']

    for num_neurons in param_grid['num_neurons']:
        for num_steps in param_grid['num_steps']:
            lst_result = zip(loss, accuracy)
            dir_output_result = f"{dir_output}/MLP/{tempo_antecedencia}hours"
            Path(dir_output_result).mkdir(exist_ok=True, parents=True)
            with open(f'{dir_output_result}/resultTrain_{num_neurons}neurons_{num_steps}steps.txt', 'w') as arquivo:
                for idx, epoch in enumerate(lst_result):
                    arquivo.write(f'EPOCH {idx+1} - loss: {round(epoch[0], 4)}, accuracy: {round(epoch[1], 4)} \n')

    # TESTE
    X_test = np.expand_dims(X_test, axis=1)
    X_test = np.concatenate(X_test, axis=0)
    Y_test = Y_test.astype(int)
    predict_test = test_neural_network(X_test, model)
    predict_test_normlized = predict_test / np.sum(predict_test)
    classe_predita = np.argmax(predict_test_normlized, axis=1)
    data_Y_predict = classe_predita.ravel().tolist()
    data_Y_org = Y_test.ravel().tolist()

    file_info_test = f'{dir_output_result}/resultTest_{num_neurons}neurons_{num_steps}steps.txt'
    gerar_csv_teste(data_Y_predict, data_Y_org, file_info_test)

def rotular_cota(valor, p1, p2):
    if valor >= 0 and valor <= p1:
        return 0
    elif valor > p1 and valor <= p2:
        return 1
    elif valor > p2:
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

def test_neural_network(x_test, model):
    predict_values = model.predict(x_test)
    return predict_values

def create_model(num_neurons=1, learning_rate=0.001, func_camada_oculta='relu', func_camada_saida='softmax'):
    num_classes = 3
    model = Sequential()
    model.add(SimpleRNN(units=num_neurons, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def GrindSearch(x_train, y_train, create_model, param_grid):
    num_classes = 3 
    model = KerasClassifier(build_fn=create_model, verbose=2)
    y_train_encoded = to_categorical(y_train, num_classes=num_classes)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=None, cv=5)
    grid_result = grid.fit(x_train, y_train_encoded)
    return grid_result




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

if __name__ == "__main__":
    main(file_input_cota='../IC/Entrada/cota.csv',
         file_input_chuva='../IC/Entrada/chuva.csv',
         tempo_antecedencia=8,
         num_steps = 6,
         param_grid = dict(num_neurons=[24,36,48,60,72], num_epochs=500, batch_size=100),
       #  batch_size=100,
         porc_registro_por_row = 0.5,
         dir_output="../IC/Saida")
import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime, timedelta
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def main(file_input_cota: str, 
         file_input_chuva: str,
         tempo_antecedencia: int,
         num_steps: int,
         num_celulas_lstm: int,
         num_camadas_lstm_empilhadas: int,
         num_epochs: int,
         batch_size: int,
         loss: str,
         porc_registro_por_row: float
         ):
    start_time = time.time()
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
    estacao_interesse = lst_cod_estacoes[-1]
    num_necessario_registros_por_row = int(porc_registro_por_row * len(lst_cod_estacoes))
    estacao_interesse = lst_cod_estacoes[-1]
    df_concatenado_filter = df_concatenado.loc[df_concatenado[estacao_interesse] != -999.99] 
    
    colunas_interesse = df_concatenado_filter.columns[~df_concatenado_filter.columns.isin(['data', estacao_interesse])]
    condicao = (df_concatenado_filter[colunas_interesse] != -999.99).sum(axis=1) >= num_necessario_registros_por_row
    df_concatenado_filter = df_concatenado_filter.loc[condicao]
    df_concatenado_filter = df_concatenado_filter.reset_index(drop=True)
    
    df_cota_estacao_interesse = df_concatenado[estacao_interesse]
    valores_niveis = list(df_cota_estacao_interesse)
    percentis_corte = [90, 93.33, 96.66,100]
    limites = list(np.percentile(valores_niveis, percentis_corte))
    df_concatenado_filter['classe'] = df_concatenado_filter[estacao_interesse].apply(rotular_cota, p1=limites[0], p2=limites[1])
    df_concatenado_filter = df_concatenado_filter[df_concatenado_filter['classe'] != 0]
    valores_limites = list(df_concatenado_filter[estacao_interesse])
    limites = list(np.percentile(valores_limites, percentis_corte))
    df_concatenado_filter['classe_final'] = df_concatenado_filter[estacao_interesse].apply(rotular_cota, p1=limites[0], p2=limites[1])
    contagem_eventos_inundação = df_concatenado_filter['classe'].value_counts().get(2, 0) 
    lst_datas = list(df_concatenado_filter['data'])
    df_concatenado_filter['classe'] = df_concatenado_filter['classe_final']
    df_concatenado_filter.drop(columns=['classe_final'], inplace=True)
    formato_string = '%d/%m/%Y %H:%M'
    lst_datetimes = []
    for data_iter in lst_datas:
        data_iter_datetime = datetime.strptime(data_iter, formato_string)
        lst_datetimes.append(data_iter_datetime)
    
    scaler = MinMaxScaler(feature_range=(0,1))
    data_X = scaler.fit_transform(df_concatenado_filter[lst_cod_estacoes[1:-1]].values)
    data_Y = np.array(df_concatenado_filter[['classe']].values)
    data_Y = data_Y.astype(int)
    
    data_X_seq, data_Y_seq = create_sequences(data_X, data_Y, tempo_antecedencia, lst_datetimes, num_steps)
    X_train, X_test, Y_train, Y_test = train_test_split(data_X_seq, data_Y_seq, test_size=0.3, random_state=42)
    Y_train = Y_train.astype(int)
    Y_test = Y_test.astype(int)
    
    # Redimensionando os dados de entrada da LSTM que precisam estar no formato:
    # (número de amostras, número de passos no tempo, número de características).
    num_features = len(X_train[0][0])
    x_train_redimension = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], num_features))
    x_test_redimension = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], num_features))

    model = Sequential()
    for _ in range(num_camadas_lstm_empilhadas):
        model.add(LSTM(units=num_celulas_lstm, 
                    return_sequences=True, input_shape=(x_train_redimension.shape[1], num_features)))
    
    model.add(LSTM(units=num_celulas_lstm, return_sequences=False))
    model.add(Dense(units=len(percentis_corte), activation='softmax'))
    model.add(Dropout(0.2))
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    result_fit = model.fit(x_train_redimension, Y_train, epochs=num_epochs, batch_size=batch_size)
    lst_loss = result_fit.history['loss']
    lst_accuracy = result_fit.history['accuracy']

    end_time = time.time()
    execution_time = end_time - start_time
    dir_result = f"Saida/LSTM/{tempo_antecedencia}hours_new/90-95-97"
    Path(dir_result).mkdir(exist_ok=True, parents=True)
    file_output = f"resultTrain_{num_celulas_lstm}neurons_{num_steps}_steps_{batch_size}_batch_0.2_dropout.txt"
    filename_output = f"{dir_result}/{file_output}"
    with open(filename_output, 'w') as arquivo:
        for idx, epoch in enumerate(lst_loss):
            arquivo.write(f'EPOCH {idx+1} - loss: {round(epoch, 4)}, accuracy: {round(lst_accuracy[idx], 4)} \n')
        arquivo.write(f'\nExecution Time: {round(execution_time, 2)} seconds\n')

    # TESTE
    predict_test = model.predict(x_test_redimension)
    predict_test_normlized = predict_test / np.sum(predict_test)
    data_Y_predict = list(np.argmax(predict_test_normlized, axis=1))
    data_Y_org = Y_test.ravel().tolist()
    file_output = f"resultTest_{num_celulas_lstm}neurons_{num_steps}_steps_{batch_size}_batch_0.2_dropout.csv"
    filename_output = f"{dir_result}/{file_output}"
    gerar_csv_teste(data_Y_predict, data_Y_org, filename_output)

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



if __name__ == "__main__":
 #   input_data = [("tempo_antecedencia", lst_tempo_antecedencia)]
#    input_data = [("steps", lst_steps)]
#    input_data = [("n_estimators", lst_n_estimators)]
#    input_data = [("batch_size", batch_size)]

    lst_tempo_antecedencia = [12] # 6 a 24
    lst_steps = [24] # 6 a 12
    lst_n_estimators = [84] # 60, 72 e 84
    batch_size = [150]

    for tempo_antecedencia in lst_tempo_antecedencia:
        for estimador in lst_n_estimators:
            for step in lst_steps:
                for batch in batch_size:
                    main(file_input_cota='cota.csv',
                        file_input_chuva='chuva.csv',
                        tempo_antecedencia=tempo_antecedencia,
                        num_steps = step,
                        num_celulas_lstm=estimador,
                        num_camadas_lstm_empilhadas=1,
                        num_epochs=500,
                        batch_size=batch, #50 e 150
                        loss = 'sparse_categorical_crossentropy',
                        porc_registro_por_row = 0.5
                        )
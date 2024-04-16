import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def main(file_input_cota: str, 
         file_input_chuva: str,
         tempo_antecedencia: int,
         n_estimators: int, 
         porc_necessario_registros_por_row: float,
         num_steps: int, 
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
    num_necessario_registros_por_row = int(porc_necessario_registros_por_row * len(lst_cod_estacoes))
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
    
    # TESTE
    model = train_random_forest(X_train, Y_train, n_estimators)
    predict_test = test_random_forest(X_test, model)
    data_Y_org = Y_test.ravel().tolist()
    data_Y_predict = predict_test.ravel().tolist()
    data_Y_org = [int(round(valor, 2)) for valor in data_Y_org]
    data_Y_predict = [int(round(valor, 2)) for valor in predict_test]
    
    dir_output_result = f"{dir_output}/RF/{tempo_antecedencia}hours"
    Path(dir_output_result).mkdir(exist_ok=True, parents=True)
    file_info_test = f'{dir_output_result}/resultTest_{n_estimators}estimators_{num_steps}steps.txt'
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
    
def test_random_forest(x_test, model):
    x_test = x_test.reshape(x_test.shape[0], -1)
    y_pred = model.predict(x_test)
    return y_pred

def train_random_forest(x_train, y_train, n_estimators):
    x_train = x_train.reshape(x_train.shape[0], -1)
    y_train = y_train.ravel()
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(x_train, y_train)
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

if __name__ == "__main__":
    lst_tempo_antecedencia = [6, 8, 10, 12, 14, 16,20,24] # 6 a 24
    lst_steps = [6, 12] # 6 a 12
    lst_n_estimators = [24, 36, 48, 60, 72] # 24 ate 72, variando de 12 em 12
    
    for estimador in lst_n_estimators:
        for step in lst_steps:
            for tempo_antecedencia in lst_tempo_antecedencia:
                main(file_input_cota='/content/drive/Shareddrives/PREDIÇÃO_INUNDAÇÃO/2024/DADOS/BACIA_DOCE/SUB_BACIAS_DOCE/1_5611005_RIO_PIRANGA/ESTACOES/DEEP_LEARNING/IC/Entrada/cota.csv',
                    file_input_chuva='/content/drive/Shareddrives/PREDIÇÃO_INUNDAÇÃO/2024/DADOS/BACIA_DOCE/SUB_BACIAS_DOCE/1_5611005_RIO_PIRANGA/ESTACOES/DEEP_LEARNING/IC/Entrada/chuva.csv',
                    tempo_antecedencia=tempo_antecedencia,
                    num_steps=step,
                    n_estimators=estimador,
                    porc_necessario_registros_por_row=0.5,
                    dir_output="Saida/bacia_piranga")
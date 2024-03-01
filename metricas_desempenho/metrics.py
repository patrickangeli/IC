#accuracy e metricas gerais
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def accuracy(entrada_dir):
    df = pd.read_csv(entrada_dir)
    
    y_true = np.array(df['y_true'])
    y_pred = np.array(df['y_pred'])
    
    accScore = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Acurácia: {accScore}')

    f1Score = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    print(f'Escore F1: {f1Score}')
    
    precScore = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
    print(f'Precision Score: {precScore}')

    recallScore = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
    print(f'Recall Score: {recallScore}')



if __name__ == '__main__':
    entrada_dir = 'Saida/bacia_piranga/3hours/relu/resultTest_24neurons_500epochs.csv'
    accuracy(entrada_dir)

import pandas as pd
from sklearn.metrics import accuracy_score

def accuracy(entrada_dir):
    df = pd.read_csv(entrada_dir)
    y_true = df['y_true']
    y_pred = df['y_pred']
    acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(acc_score*100)


if __name__ == '__main__':
    entrada_dir = 'Saida/bacia_piranga/3hours/relu/resultTest_24neurons_500epochs.csv'
    accuracy(entrada_dir)
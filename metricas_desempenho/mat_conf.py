import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plt_confusion_matrix(y_true, y_pred, labels):
    # 0 -> normal, 1 -> atenção, 2 -> inundação
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()



def main(entrada_dir):

    #ler entrada em um df
    df = pd.read_csv(entrada_dir)
    y_true = df['y_true']
    y_pred = df['y_pred']
    labels = [0, 1, 2]
    # print(df.head())
    plt_confusion_matrix(y_true, y_pred, labels)




if __name__ == '__main__':
    entrada_dir = 'Saida/bacia_piranga/3hours/relu/resultTest_24neurons_500epochs.csv'
    main(entrada_dir)
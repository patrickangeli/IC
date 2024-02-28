import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plt_confusion_matrix(y_true, y_pred, labels):
    confusionMatrix = confusion_matrix(y_true, y_pred, labels=labels)
    #normalizedMatrix = confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis]
    normalizedMatrix = confusionMatrix.astype('float') / (confusionMatrix.sum(axis=1)[:, np.newaxis] + 1e-10)

    plt.imshow(normalizedMatrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Add percentage values within the plot
    threshold = normalizedMatrix.max() / 2.0
    for i in range(normalizedMatrix.shape[0]):
        for j in range(normalizedMatrix.shape[1]):
            plt.text(j, i, '{:.2%}'.format(normalizedMatrix[i, j]), horizontalalignment="center",
                     color="white" if normalizedMatrix[i, j] > threshold else "black")

    plt.show()

def main(entrada_dir):

    #ler entrada em um df
    df = pd.read_csv(entrada_dir)
    y_true = df['y_true']
    y_pred = df['y_pred']
    labels = [0, 1]
    # print(df.head())
    plt_confusion_matrix(y_true, y_pred, labels)





if __name__ == '__main__':
    entrada_dir = 'Saida/bacia_piranga/3hours/relu/resultTest_24neurons_500epochs.csv'
    main(entrada_dir)
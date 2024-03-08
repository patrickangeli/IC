import os
from metrics import accuracy
from mat_conf import confusionMatrixPlot


def main(hours):
    input_dir = "Saida/" + hours + "hours/relu/"
    output_dir = "Metrics_results/" + (input_dir.replace("Saida/", ""))

    #lista todos os csv's da saida
    csv_list = [file for file in os.listdir(input_dir) if file.endswith(".csv")]

    #preparando a saida
    if not os.path.exists(output_dir):
        # Cria a pasta se não existir
        os.makedirs(output_dir)

    for csv in csv_list:
        csv_path = input_dir + csv
        acc, f1, prec, rec = accuracy(csv_path)
        #abre e salva em um arquivo
        with open(output_dir + csv.replace(".csv", ".txt"), 'w') as output_path:
            # print(csv, end="\n")
            output_path.write("Rede: {}\nAccuracy: {}\nF1 Score: {}\nPrecision Score: {}\nRecall Score: {}".format(csv,acc, f1, prec, rec))

if __name__ == "__main__":
    #4, 6, 8, 10, 12, 16, 20, 24
    hours = ["4", "6", "8", "10", "12", "16", "20", "24"]
    for hour in hours:
        #automatiza o processo
        main(hour)

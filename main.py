
import gzip # pour décompresser les données
import pickle 
import matplotlib.pyplot as plt # pour l'affichage
import torch,torch.utils.data
import numpy as np

if __name__ == '__main__':

    # nombre d'image lues à chaque fois dans la base d'apprentissage (laisser à 1 sauf pour la question optionnelle sur les minibatchs)
    TRAIN_BATCH_SIZE = 1
    # on charge les données de la base MNIST
    data = pickle.load(gzip.open('mnist.pkl.gz'),encoding='latin1')
    # images de la base d'apprentissage
    train_data = torch.Tensor(data[0][0])
    # labels de la base d'apprentissage
    train_data_label = torch.Tensor(data[0][1])
    # images de la base de test
    test_data = torch.Tensor(data[1][0])
    # labels de la base de test
    test_data_label = torch.Tensor(data[1][1])
    # on crée la base de données d'apprentissage (pour torch)
    train_dataset = torch.utils.data.TensorDataset(train_data,train_data_label)
    # on crée la base de données de test (pour torch)
    test_dataset = torch.utils.data.TensorDataset(test_data,test_data_label)
    # on crée le lecteur de la base de données d'apprentissage (pour torch)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    # on crée le lecteur de la base de données de test (pour torch)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    weight = 0.5
    nb_step = 0.001
    w = np.full(785, weight)
    
    for d in range(len(train_data)):
        x = train_data[d].numpy()
        x = np.append(x, 1)
        y = np.dot(x, w)
        for index in range(len(train_data_label[d])):
            if train_data_label[d][index] == 1:
                break
        a = nb_step * (index - y)
        w += np.dot(a, x)

    # nbTrue = 0
    # for d in range(100):
    #     print(d)
    #     x = test_data[d].numpy()
    #     x = np.append(x, 1)
    #     y = np.dot(x, w)
    #     for index in range(len(test_data_label[d])):
    #         if train_data_label[d][index] == 1:
    #             break
    #     if index == round(y):
    #         nbTrue += 1
    #     print(y, "     ", index)
        # print(neuron_weight)
        # activity = np.full(784, 0)
        # for i in range(len(neuron_bias)):
        #     for j in range(len(neuron_weight[i])):
        #         activity[i] += neuron_weight[i][j] * train_data[d][i]
        # print(activity)
        # for i in range(len(neuron_bias)):
        #     for index in range(len(train_data_label[d])):
        #         if train_data_label[d][index] == 1:
        #             break
        #     for j in range(len(neuron_weight[i])):
        #         neuron_weight[i][j] += train_data[d][i] * nb_step * (index - activity[i])
            
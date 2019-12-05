# Valentin Comas 11500223
# Arnaud Bressot 11505990


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
    w = np.full((785, 10), weight)
    for d in range(len(train_data)):
        x = train_data[d].numpy()
        x = np.append(x, 1)
        y = np.dot(x, w)
        index = np.argmax(train_data_label[d])
        dl = train_data_label[d].numpy()
        a = nb_step * (dl - y)
        b = np.outer(a, x)
        w += np.transpose(b)

    nbCorrect = 0
    for d in range(len(test_data)):
        x = test_data[d].numpy()
        x = np.append(x, 1)
        y = np.dot(x, w)
        index = np.argmax(test_data_label[d])
        index2 = np.argmax(y)
        if index == index2:
            nbCorrect += 1
    print(nbCorrect/len(test_data) * 100,"%")
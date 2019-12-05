# Valentin Comas 11500223
# Arnaud Bressot 11505990


import gzip # pour décompresser les données
import pickle 
import matplotlib.pyplot as plt # pour l'affichage
import torch,torch.utils.data
import numpy as np
import math

if __name__ == '__main__':

    # nombre d'image lues à chaque fois dans la base d'apprentissage (laisser à 1 sauf pour la question optionnelle sur les minibatchs)
    TRAIN_BATCH_SIZE = 1
    # on charge les données de la base MNIST
    data = pickle.load(gzip.open('mnist_light.pkl.gz'),encoding='latin1')
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
    w_h = np.full((785, 50), weight)
    w_o = np.full((50, 10), weight)
    for d in range(len(train_data)):
        x = train_data[d].numpy()
        x = np.append(x, 1)

        s_h = np.dot(x, w_h)
        e_h = np.exp(-s_h)
        y_h = 1 / (1 + e_h)

        y_o = np.dot(y_h, w_o)


        dl = train_data_label[d].numpy()
        m_o = dl - y_o

        a1 = np.dot(y_h, np.transpose(1 - y_h))
        a2 = np.dot(w_o, np.transpose(m_o))
        m_h = np.dot(a1, a2)
        
        dw_o = nb_step * np.outer(m_o, y_h)
        w_o += np.transpose(dw_o)
        dw_h = nb_step * np.outer(m_h, x)
        w_h += np.transpose(dw_h)
    
    nbCorrect = 0
    for d in range(len(test_data)):
        x = train_data[d].numpy()
        x = np.append(x, 1)

        s_h = np.dot(x, w_h)
        e_h = np.exp(-s_h)
        y_h = 1 / (1 + e_h)

        y_o = np.dot(y_h, w_o)
        index = np.argmax(test_data_label[d])
        index2 = np.argmax(y_o)
        if index == index2:
            nbCorrect += 1
    print(nbCorrect/len(test_data) * 100,"%")
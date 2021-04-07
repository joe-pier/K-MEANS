import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.spatial import distance

##### data ##############################################################
center_1 = np.array([-10, 45])
center_2 = np.array([-5, 5])
center_3 = np.array([10, 1])
center_4 = np.array([30, -5])
#center_5 = np.array([40, 40])
#center_6 = np.array([10, 30])

data_1 = random.randint(1, 3) * np.random.randn(200, 2) + center_1
data_2 = random.randint(1, 3) * np.random.randn(200, 2) + center_2
data_3 = random.randint(1, 3) * np.random.randn(200, 2) + center_3
data_4 = random.randint(1, 3) * np.random.randn(200, 2) + center_4
#data_5 = random.randint(1, 3) * np.random.randn(200, 2) + center_5
#data_6 = random.randint(1, 3) * np.random.randn(200, 2) + center_6
#dati = (data_1, data_2, data_3, data_4, data_5, data_6)

dati = (data_1, data_2, data_3, data_4)

data = np.concatenate(dati, axis=0)

dataset = set()
for x in data:
    dataset.add(tuple(x))


###########################################################################


class kmeans():
    def __init__(self, K, D, iteration, verbose=False, pause=0.5):
        self.K = K
        self.iteration = iteration
        self.df = pd.DataFrame(D)
        self.D = D
        self.verbose = verbose
        self.pause = pause

        self.randomlist = set()
        self.random_df = pd.DataFrame

        self.points = set()
        self.points_df = pd.DataFrame

        self.avarages_df = pd.DataFrame

        self.fig, self.ax = plt.subplots()
        self.n_iter = 0

        self.centroids = set()

        self.datas = []

    def Kpoints(self):
        for i in range(0, self.K):
            x = random.randint(int(min(self.df[0])), int(max(self.df[0])))
            y = random.randint(int(min(self.df[1])), int(max(self.df[1])))
            self.randomlist.add((x, y))
            self.random_df = pd.DataFrame(self.randomlist)
        return self.randomlist

    def samepoint(self):
        for i in range(0, self.K):
            value = random.sample(self.D, 1)[0]
            # print(value)
            self.points.add(value)
            self.points_df = pd.DataFrame(self.points)
        return self.points

    def distance_(self, set):
        distances = []
        clusters = []
        for i in self.D:
            for k in set:
                dist = distance.euclidean(i, k)
                distances.append(dist)
            cluster = np.argmin(distances)
            clusters.append(cluster)
            distances = []
        self.df['cluster'] = clusters
        return self.df

    def update(self):
        avarages = set()

        datas_ = []
        # print(list(range(self.K)))

        for i in list(range(self.K)):
            # print(self.df[self.df['cluster'] == i])
            data = self.df[self.df['cluster'] == i]

            datas_.append(data)
            if data.empty:
                # riassegniamo in maniera casuale i centroidi che non hanno valori collegati
                data = pd.DataFrame({0: (random.randint(int(min(self.df[0])), int(max(self.df[0]))),
                                         random.randint(int(min(self.df[1])), int(max(self.df[1]))))})
            avarage = data.mean(axis=0)
            avarages.add(tuple(avarage.values[0:2]))

        if self.verbose == True:
            print(f'iteration {self.n_iter}: {avarages}')
        self.avarages_df = pd.DataFrame(avarages)

        self.datas = datas_
        return avarages

    def scatter(self, df_):
        plt.ion()

        cmap = cm.get_cmap('gist_rainbow')
        self.ax.scatter(self.df[0], self.df[1], cmap=cmap, marker='o', c=self.df['cluster'])
        self.ax.scatter(df_[0], df_[1], marker='x', color='black')

        self.ax.grid(True)
        plt.pause(self.pause)
        self.ax.clear()

    def fit(self, plot=False):
        '''
        #per una scelta completamente casuale all'inizio
        randompoints = self.Kpoints()
        self.distance_(randompoints)
        if plot == True:
            self.scatter(self.random_df)
        '''

        # le convenzioni di K-means voglio che si usi questa modalità
        samepoints_ = self.samepoint()
        self.distance_(samepoints_)
        if plot == True:
            self.scatter(self.points_df)
        ##############################################################

        for i in range(0, self.iteration):
            self.n_iter = self.n_iter + 1
            new_values = self.update()
            self.distance_(new_values)
            if plot == True:
                self.scatter(self.avarages_df)

        if plot == True:
            plt.show()

        self.centroids = list(new_values)

        dic = {}
        for m, h in enumerate(self.centroids):
            dic[m] = h
        centroid_df = pd.DataFrame(dic, index=None)
        centroid_df = centroid_df[0:3].round(decimals=2)
        centroid_df = centroid_df.add_prefix('centroid ')
        centroid_df = centroid_df.rename(index={0: 'X', 1: 'Y'})
        centroid_df.to_csv('centroids.csv', header=True, index=True)
        return centroid_df

    def predict(self, X):

        predictions = []
        for j in X:
            temp_2 = []
            for i in self.centroids:
                dist_temp = distance.euclidean(i, j)
                temp_2.append(dist_temp)
            prediction = np.argmin(temp_2)
            predictions.append(prediction)
        d = {}
        for n, x in enumerate(X):
            d[x] = predictions[n]
        series = pd.Series(d).to_frame()
        df = pd.DataFrame(series)
        df = df.rename(columns={0: 'predicted cluster'}, errors="raise")


        df.to_csv('predictions.csv', header=True, index=True)


        return df


    def cost(self):
        for i in self.datas:
            temp = []
            for j in range(len(i)):
                for k in self.centroids:
                    miao = i.values[j][0:2]
                    distance_ = distance.euclidean(miao, k)
                    squrt = distance_ ** 2
                    temp.append(squrt)
            cost = sum(temp) / len(self.D)
        return cost



if __name__ == "__main__":
    test = kmeans(5, dataset, 20, verbose=False, pause=0.2)  # definisco il modello e inserisco i dati
    centroid = test.fit(plot=False)  # fitto il modello e creo i centroidi

    '''
    costo = test.cost()
    print(costo)
    '''

    p_test = test.predict({(-10, 45), (-5, 5), (30, -5), (9, 9), (90, 21), (-11, 43)})  # prova del modello con dati a caso
    print('\n')
    print(centroid)  # print dei centroidi
    print('\n')
    print(p_test)  # print delle previsioni


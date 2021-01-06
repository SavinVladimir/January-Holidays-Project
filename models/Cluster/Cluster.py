import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

sns.set_theme(style="darkgrid")
sns.set(rc={'figure.figsize': (15, 10)})

data = pd.read_csv('../../data/Mall_Customers.csv')

data = data.rename(columns={'Annual Income (k$)': 'Annual_income', 'Spending Score (1-100)': 'Spending_score'})

x = data.loc[:, ['Annual_income', 'Spending_score']].values


s = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++').fit(x)

    s.append(kmeans.inertia_)


def clustering():

    global kmeans

    model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    y_model = model.fit_predict(x)

    plt.scatter(x[y_model == 0, 0], x[y_model == 0, 1], s=100, c='pink', label='Не целевая аудитория')
    plt.scatter(x[y_model == 1, 0], x[y_model == 1, 1], s=100, c='yellow', label='Основной')
    plt.scatter(x[y_model == 2, 0], x[y_model == 2, 1], s=100, c='cyan', label='Цель')
    plt.scatter(x[y_model == 3, 0], x[y_model == 3, 1], s=100, c='magenta', label='Расточитель')
    plt.scatter(x[y_model == 4, 0], x[y_model == 4, 1], s=100, c='orange', label='Бережный')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='blue', label='Центр')

    plt.title('Clustering', fontsize=20)
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.legend()
    plt.show()

clustering()







import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

sns.set_theme(style="darkgrid")

data = pd.read_csv('../../data/Mall_Customers.csv')


def preprocessing():
    global data
    data = data.rename(columns={'Annual Income (k$)': 'Annual_income', 'Spending Score (1-100)': 'Spending_score'})

preprocessing()

x = data.iloc[:, [3, 4]].values


def model():
    global x
    s = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        km.fit(x)
        s.append(km.inertia_)

    plt.plot(range(1, 11), s)
    plt.show()

model()

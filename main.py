import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize': (15, 10)})

data = pd.read_csv('data/Mall_Customers.csv')


def preprocessing():
    global data
    data = data.rename(columns={'Annual Income (k$)': 'Annual_income', 'Spending Score (1-100)': 'Spending_score'})

preprocessing()


def pair():
    sns.pairplot(data)
    plt.show()
pair()


def heatmap():
    sns.heatmap(data.corr(), annot=True)
    plt.show()

heatmap()

print(data.head())
print('----------------------------------------------------------')
print(data.dtypes)
print('----------------------------------------------------------')
print(data.describe())





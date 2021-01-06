import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

sns.set_theme(style="darkgrid")

data = pd.read_csv('../../data/Mall_Customers.csv')


def preprocessing():
    global data
    data = data.rename(columns={'Annual Income (k$)': 'Annual_income', 'Spending Score (1-100)': 'Spending_score'})

preprocessing()


def models():
    x = np.array(data['Age'])
    y = np.array(data['Spending_score'])

    x_test = np.array([8, 25, 43])

    model = LinearRegression()

    model.fit(x.reshape(-1, 1), y)
    model.predict(x_test.reshape(-1, 1))

    plt.figure(figsize=(15, 10))

    plt.scatter(x, y, alpha=0.7, marker='x')
    plt.plot(x, model.coef_[0] * x + model.intercept_, color='red')

    plt.title('Linear model')
    plt.xlabel('Age')
    plt.ylabel('Spending score')

    plt.show()


models()


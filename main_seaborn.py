from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


import pandas as pd # Manejo de datos
import seaborn as sns # Creación de gráficas y visualización de datos
import numpy as np
import matplotlib.pyplot as plt


def main():

    df = pd.read_csv('./studentsperformance.csv')
    #print(df.head())

    X = df['reading score'].values
    Y = df['writing score'].values

    X = X.reshape(-1,1)

    #Dividir los datos para entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    #cargar elmodelo de regresion lineal
    reg = LinearRegression()
    reg.fit(X_train, Y_train)
    
    x_flat = X_train.flatten()

    y_hat = reg.predict(X_train)

    sns.scatterplot(x=x_flat, y=Y_train)
    sns.lineplot(x=x_flat, y=y_hat, color='r')

    #Evaluacion del modelo

    y_pred = reg.predict(X_test)

    #calculo el error caudratico medio
    print(mean_squared_error(Y_test, y_pred))

    print('La pendiente es: ', reg.coef_)
    print('El bias es: ', reg.intercept_)
    print('Coeficiente de determinacion: ', reg.score(X_train, Y_train))
    
    value = pd.DataFrame({'Actual test': Y_test.flatten(), 'predict':y_pred.flatten()})
    print()
    print(value)
    print('error caudratico medio: ',mean_squared_error(Y_test, y_pred))

    plt.show()

if __name__ == '__main__':
    main()



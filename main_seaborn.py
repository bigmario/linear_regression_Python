from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


import pandas as pd # Manejo de datos
import seaborn as sns # Creación de gráficas y visualización de datos
import numpy as np
import matplotlib.pyplot as plt

def estimate_b0_b1(x, y):
    n = np.size(x)
    

    #Promedios
    mean_x, mean_y = np.mean(x), np.mean(y)

    #sumatorias
    sumatorias_XY = np.sum((x - mean_x)*(y - mean_y))
    sumatorias_XX = np.sum((x - mean_x)**2)

    
    #Coeficientes de regresion
    b1 = sumatorias_XY / sumatorias_XX
    b0 = mean_y - b1 * mean_x

    return (b0, b1)

def plot_regresion(data, X, Y, b):
    sns.scatterplot(data=data, x=X, y=Y)
    
    prediccion_Y = b[0] + b[1] * X
    
    sns.lineplot(data=data, x=X, y=Y)

    plt.show()
    

def main():
    reg = LinearRegression()
    reg.fit
    
    df = pd.read_csv('./studentsperformance.csv')
    print(df.head())

    #Dividir los datos para entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    #cargar elmodelo de regresion lineal
    reg = LinearRegression()
    reg.fit

    X = df['reading score'].values
    Y = df['writing score'].values

    b = estimate_b0_b1(X,Y)

    print(f'Ĺos valores de b0 = {b[0]} y b1 = {b[1]}')

    plot_regresion(df, X, Y, b)

if __name__ == '__main__':
    main()



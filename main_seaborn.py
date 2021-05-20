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
    df = pd.read_csv('./studentsperformance.csv')
    print(df.head())

    X = df['reading score']
    Y = df['writing score']

    b = estimate_b0_b1(X,Y)

    print(f'Ĺos valores de b0 = {b[0]} y b1 = {b[1]}')

    plot_regresion(df, X, Y, b)

if __name__ == '__main__':
    main()



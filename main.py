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

def plot_regresion(X, Y, b):
    plt.scatter (X,Y,color='b', marker= 'o', s= 30)
    prediccion_Y = b[0] + b[1] * X
    plt.plot(X, prediccion_Y, color='r')

    #leyenda
    plt.xlabel('X Independiente (Dominio)')
    plt.ylabel('Y Dependiente (Rango / Imagen)')

    plt.show()
    

def main():
    X = np.array([1,2,3,4,5])
    Y = np.array([2,3,5,6,5])

    b = estimate_b0_b1(X,Y)

    print(f'Ĺos valores de b0 = {b[0]} y b1 = {b[1]}')

    plot_regresion(X,Y,b)

if __name__ == '__main__':
    main()



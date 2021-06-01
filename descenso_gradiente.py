from matplotlib import cm # Para manejar colores
import numpy as np
import matplotlib.pyplot as plt

def f(x,y):
    return x**2 + y**2

def derivate(_p,p,h):
    return  (f(_p[0],_p[1]) - f(p[0],p[1])) / h

def run():
    '''
    Gráfica en 3D de nuestra función de coste
    '''
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    res = 100

    X = np.linspace(-4, 4, res)
    Y = np.linspace(-4, 4, res)

    X, Y = np.meshgrid(X, Y)

    Z = f(X,Y) 

    # Gráficar la superficie
    surf = ax.plot_surface(X, Y, Z, cmap=cm.cool,
                        linewidth=0, antialiased=False)

    fig.colorbar(surf)

    '''
    Descenso del gradiente
    '''
    fig2, ax2 = plt.subplots()

    level_map = np.linspace(np.min(Z), np.max(Z),res) 
    cp = ax2.contourf(X, Y, Z, levels=level_map,cmap=cm.cool)
    fig2.colorbar(cp)
    plt.title('Descenso del gradiente')
    

    p = np.random.rand(2) * 8 - 4 # generar dos valores aleatorios

    plt.plot(p[0],p[1],'o', c='k')

    lr = 0.01
    h = 0.01

    grad = np.zeros(2)

    for i in range(10000):
        for idx, val in enumerate(p): 
            _p = np.copy(p)

            _p[idx] = _p[idx] + h

            dp = derivate(_p,p,h) 

            grad[idx] = dp

        p = p - lr * grad

        if(i % 10 == 0):
            plt.plot(p[0],p[1],'o', c='r')

    plt.plot(p[0],p[1],'o', c='w')
    plt.show()

    print("El punto mínimo se encuentra en: ", p)

if __name__ == '__main__':
    run()
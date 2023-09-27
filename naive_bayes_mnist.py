########################################
#Hallamos los parámetros mu y sigma para cada imagen a partir de los datos de entrenamiento
import numpy as np

def estadisticos(x,y):
    dim=x.shape
    nums=np.unique(y)
    av=np.zeros((len(nums),dim[1],dim[2]))
    sigma=np.zeros((len(nums),dim[1],dim[2]))
    for i in np.unique(y):
    #     s+=len(x_train[y_train==i])
        av[i]=np.mean(x[y==i], axis=0) # Números promedio del training (por pixel)
        sigma[i]=np.std(x[y==i], axis=0) # desviaciones del training(por pixel) 
    return av, sigma
##########################################
# Método naive bayes para encontrar las características
def nb_mnist(x_train,y_train,x_test,offset=10):
    av,sigma=estadisticos(x_train,y_train)
    nums=np.unique(y_train)
    log_p=np.zeros((len(x_test),len(nums))) # diez probabilidades por cada valor de testeo (una por cada número)
    if(len(np.shape(x_test))>2): #numero de elementos a ingresar en el test
        n=len(x_test)
        y_pred=np.zeros(len(x_test))
        max_p=np.zeros(len(x_test))
    else:
    	n=1
    	y_pred=[0]
    	max_p=[0]
    # la resta de la gaussiana se hace con el x_test porque son los valores que queremos predecir
    # mientras que las medias y las sigmas las hacemos con los datos de entrenamiento
    for i in range(n):
        for l in nums:
            s_off=sigma[l]+offset
            A=np.sqrt(2*np.pi*s_off**2)
            log_p[i,l]=np.sum(np.log(np.exp(-(x_test[i]-av[l])**2/(2*s_off**2))/A))
#             prob[i,l]=np.prod(np.exp(-(x_test[i]-av[l])**2/(2*s_off**2))/A)
        y_pred[i]=np.argmax(log_p[i])
        max_p[i]=np.exp(log_p[i,int(y_pred[i])])
    return y_pred,max_p

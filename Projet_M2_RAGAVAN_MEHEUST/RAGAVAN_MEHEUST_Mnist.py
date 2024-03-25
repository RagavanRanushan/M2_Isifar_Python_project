import numpy as np
from scipy import special

rng = np.random.default_rng()  # création du générateur de nombres aléatoires



#-------------------- inport de l'image-------------------------
dt = np.dtype('uint32')
dt = dt.newbyteorder('>')  # big-endian, à commenter si besoin

f = open('/Users/williammeheust/Desktop/M2_ISIFAR/Semestre_1/PYTHON/PROJET/train-images-idx3-ubyte', mode = 'rb')
x = f.read(16)
y = np.frombuffer(x, dtype = dt)
truc, nb_im, nb_rows, nb_cols = y
x = f.read(nb_rows*nb_cols*nb_im)
f.close()
train_images = np.frombuffer(x, dtype = 'ubyte').reshape(nb_im, nb_rows, nb_cols)

f = open('/Users/williammeheust/Desktop/M2_ISIFAR/Semestre_1/PYTHON/PROJET/train-labels-idx1-ubyte', mode = 'rb')
x = f.read(8)
y = np.frombuffer(x, dtype = dt)
truc, nb_exemples = y
x = f.read(nb_exemples)
f.close()
train_labels = np.frombuffer(x, dtype = 'ubyte')

f = open('/Users/williammeheust/Desktop/M2_ISIFAR/Semestre_1/PYTHON/PROJET/t10k-images-idx3-ubyte', mode = 'rb')
x = f.read(16)
y = np.frombuffer(x, dtype = dt)
truc, nb_im, nb_rows, nb_cols = y
x = f.read(nb_rows*nb_cols*nb_im)
f.close()
test_images = np.frombuffer(x, dtype = 'ubyte').reshape(nb_im, nb_rows, nb_cols)

f = open('/Users/williammeheust/Desktop/M2_ISIFAR/Semestre_1/PYTHON/PROJET/t10k-labels-idx1-ubyte', mode = 'rb')
x = f.read(8)
y = np.frombuffer(x, dtype = dt)
truc, nb_exemples = y
x = f.read(nb_exemples)
f.close()
test_labels = np.frombuffer(x, dtype = 'ubyte')


#----------------------Calcul de l'entropie--------------
def D(P,Q):
    r = 0
    x = np.sum(Q.T, axis=0)
    
    
    for i in range(len(P)):
        if P[i] != 0:
            r += P[i]*np.log((P[i]/x[i]))
    
    return r

#------------------------------------------------------------
class Rn:
    def __init__(self,shape, sigma):
        self.shape = shape

        self.b = [ np.zeros(n) for n in shape[1:] ]
        self.a = [ (rng.random((shape[ell+1],shape[ell])) - .5) * np.sqrt( 24/shape[ell] ) for ell in range(len(shape)-1) ]
        self.a[0] /= sigma
        self.a[-1] /= np.sqrt(2)

    def __str__(self):
        msb = [ (b.mean(), b.std()) for b in self.b]
        msa = [(a.mean(), a.std()) for a in self.a]
        return str(msb) + "\n" + str(msa)

    def Copy(self):
        rn = Rn(self.shape,1)
        rn.b = [ x.copy() for x in self.b ]
        rn.a =[x.copy() for x in self.a ]
        return rn

    def CalculSortie(self, X):
        for ell in range(len(self.shape)-2):
            X[ell+1][:] = np.maximum( 0,self.b[ell][:, np.newaxis] + self.a[ell] @ X[ell] )
        X[-1] = self.b[-1][:, np.newaxis] + self.a[-1] @ X[-2]

    def Retro(self, X, grad):
        for ell in range(len(self.shape)-2,-1,-1):
            aux = np.copy(grad)
            if ell < (len(self.shape)-2):
                grad *= ( X[ell+1]>0 )
            self.b[ell] -= grad.mean( axis = 1 )
            self.a[ell] -= (grad[:,np.newaxis,:] * X[ell][np.newaxis,:,:]).mean(axis = 2)
            grad = (self.a[ell][:,:,np.newaxis]*grad[:,np.newaxis,:]).sum(axis = 0)


if __name__ == "__main__":

    def initialisation():
        m = 1
        X = [ np.zeros(shape = (n, m)) for n in shape ]

        for rep in range(5):
            Y = [ np.zeros(shape = (n, 1)) for n in shape ]
            for i in range(10000):
                theta = np.zeros(10)
                theta[train_labels[i]] = 1 #loi de theta
                X[0][:] = train_images[i].reshape(784,1)
                rn.CalculSortie(X)
                for j in range(len(Y)):
                    Y[j] += np.mean(X[j]**2, axis = 1)[:,np.newaxis]
            res = [  np.mean(y)/10000 for y in Y]
 
            for j in range(len(shape)-1):
                rn.a[j] /= np.sqrt(res[j+1])

    n0 = 20
    shape = (28*28,10)
    rn = Rn( shape , np.sqrt(2))
    initialisation()



    def test2():
        global rn
        m = 1
        X = [ np.zeros(shape = (n, m)) for n in shape ]

        nb_epi = 10
        taille_epi = 60000
        pas = 1e-8
        

        for rep in range(nb_epi):
            s = 0.0
            W = 0
            for i in range(taille_epi): 
                theta = np.zeros(10)
                theta[train_labels[i]] = 1 #loi de theta
                X[0][:] = train_images[i].reshape(784,1)
                rn.CalculSortie(X)
                proba = special.softmax(X[-1])
                entropy = D(theta,proba)
                s += entropy
                if np.argmax(proba) == np.argmax(theta):
                    W += 1

                grad = pas*(proba - theta[:,np.newaxis])
        
                
                rn.Retro(X, grad)

        total = 0
        for j in range(0,test_images.shape[0],m):
            img_shp = test_images[j:j+m].shape
            X[0][:] = test_images[j:j+m].reshape(img_shp[0], img_shp[1]*img_shp[2]).transpose()
                #on envoie m image dans le reseau et on calcul la sortie
            rn.CalculSortie(X)
                
            g_alpha = special.softmax(X[-1].transpose())
                #on compte le nombre de fois ou le reseau donne le bon label en sortie 
            count = (g_alpha.argmax(axis=1) == test_labels[j:j+m]).sum(dtype=int)
            total += count 
                
        print(("Perfomance sur les images test :" +str(total/test_images.shape[0])))

    test2()
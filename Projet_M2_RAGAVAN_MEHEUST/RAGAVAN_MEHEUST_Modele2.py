import numpy as np
rng = np.random.default_rng()  # création du générateur de nombres aléatoires

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
            self.b[ell] -= grad.mean( axis=1 )
            self.a[ell] -= (grad[:,np.newaxis,:] * X[ell][np.newaxis,:,:]).mean(axis =2)
            grad = (self.a[ell][:,:,np.newaxis]*grad[:,np.newaxis,:]).sum(axis=0)

if __name__ == "__main__":

    def initialisation():
        m = 1
        X = [ np.zeros(shape = (n, m)) for n in shape ]

        for rep in range(5):
            Y = [ np.zeros(shape = (n, 1)) for n in shape ]
            for i in range(10000):
                theta = rng.uniform(-np.pi,np.pi,size = m) #loi de theta
                X[0][:] = theta[np.newaxis,:] + rng.normal(size = (n0,m))
                rn.CalculSortie(X)
                for j in range(len(Y)):
                    Y[j] += np.mean(X[j]**2, axis = 1)[:,np.newaxis]
            res = [  np.mean(y)/10000 for y in Y]
            for j in range(len(shape)-1):
                rn.a[j] /= np.sqrt(res[j+1])

    n0 = 20
    shape = (n0,n0,n0,2)
    rn = Rn( shape , np.sqrt(2))
    initialisation()


    def test2():
        global rn
        m = 1
        
        X = [ np.zeros(shape = (n, m)) for n in shape ]
        
        taille_epi = 5000
        nb_epi = 50
        pas = 1e-10

        for rep in range(nb_epi):
            s = 0.0
            for i in range(taille_epi):
                
                theta = rng.uniform(-np.pi,np.pi,size = m) #loi de theta
                
                X[0][:] = theta[np.newaxis,:] + rng.normal(size = (n0,m))
            
                rn.CalculSortie(X)

                X_bar = X[0].mean()        
                
                F = np.exp(-1/(2*n0))*np.array([np.cos(X_bar),np.sin(X_bar)]).reshape(2,1) #dans le cas m=1
                
                G = np.array([np.cos(theta),np.sin(theta)])
        
                s += (np.linalg.norm((F - G))**2).mean() 
                
                grad = pas*(F - G) 
            
                rn.Retro(X, grad)

            print(f"Episode {rep}  pas ={pas}")
            print('risque=', s / taille_epi, 'risque optimal =', 1-np.exp(-1/n0))
            print(rn)
            print("\n")

    test2()
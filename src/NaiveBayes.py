import numpy as np

class NaiveBayes:
    def __init__(self,xlable,alpha=1):
        #self.X = X
        #self.y = y
        self.xlable = np.array(xlable)
        #self.C=len( np.unique(self.y))
        #self.N = X.shape[0]
        self.alpha=alpha

 
    def modelInformation(self):
        unique, self.y_prior = np.unique(self.y, return_counts=True)
        self.prior = np.log(self.y_prior/self.N)[:,None]

    def prior_prob(self):
        unique, self.y_prior = np.unique(self.y, return_counts=True)
        self.prior = np.log(self.y_prior/self.N)[:,None]

    
    def get_Guassian_parameters (self):
        self.ind_Guasssin=np.where(self.xlable == 1)[1]
        if self.ind_Guasssin.size==0:
            print("")
        else:
            X_guassin=self.X[:, self.ind_Guasssin]
            
            self.mu=np.zeros((len(np.unique(self.y)),len(self.ind_Guasssin)))
            self.std=np.zeros((len(np.unique(self.y)),len(self.ind_Guasssin)))
            for r in range (0,len(np.unique(self.y))):
                ind=np.where(self.y == r)[0]
                zz=X_guassin[ind, :]
                self.mu[r , :] = np.mean(zz, 0)
                self.std[r, :] = np.std(zz, 0) 
    def get_Guassian_log_liklihood (self,Xtest):
        
        if self.ind_Guasssin.size == 0:
            self.Guassian_liklihood=np.zeros((self.C,Xtest.shape[0])) 
        else:
            #n_features = Xtest.shape[1]
            pfc = np.ones((self.C,Xtest.shape[0]))
            for k in range(0, Xtest.shape[0]):
                for i in range(0, self.C):
                    sum_ = 0
                    for j in range(0, Xtest.shape[1]):
                        sum_ = sum_ -np.log( self.std[i,j]+.00001)- .5* ((Xtest[k,j] - self.mu[i,j])/(self.std[i,j]))**2
                        pfc[i,k] = sum_
            
            self.Guassian_liklihood=pfc
           
            
            #X_guassin=Xtest[:,self.ind_Guasssin]    
            #self.Guassian_liklihood = - np.sum( np.log (self.std[:,None,:]) + .5*(((X_guassin[None,:,:]-self.mu[:,None,:])/(self.std[:,None,:]))**2),2)
    #def get_multinomial_parameters(self):
    
        
    #def get_multinomial_liklihood(self,Xtest):
    def get_Berrnolly_parameters(self):
        self.ind_Berrnolly=np.where(self.xlable == 0)[1]
        

        if self.ind_Berrnolly.size == 0:
            print("") 
        else:
            X_Berrnolly=self.X[:, self.ind_Berrnolly]
            self.Berrnolly_liklihood=np.zeros((len(np.unique(self.y)),len(self.ind_Berrnolly)))
            for r in range (len(np.unique(self.y))):   
                ind=np.where(self.y == r)[0]
                
                self.Berrnolly_liklihood[r , :] = np.sum(X_Berrnolly[ind, :], 0)/ind.shape[0] #+ self.alpha
            #print(self.Berrnolly_liklihood)
                # separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(self.y)]
                #self.log_prior = [np.log(len(i) / countSample) for i in separated]
                #count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
                #self.log_likelihood = np.log(count / count.sum(axis=1)[np.newaxis].T)
        
    def get_Berrnolly_log_liklihood(self,Xtest):
        ind_Berrnolly=np.where(self.xlable == 0)[1]
        self.Berrnolly_log_liklihood=np.zeros((self.C,Xtest.shape[0]))
        if ind_Berrnolly.size == 0:
            print("")  
        else:
            X_Berrnolly=Xtest[:,ind_Berrnolly]
            for rowIndex in range(X_Berrnolly.shape[0]):
                temp=X_Berrnolly[rowIndex,:]
                self.Berrnolly_log_liklihood[:,rowIndex]=np.dot(np.log1p(1-self.Berrnolly_liklihood),temp) + np.dot(np.log1p(self.Berrnolly_liklihood) ,(1 - temp))
    
    def fit(self,X,y):
        self.X = X
        self.y = y
        self.C=len( np.unique(self.y))
        self.N = X.shape[0]
        #self.get_multinomial_parameters (self,Xtest)
        self.prior_prob()
        self.get_Guassian_parameters ()
        self.get_Berrnolly_parameters ()

    def predict(self, Xtest):
        self.get_Guassian_log_liklihood(Xtest)
        self.get_Berrnolly_log_liklihood (Xtest)
        self.liklihood=  self.prior +self.Guassian_liklihood+self.Berrnolly_log_liklihood
       
        for i in range(Xtest.shape[0]):
            """
            self.get_Guassian_log_liklihood (Xtest[i,:])
            self.get_Berrnolly_log_liklihood (Xtest[i,:])
            self.get_multinomial_liklihood (self,Xtest)
            """
            #+self.multinomial_liklihood
            self.liklihood[:,i] -= np.max(self.liklihood[:,i]) 
            self.liklihood[:,i] = np.exp(self.liklihood[:,i])
            self.liklihood[:,i] /= np.sum(self.liklihood[:,i])   
        return self.liklihood.argmax(axis=0)
    
    def evaluate_acc(self,y_test, yh):
        acc=(len(y_test)-np.sum(np.absolute(y_test.T-yh.T)))/len(y_test)
        err=1-acc
        return  acc,err


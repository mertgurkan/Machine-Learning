import numpy as np
import sys 

class LogisticRegression:

    def __init__(self, lr=0.01, LxReg=1, num_iters=1000, eps=1e-9):
        """
        Constructor for the logistic regression class with regularization

        :param lr: learning rate
        :param LxReg: The lambda or regularization factor
        :param num_iters: max number of iterations for stability
        :param eps: threshold to stop the iterations of gradient descent
        """



        self.num_iters = num_iters
        self.lr = lr
        self.LxReg = LxReg
        self.eps = eps
        self.h_cost = np.zeros((self.num_iters, 1))
        
        

    @staticmethod
    def sigmoid(t):
        """
        Calculate the sigmoid of the respective scalar
        :param t: w^t.x_i
        :return: sigmoid(t)
        """
        return 1/(1+np.exp(-t))
    def gradient(self):

        yh = self.sigmoid(np.matmul(self.X, self.w))
        return np.matmul(self.X.T, yh-self.y) / self.N

        #z = y - self.sigmoid(np.matmul(x, w_init))
            #w = w_init + ((alpha / n) * (np.matmul(x.T, z))) + ((self.reg_val/n)*w_init)
    def fit(self,X, y):
        """
        This method is used to train (fit) the weight of the logistic regression model
        :param X:  training features data
        :param y: classification training data
        """
        self.X = X
        self.y = y
        self.N, self.D = np.shape(X)
        self.w = np.random.rand(self.D, 1)  # random weights initialization
        #self.w_final = np.zeros((self.D, 1))
        self.w_itration = np.empty([self.D,0])
        cost=np.inf
        
        i=0
        while np.linalg.norm(cost)>self.eps and i<self.num_iters: #1000 to avoide infinit loop
            
            grad = self.gradient()
            self.w = self.w - self.lr*grad+self.LxReg*self.w/self.N
            cost=abs(cost-self.cost())
            self.h_cost[i]= self.cost()
            
            
            cost=abs(self.h_cost[i]-cost)

            self.w_itration=np.concatenate((self.w_itration,self.w),axis=1)
            #self.w_itration=np.c_[ self.w_itration, self.w]
            #yh=predict(self, self.X_test)
            #acc,err=evaluate_acc(self.y_test, yh)
            #self.acc[i]= acc
            #self.err[i]= err
            i += 1
        self.w_final = self.w
        
        while i<self.num_iters-1:# for the aim of coparison 
            self.w_itration=np.concatenate((self.w_itration,self.w),axis=1)
        i += 1          
        
    def set_w_final(self,w):
        self.w_final=w
    def get_w_itration(self,i):
        return self.w_itration[:,i-2]
        

    def predict(self, X_test):
        """
        This method is used to predict the classification of the test data or new acquired features
        :param X_test: features test or newly acquired
        :return: the predicted classification of each set of features
        """
        
        yh=np.around(self.sigmoid(np.matmul(X_test, self.w_final)))    
        return  yh

    def cost(self):
        """
        This method is used to calculate the cost function for each iteration and record it

        """ 
        z = np.around(self.sigmoid(np.matmul(self.X, self.w)))
        #cost=(1/self.N) * ((np.matmul((-self.y), np.log1p(z )))-(np.matmul((1-self.y), np.log1p(1-z))))+((self.LxReg/2)*(np.linalg.norm(self.w, ord=2)**2))
        cost=-np.mean(np.matmul(self.y.T , np.log1p(np.exp(-z))) + np.matmul((1-self.y).T , np.log1p(np.exp(z))))+((self.LxReg/2)*(np.linalg.norm(self.w, ord=2)**2))
        #print(cost)
        return cost
    
    def evaluate_acc(self,y_test, yh):
        
    
        acc=(len(y_test)-np.sum(np.absolute(y_test.T-yh.T)))/len(y_test)
        err=1-acc
        return  acc,err
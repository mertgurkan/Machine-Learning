import numpy as np
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt
import preProcessing  as prePros
import LogisticRegression as LR
import modelEvaluation as ME
import matplotlib.patches as mpatches

def plot_exp_2(avg_acc_models,lr_,LxReg_,eps_,num_iters,img_name):
    index= -1
    plt.figure(figsize=(800/90, 800/90), dpi=90)
    for i in range(0,len(lr_)):
            for j in range(0,len(LxReg_)): 
                for k in range(0,len(eps_)):
                    index +=1
                    plt.plot(np.linspace(10, num_iters, num=num_iters-10),avg_acc_models[index,9:-1], label= r"lr=" + str(lr_[i])+r" $\lambda$="+ str(LxReg_[j]))    
    #plt.title("Convergence Graph of Accurecy")
    #plt.figure(figsize=(16,4))
    plt.title("Convergence Graph of Accuracy")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.savefig("../img/experment2/%s.png" % img_name)
    plt.show() 
    plt.close()
    print("Created ../img/experment2/%s" % img_name)
    


def exp_2_ionosphere():
    output='y'
    df=prePros.get_ionosphere_data()
    df_test= pd.DataFrame(None)
    
    # Preparing data
    binary_columns=["Re1","Im1"]
    df=prePros.get_cleaned_data_ionosphere(df, [''],binary_columns,corr_limite=.1, corr_flag=True,limite=.8,delete_outliers=False)
    x_headers,y_header=prePros.define_variables(df,output)
    X_train1, y_train, X_test1, y_test=prePros.Data_spliting(df,df_test, x_headers,y_header, training_percent=0.85, shuffle=True, random_seed_value=42)
    #prepare data for LR
    X_train,X_test=prePros.prepXforLR(X_train1,X_test1)
    
    # define the machine learning model=ml_model

    num_iters=2000
    avg_acc_models=np.empty([0,num_iters])
    std_acc_models=np.empty([0,num_iters])
    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    eps_=np.array([ .0001])
    #eps_=eps_[:,None,None]
    lr_=np.array([.1,.01])
    #lr_=lr_[None,:,None]
    LxReg_=np.array([0,.01])
    #LxReg_=LxReg_[None,None,:]
    for i in range(0,len(lr_)):
        for j in range(0,len(LxReg_)): 
            for k in range(0,len(eps_)):
                eps  =eps_[k]
                lr=lr_[i]
                LxReg  =LxReg_[j]
                ml_model=LR.LogisticRegression(lr, LxReg, num_iters, eps)
    # append results for comparison 
                avg_acc,std_acc=ME.model_selection(X_train, y_train, ml_model,Normalization=True,FinalModel=False, k=5)
                
                avg_acc=avg_acc[None,:]
                std_acc=std_acc[None,:]
                
                avg_acc_models = np.append(avg_acc_models, avg_acc, axis=0)
                std_acc_models = np.append(std_acc_models, std_acc, axis=0)

    
    #ploting results experment 2
    img_name=["Experment_2_ionosphere"]
    plot_exp_2(avg_acc_models,lr_,LxReg_,eps_,num_iters,img_name)    
    
    
    
    
def exp_2_parkinsons():
    output='y'
    df=prePros.get_parkinsons_data()
    df_test= pd.DataFrame(None)
    
    df=prePros.get_cleaned_data_Parkinsons(df, [''],[''],corr_limite=.1, corr_flag=True,limite=.8,delete_outliers=False)
    x_headers,y_header=prePros.define_variables(df,output)
    X_train1, y_train, X_test1, y_test=prePros.Data_spliting(df,df_test, x_headers,y_header, training_percent=0.85, shuffle=True, random_seed_value=42)
    #prepare data for LR
    # Preparing data
    #binary_columns=["Re1","Im1"]

    X_train,X_test=prePros.prepXforLR(X_train1,X_test1)
    
    # define the machine learning model=ml_model

    num_iters=2000
    avg_acc_models=np.empty([0,num_iters])
    std_acc_models=np.empty([0,num_iters])
    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    eps_=np.array([ .00001])
    #eps_=eps_[:,None,None]
    lr_=np.array([.0001,.001 ])
    #lr_=lr_[None,:,None]
    LxReg_=np.array([0,.1])
    #LxReg_=LxReg_[None,None,:]
    for i in range(0,len(lr_)):
        for j in range(0,len(LxReg_)): 
            for k in range(0,len(eps_)):
                eps  =eps_[k]
                lr=lr_[i]
                LxReg  =LxReg_[j]
                ml_model=LR.LogisticRegression(lr, LxReg, num_iters, eps)
    # Model_selection
                avg_acc,std_acc=ME.model_selection(X_train, y_train, ml_model,Normalization=True,FinalModel=False, k=5)

                
                avg_acc=avg_acc[None,:]
                std_acc=std_acc[None,:]
                
                avg_acc_models = np.append(avg_acc_models, avg_acc, axis=0)
                std_acc_models = np.append(std_acc_models, std_acc, axis=0)
    img_name=["Experment_2_Parkinsons"]
    plot_exp_2(avg_acc_models,lr_,LxReg_,eps_,num_iters,img_name) 

    
def exp_2_Iris():
    output='y'
    df=prePros.get_Iris_data()
    df_test= pd.DataFrame(None)
    
    df=prePros.get_cleaned_data_Iris(df, [''],[''],corr_limite=.1, corr_flag=True,limite=.8,delete_outliers=False)
    x_headers,y_header=prePros.define_variables(df,output)
    X_train1, y_train, X_test1, y_test=prePros.Data_spliting(df,df_test, x_headers,y_header, training_percent=0.85, shuffle=True, random_seed_value=42)
    #prepare data for LR
    # Preparing data
    #binary_columns=["Re1","Im1"]

    X_train,X_test=prePros.prepXforLR(X_train1,X_test1)
    
    # define the machine learning model=ml_model

    num_iters=400
    avg_acc_models=np.empty([0,num_iters])
    std_acc_models=np.empty([0,num_iters])
    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    eps_=np.array([ .00001])
    #eps_=eps_[:,None,None]
    lr_=np.array([.01,.05 ])
    #lr_=lr_[None,:,None]
    LxReg_=np.array([0,.01])
    #LxReg_=LxReg_[None,None,:]
    for i in range(0,len(lr_)):
        for j in range(0,len(LxReg_)): 
            for k in range(0,len(eps_)):
                eps  =eps_[k]
                lr=lr_[i]
                LxReg  =LxReg_[j]
                ml_model=LR.LogisticRegression(lr, LxReg, num_iters, eps)
    # Model_selection
                avg_acc,std_acc=ME.model_selection(X_train, y_train, ml_model,Normalization=True,FinalModel=False, k=5)

                
                avg_acc=avg_acc[None,:]
                std_acc=std_acc[None,:]
                
                avg_acc_models = np.append(avg_acc_models, avg_acc, axis=0)
                std_acc_models = np.append(std_acc_models, std_acc, axis=0)
   
    
    #ploting results
  
    img_name=['Experment_2_Iris']
    plot_exp_2(avg_acc_models,lr_,LxReg_,eps_,num_iters,img_name)

def exp_2_adult():
    output='y'
    df=prePros.get_adult_data()
    df_test= pd.DataFrame(None)
    
    # Preparing data
    output=['y']
    df=prePros.get_cleaned_data_adult_2(df, [''],[''],corr_limite=.1, corr_flag=True,limite=.8,delete_outliers=True)
   
    x_headers,y_header=prePros.define_variables(df,output)
    y_header=output
    X_train1, y_train, X_test1, y_test=prePros.Data_spliting_2(df,df_test, x_headers,y_header, training_percent=0.85, shuffle=True, random_seed_value=42)
    
    #prepare data for LR
    X_train,X_test=prePros.prepXforLR(X_train1,X_test1)
    
    # define the machine learning model=ml_model

    num_iters=2000
    avg_acc_models=np.empty([0,num_iters])
    std_acc_models=np.empty([0,num_iters])
    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    eps_=np.array([ .00001])
    #eps_=eps_[:,None,None]
    lr_=np.array([.1,.01 ])
    #lr_=lr_[None,:,None]
    LxReg_=np.array([0,.01])
    #LxReg_=LxReg_[None,None,:]
    for i in range(0,len(lr_)):
        for j in range(0,len(LxReg_)): 
            for k in range(0,len(eps_)):
                eps  =eps_[k]
                lr=lr_[i]
                LxReg  =LxReg_[j]
                ml_model=LR.LogisticRegression(lr, LxReg, num_iters, eps)
    # Model_selection
                avg_acc,std_acc=ME.model_selection(X_train, y_train, ml_model,Normalization=False,FinalModel=False, k=5)
 

                
                avg_acc=avg_acc[None,:]
                std_acc=std_acc[None,:]
                
                avg_acc_models = np.append(avg_acc_models, avg_acc, axis=0)
                std_acc_models = np.append(std_acc_models, std_acc, axis=0)

    
    #ploting results
    img_name=['Experment_2_Adult']
    plot_exp_2(avg_acc_models,lr_,LxReg_,eps_,num_iters,img_name)


def exp_2_WineQualityRed():
    output='y'
    df=prePros.get_WineQualityRed_data()
    df_test= pd.DataFrame(None)
    
    # Preparing data
    output=['y']
    df=prePros.get_cleaned_data_WineQualityRed(df, [''],[''],corr_limite=.1, corr_flag=True,limite=.8,delete_outliers=False)
   
    x_headers,y_header=prePros.define_variables(df,output)
    y_header=output
    X_train1, y_train, X_test1, y_test=prePros.Data_spliting_2(df,df_test, x_headers,y_header, training_percent=0.85, shuffle=True, random_seed_value=42)
    
    #prepare data for LR
    X_train,X_test=prePros.Normalization(X_train1,X_test1)
    X_train,X_test=prePros.prepXforLR(X_train1,X_test1)
    
    
    # define the machine learning model=ml_model
    
    num_iters=500
    avg_acc_models=np.empty([0,num_iters])
    std_acc_models=np.empty([0,num_iters])
    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    eps_=np.array([ .01])
    #eps_=eps_[:,None,None]
    lr_=np.array([.0009,.0013,.0015])
    #lr_=lr_[None,:,None]
    LxReg_=np.array([0,.01])
    #LxReg_=LxReg_[None,None,:]
    for i in range(0,len(lr_)):
        for j in range(0,len(LxReg_)): 
            for k in range(0,len(eps_)):
                eps  =eps_[k]
                lr=lr_[i]
                LxReg  =LxReg_[j]
                ml_model=LR.LogisticRegression(lr, LxReg, num_iters, eps)
    # Model_selection
                avg_acc,std_acc=ME.model_selection(X_train, y_train, ml_model,Normalization=False,FinalModel=False, k=5,)

                
                avg_acc=avg_acc[None,:]
                std_acc=std_acc[None,:]
                
                avg_acc_models = np.append(avg_acc_models, avg_acc, axis=0)
                std_acc_models = np.append(std_acc_models, std_acc, axis=0)
  
    
    #ploting results
    img_name=['Experment2_WineQualityRed']
    plot_exp_2(avg_acc_models,lr_,LxReg_,eps_,num_iters,img_name)
    
    

def experment2():
    exp_2_ionosphere()
    exp_2_Iris()
    
    #exp_2_parkinsons()
    
    exp_2_WineQualityRed()
    exp_2_adult()   
def main():
    exp_2_ionosphere()
    exp_2_Iris()
    
    #exp_2_parkinsons()
    
    exp_2_WineQualityRed()
    exp_2_adult()
    
if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt
import preProcessing  as prePros
import LogisticRegression as LR
import modelEvaluation as ME
import NaiveBayes as NB

def plot_exp_3(avg_acc_LR,avg_acc_NB,instances,img_name):
  
    instances=instances[None,:]
    fig = plt.figure()
    plt.plot(instances.T,avg_acc_LR.T,linestyle='dashed', label= 'LR')
    plt.plot(instances.T,avg_acc_NB.T, label= 'NB' )
    #plt.title("Convergence Graph of Accurecy")
    plt.xlabel("Training instances %")
    plt.ylabel("Accurecy")
    plt.legend()
    plt.savefig("../img/experment3/%s.png" % img_name)    
    plt.show()
    plt.close()
    print("Created ../img/experment3/%s" % img_name)


def exp_3_ionosphere():
    output='y'
    df=prePros.get_ionosphere_data()
    df_test= pd.DataFrame(None)
    
    # Preparing data
    binary_columns=["Re1","Im1"]
    df=prePros.get_cleaned_data_ionosphere(df, [''],binary_columns,corr_limite=.1, corr_flag=False,limite=.8,delete_outliers=False)
    x_headers,y_header=prePros.define_variables(df,output)
    instances=np.linspace(.01,.8,80)
    avg_acc_LR=np.empty([1,len(instances)])
    avg_acc_NB=np.empty([1,len(instances)])
    for i in range(0,len(instances)):
        #data spliting
        X_train1, y_train, X_test1, y_test=prePros.Data_spliting_2(df,df_test, x_headers,y_header, training_percent=instances[i], shuffle=True, random_seed_value=42)
        
    #prepare data for LR
        X_train,X_test=prePros.prepXforLR(X_train1,X_test1)
        #define parameters
        num_iters=1000# to ovide infinite loop only
        lr=.01
        eps=.0001
        LxReg=0
    

    
        #normaliztion
#        X_train = X_train / np.abs(X_train).max( axis = 0) 
#        X_train=np.nan_to_num(X_train)
#        X_train[X_train == 0] = 1 
#    
#        X_test = X_test / np.abs(X_test).max( axis = 0)
#        X_test=np.nan_to_num(X_test)
#        X_test[X_test == 0] = 1 
        
        #define model
        ml_model=LR.LogisticRegression(lr, LxReg, num_iters, eps)

    
        #train
        ml_model.fit(X_train, y_train)
        #test    
        yh=ml_model.predict( X_test)
        #accurcy
        test_acc_LR,test_err_LR=ml_model.evaluate_acc(y_test,yh)
        avg_acc_LR[:,i]=test_acc_LR
        #
        #NaiveBase part
        xlable=np.ones((1,X_train1.shape[1]))# GNB
        ml_model=NB.NaiveBayes(xlable)
        ml_model=LR.LogisticRegression(lr, LxReg, num_iters, eps)

    
        #train
        ml_model.fit(X_train, y_train)
        #test    
        yh=ml_model.predict( X_test)
        #accurcy
        test_acc_NB,test_err_LR=ml_model.evaluate_acc(y_test,yh)
        avg_acc_NB[:,i]=test_acc_NB        
    img_name="experment_3_ionosphere"
    plot_exp_3(avg_acc_LR,avg_acc_NB,instances,img_name) 
    
    
    
def exp_3_parkinsons():
    output='y'
    df=prePros.get_parkinsons_data()
    df_test= pd.DataFrame(None)
    
    # Preparing data
    df=prePros.get_cleaned_data_Parkinsons(df, [''],[''],corr_limite=.1, corr_flag=False,limite=.8,delete_outliers=False)
    x_headers,y_header=prePros.define_variables(df,output)
    instances=np.linspace(.01,.8,80)
    avg_acc_LR=np.empty([1,len(instances)])
    avg_acc_NB=np.empty([1,len(instances)])
    for i in range(0,len(instances)):
        #data spliting
        X_train1, y_train, X_test1, y_test=prePros.Data_spliting_2(df,df_test, x_headers,y_header, training_percent=instances[i], shuffle=True, random_seed_value=42)
        
    #prepare data for LR
        X_train,X_test=prePros.prepXforLR(X_train1,X_test1)
        #define parameters
        num_iters=1000# to ovide infinite loop only
        lr=.01
        eps=.0001
        LxReg=0
    

    
#        #normaliztion
#        X_train = X_train / np.abs(X_train).max( axis = 0) 
#        X_train=np.nan_to_num(X_train)
#        X_train[X_train == 0] = 1 
#    
#        X_test = X_test / np.abs(X_test).max( axis = 0)
#        X_test=np.nan_to_num(X_test)
#        X_test[X_test == 0] = 1 
#        
        #define model
        ml_model=LR.LogisticRegression(lr, LxReg, num_iters, eps)

    
        #train
        ml_model.fit(X_train, y_train)
        #test    
        yh=ml_model.predict( X_test)
        #accurcy
        test_acc_LR,test_err_LR=ml_model.evaluate_acc(y_test,yh)
        avg_acc_LR[:,i]=test_acc_LR
        #
        #NaiveBase part
        xlable=np.ones((1,X_train1.shape[1]))# GNB
        ml_model=NB.NaiveBayes(xlable)
        ml_model=LR.LogisticRegression(lr, LxReg, num_iters, eps)

    
        #train
        ml_model.fit(X_train, y_train)
        #test    
        yh=ml_model.predict( X_test)
        #accurcy
        test_acc_NB,test_err_LR=ml_model.evaluate_acc(y_test,yh)
        avg_acc_NB[:,i]=test_acc_NB        
    img_name="experment_3_parkinsons"
    plot_exp_3(avg_acc_LR,avg_acc_NB,instances,img_name)  
    
def exp_3_Iris():
    output='y'
    df=prePros.get_Iris_data()
    df_test= pd.DataFrame(None)
    
    # Preparing data
    df=prePros.get_cleaned_data_Iris(df, [''],[''],corr_limite=.1, corr_flag=True,limite=.8,delete_outliers=False)
    x_headers,y_header=prePros.define_variables(df,output)
    instances=np.linspace(.01,.8,80)
    avg_acc_LR=np.empty([1,len(instances)])
    avg_acc_NB=np.empty([1,len(instances)])
    for i in range(0,len(instances)):
        #data spliting
        X_train1, y_train, X_test1, y_test=prePros.Data_spliting_2(df,df_test, x_headers,y_header, training_percent=instances[i], shuffle=True, random_seed_value=42)
        
    #prepare data for LR
        X_train,X_test=prePros.prepXforLR(X_train1,X_test1)
        #define parameters
        num_iters=1000# to ovide infinite loop only
        lr=.01
        eps=.0001
        LxReg=0
    

    
        #normaliztion
#        X_train = X_train / np.abs(X_train).max( axis = 0) 
#        X_train=np.nan_to_num(X_train)
#        X_train[X_train == 0] = 1 
#    
#        X_test = X_test / np.abs(X_test).max( axis = 0)
#        X_test=np.nan_to_num(X_test)
#        X_test[X_test == 0] = 1 
        
        #define model
        ml_model=LR.LogisticRegression(lr, LxReg, num_iters, eps)

    
        #train
        ml_model.fit(X_train, y_train)
        #test    
        yh=ml_model.predict( X_test)
        #accurcy
        test_acc_LR,test_err_LR=ml_model.evaluate_acc(y_test,yh)
        avg_acc_LR[:,i]=test_acc_LR
        #
        #NaiveBase part
        xlable=np.ones((1,X_train1.shape[1]))# GNB
        ml_model=NB.NaiveBayes(xlable)
        ml_model=LR.LogisticRegression(lr, LxReg, num_iters, eps)

    
        #train
        ml_model.fit(X_train, y_train)
        #test    
        yh=ml_model.predict( X_test)
        #accurcy
        test_acc_NB,test_err_LR=ml_model.evaluate_acc(y_test,yh)
        avg_acc_NB[:,i]=test_acc_NB        
    img_name="experment_3_Iris"
    plot_exp_3(avg_acc_LR,avg_acc_NB,instances,img_name)
    
def exp_3_WineQualityRed():
    output='y'
    df=prePros.get_WineQualityRed_data()
    df_test= pd.DataFrame(None)
    
    # Preparing data
    df=prePros.get_cleaned_data_WineQualityRed(df, [''],[''],corr_limite=.1, corr_flag=True,limite=.8,delete_outliers=False)
    x_headers,y_header=prePros.define_variables(df,output)
    instances=np.linspace(.01,.8,30)
    avg_acc_LR=np.empty([1,len(instances)])
    avg_acc_NB=np.empty([1,len(instances)])
    for i in range(0,len(instances)):
        #data spliting
        X_train1, y_train, X_test1, y_test=prePros.Data_spliting(df,df_test, x_headers,y_header, training_percent=instances[i], shuffle=True, random_seed_value=42)
        
    #prepare data for LR
        X_train,X_test=prePros.prepXforLR(X_train1,X_test1)
        #define parameters
        num_iters=1000# to ovide infinite loop only
        lr=.01
        eps=.0001
        LxReg=0
    

    
        #normaliztion
        X_train,X_test=prePros.Normalization(X_train,X_test)
        
        #define model
        ml_model=LR.LogisticRegression(lr, LxReg, num_iters, eps)

    
        #train
        ml_model.fit(X_train, y_train)
        #test    
        yh=ml_model.predict( X_test)
        #accurcy
        test_acc_LR,test_err_LR=ml_model.evaluate_acc(y_test,yh)
        avg_acc_LR[:,i]=test_acc_LR
        #
        #NaiveBase part
        xlable=np.ones((1,X_train1.shape[1]))# GNB
        ml_model=NB.NaiveBayes(xlable)
        ml_model=LR.LogisticRegression(lr, LxReg, num_iters, eps)

    
        #train
        ml_model.fit(X_train, y_train)
        #test    
        yh=ml_model.predict( X_test)
        #accurcy
        test_acc_NB,test_err_LR=ml_model.evaluate_acc(y_test,yh)
        avg_acc_NB[:,i]=test_acc_NB 
    img_name="experment_3_WineQualityRed"
    plot_exp_3(avg_acc_LR,avg_acc_NB,instances,img_name)
    
def exp_3_adult():
    df=prePros.get_adult_data()
    df_test= pd.DataFrame(None)
    
    # Preparing data
    output=['y']
    df=prePros.get_cleaned_data_adult_2(df, [''],[''],corr_limite=.1, corr_flag=False,limite=.8,delete_outliers=False)
    df=prePros.deleteCorelatedFeatures(df)
    x_headers,y_header=prePros.define_variables(df,'y')
    y_header=output

    instances=np.linspace(.01,.8,30)
    avg_acc_LR=np.empty([1,len(instances)])
    avg_acc_NB=np.empty([1,len(instances)])
    for i in range(0,len(instances)):
        #data spliting
        X_train1, y_train, X_test1, y_test=prePros.Data_spliting(df,df_test, x_headers,y_header, training_percent=instances[i], shuffle=True, random_seed_value=42)
        
    #prepare data for LR
        X_train,X_test=prePros.prepXforLR(X_train1,X_test1)
        #define parameters
        num_iters=3000# to ovide infinite loop only
        lr=.1
        eps=.0001
        LxReg=0

    
        #normaliztion
        #X_train = X_train / np.abs(X_train).max( axis = 0) 
        #X_train=np.nan_to_num(X_train)
        #X_train[X_train == 0] = 1 
    
        #X_test = X_test / np.abs(X_test).max( axis = 0)
        #X_test=np.nan_to_num(X_test)
        #X_test[X_test == 0] = 1 
        
        #define model
        ml_model=LR.LogisticRegression(lr, LxReg, num_iters, eps)

    
        #train
        ml_model.fit(X_train, y_train)
        #test    
        yh=ml_model.predict( X_test)
        #accurcy
        test_acc_LR,test_err_LR=ml_model.evaluate_acc(y_test,yh)
        avg_acc_LR[:,i]=test_acc_LR
        #
        #NaiveBase part
        xlable=np.ones((1,X_train1.shape[1]))# GNB
        ml_model=NB.NaiveBayes(xlable)
        ml_model=LR.LogisticRegression(lr, LxReg, num_iters, eps)

    
        #train
        ml_model.fit(X_train, y_train)
        #test    
        yh=ml_model.predict( X_test)
        #accurcy
        test_acc_NB,test_err_LR=ml_model.evaluate_acc(y_test,yh)
        avg_acc_NB[:,i]=test_acc_NB        
    img_name="experment_3_adult"
    plot_exp_3(avg_acc_LR,avg_acc_NB,instances,img_name)


def experment3():
    exp_3_Iris()
    exp_3_ionosphere()
    #exp_3_parkinsons()
    
    exp_3_WineQualityRed()
    exp_3_adult()   
def main():
   # exp_3_Iris()
    #exp_3_ionosphere()
    #exp_3_parkinsons()
    
   # exp_3_WineQualityRed()
    exp_3_adult()
    
if __name__ == "__main__":
    main()
     
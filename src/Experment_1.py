import numpy as np
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt
import preProcessing  as prePros
import LogisticRegression as LR
import modelEvaluation as ME 
import NaiveBayes as NB
from sklearn.naive_bayes import GaussianNB 
from sklearn.naive_bayes import BernoulliNB
import sklearn.linear_model as Sklean_LogisticRegression
from sklearn.metrics import accuracy_score as Sklean_accuracy_score

def print_exp_1(avg_acc_models,lr_,LxReg_,eps_,num_iters,img_name):
    index= -1
    fig = plt.figure()
    for i in range(0,len(lr_)):
            for j in range(0,len(LxReg_)): 
                for k in range(0,len(eps_)):
                    index +=1
                    plt.plot(np.linspace(1, num_iters, num=num_iters),avg_acc_models[index,:], label= r"lr=" + str(lr_[i])+r" $\lambda$="+ str(LxReg_[j])+r" $\epsilon$="+str(eps_[k]))    
    #plt.title("Convergence Graph of Accurecy")
    plt.title("Convergence Graph of Accuracy")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("../img/experment2/%s.png" % img_name)
    plt.show() 
    plt.close()
    print("Created ../img/experment2/%s" % img_name)
    


def exp_1_ionosphere():
    output='y'
    df=prePros.get_ionosphere_data()
    df_test= pd.DataFrame(None)
    
    # Preparing data
    binary_columns=["Re1","Im1"]
    df=prePros.get_cleaned_data_ionosphere(df, [''],binary_columns,corr_limite=.1, corr_flag=False,limite=.75,delete_outliers=True)
    
    #print(df.head(5))
    #df=prePros.deleteCorelatedFeatures(df)
    #print(df.head(5))
    
    x_headers,y_header=prePros.define_variables(df,output)
    X_train1, y_train, X_test1, y_test=prePros.Data_spliting(df,df_test, x_headers,y_header, training_percent=0.80, shuffle=True, random_seed_value=42)
    #prepare data for LR
    
    X_train,X_test=prePros.prepXforLR(X_train1,X_test1)#leanear regression matrix
    
    # define the machine learning model=ml_model LR 
    temp_acc=-1
    temp_time=np.inf
    num_iters=2000

    eps_=np.array([1e-5 ])
    #eps_=eps_[:,None,None]
    lr_=np.array([.01 ])
    #lr_=lr_[None,:,None]
    LxReg_=np.array([0])
    #LxReg_=LxReg_[None,None,:]
    for i in range(0,len(lr_)):
        for j in range(0,len(LxReg_)): 
            for k in range(0,len(eps_)):
                eps  =eps_[k]
                lr=lr_[i]
                LxReg  =LxReg_[j]
                ml_model=LR.LogisticRegression(lr, LxReg, num_iters, eps)
    # append results for comparison
                start_time = time.time()
                avg_acc,std_acc=ME.model_selection(X_train, y_train, ml_model,Normalization=False,FinalModel=True, k=5)
                training_Time=time.time() - start_time 
                if temp_acc < avg_acc:
                    selected_Model=LR.LogisticRegression(lr, LxReg, num_iters, eps)
                elif temp_acc == avg_acc:
                    if temp_time>training_Time:
                        selected_Model=LR.LogisticRegression(lr, LxReg, num_iters, eps)
   
    
   #evaluation phase: parameters acc, complexity

    start_time = time.time()
    CV_LR_acc,std_acc=ME.model_selection(X_train, y_train, selected_Model,Normalization=False,FinalModel=True, k=5)
    CV_LR_Time=time.time() - start_time  
    
     # Normalization
    #X_train,X_test=prePros.Normalization(X_train,X_test)
    #evaluation phase
    start_time = time.time()
    selected_Model.fit(X_train, y_train)
    training_LR_Time=time.time() - start_time
    
    
    yh=selected_Model.predict( X_train)
    train_LR_acc,test_err_LR=selected_Model.evaluate_acc(y_train,yh)
    

    
    start_time = time.time()
    yh=selected_Model.predict( X_test)
    test_LR_Time=time.time() - start_time
    
    test_LR_acc,test_err_LR=selected_Model.evaluate_acc(y_test,yh)
    
    

#    X_train1,X_test1=prePros.Normalization(X_train1,X_test1)
    xlable=np.zeros((1,X_train1.shape[1]))
    
    NB_model=NB.NaiveBayes(xlable)
    
    X_train1, y_train, X_test1, y_test=prePros.Data_spliting(df,df_test, x_headers,y_header, training_percent=.8, shuffle=True, random_seed_value=42)

    start_time = time.time()
    CV_NB_acc,std_acc=ME.model_selection(X_train1, y_train, NB_model,Normalization=False,FinalModel=True, k=5)
    CV_NB_Time=time.time() - start_time  
    
    #X_train1,X_test1=prePros.Normalization(X_train1,X_test1)
    #evaluation phase
    start_time = time.time()
    NB_model.fit(X_train1, y_train)
    train_NB_Time=time.time() - start_time
    
    
    yh=NB_model.predict( X_train1)
    train_NB_acc,test_err_LR=NB_model.evaluate_acc(y_train,yh)
    

    
    start_time = time.time()
    yh=NB_model.predict( X_test1)
    test_NB_Time=time.time() - start_time
    
    test_NB_acc,test_err_LR=NB_model.evaluate_acc(yh,y_test)

    results_df = pd.DataFrame(None)
    Parameters = pd.Series([])
    Logistic_Regression=pd.Series([])
    NaiveBayse=pd.Series([])
    for i in range(0,9):
        
        if i==0:
            Parameters[i]="CV_ACC"     
            Logistic_Regression[i]=CV_LR_acc
            NaiveBayse[i]=CV_NB_acc          
        elif i==1:
            Parameters[i]="CV_TIME"     
            Logistic_Regression[i]=CV_LR_Time
            NaiveBayse[i]=CV_NB_Time           
        elif i==2:
            Parameters[i]="train_ACC"     
            Logistic_Regression[i]=train_LR_acc
            NaiveBayse[i]=train_NB_acc          
        elif i==3:
            Parameters[i]="train_TIME"     
            Logistic_Regression[i]=training_LR_Time
            NaiveBayse[i]=train_NB_Time          
        elif i==4:            
            Parameters[i]="test_ACC"     
            Logistic_Regression[i]=test_LR_acc
            NaiveBayse[i]=test_NB_acc           
        elif i==5:    
            Parameters[i]="test_TIME"     
            Logistic_Regression[i]=test_LR_Time
            NaiveBayse[i]=test_NB_Time           
        elif i==6:
            Parameters[i]="LR_LEARNINGRATE"     
            Logistic_Regression[i]=selected_Model.lr
            NaiveBayse[i]=0           
        elif i==7:
            Parameters[i]="LR_Lambda"     
            Logistic_Regression[i]=selected_Model.LxReg
            NaiveBayse[i]=0          
        elif i==8:
            Parameters[i]="LR_eps"     
            Logistic_Regression[i]=selected_Model.eps
            NaiveBayse[i]=0
         
    
    results_df.insert(0, "NaiveBayse", NaiveBayse) 
    results_df.insert(0, "Logistic_Regression", Logistic_Regression)
    results_df.insert(0, "Parameter", Parameters)
    print("_____experment 1 ionosphere_____")
    print(results_df)
    
    #sklearn
    print(X_train1.shape)
    print(X_test1.shape)
    clf=Sklean_LogisticRegression.LogisticRegression()
    clf = clf.fit(X_train1, y_train)
    yh=clf.predict(X_test1)
    sklearn_acc_LR=Sklean_accuracy_score(y_test, yh)
    print("sklearn_acc_LR="+str(sklearn_acc_LR))
    
    nbb=GaussianNB()
    nbb = nbb.fit(X_train1, y_train)
    yh=nbb.predict(X_test1)
    sklearn_acc_LR=Sklean_accuracy_score(y_test, yh)
    print("sklearn_acc_GaussianNB="+str(sklearn_acc_LR))
    
    
def exp_1_parkinsons():
    output='y'
    df=prePros.get_parkinsons_data()
    df_test= pd.DataFrame(None)
    
    df=prePros.get_cleaned_data_Parkinsons(df, [''],[''],corr_limite=.1, corr_flag=False,limite=.7,delete_outliers=False)
    x_headers,y_header=prePros.define_variables(df,output)
    X_train1, y_train, X_test1, y_test=prePros.Data_spliting(df,df_test, x_headers,y_header, training_percent=0.80, shuffle=True, random_seed_value=42)
    #prepare data for LR
    # Preparing data
    #binary_columns=["Re1","Im1"]

    X_train,X_test=prePros.prepXforLR(X_train1,X_test1)
    
    # define the machine learning model=ml_model

    # define the machine learning model=ml_model LR 

    temp_acc=-1
    temp_time=np.inf
    num_iters=2000

    eps_=np.array([1e-5 ])
    #eps_=eps_[:,None,None]
    lr_=np.array([.001 ])
    #lr_=lr_[None,:,None]
    LxReg_=np.array([0])
    #LxReg_=LxReg_[None,None,:]
    for i in range(0,len(lr_)):
        for j in range(0,len(LxReg_)): 
            for k in range(0,len(eps_)):
                eps  =eps_[k]
                lr=lr_[i]
                LxReg  =LxReg_[j]
                ml_model=LR.LogisticRegression(lr, LxReg, num_iters, eps)
    # append results for comparison
                start_time = time.time()
                avg_acc,std_acc=ME.model_selection(X_train, y_train, ml_model,Normalization=False,FinalModel=True, k=5)
                training_Time=time.time() - start_time 
                if temp_acc < avg_acc:
                    selected_Model=LR.LogisticRegression(lr, LxReg, num_iters, eps)
                elif temp_acc == avg_acc:
                    if temp_time>training_Time:
                        selected_Model=LR.LogisticRegression(lr, LxReg, num_iters, eps)
   
    
   #evaluation phase: parameters acc, complexity
    start_time = time.time()
    CV_LR_acc,std_acc=ME.model_selection(X_train, y_train, selected_Model,Normalization=False,FinalModel=True, k=5)
    CV_LR_Time=time.time() - start_time  
    
     # Normalization
    #X_train,X_test=prePros.Normalization(X_train,X_test)
    #evaluation phase
    start_time = time.time()
    selected_Model.fit(X_train, y_train)
    training_LR_Time=time.time() - start_time
    
    
    yh=selected_Model.predict( X_train)
    train_LR_acc,test_err_LR=selected_Model.evaluate_acc(y_train,yh)
    

    
    start_time = time.time()
    yh=selected_Model.predict( X_test)
    test_LR_Time=time.time() - start_time
    
    test_LR_acc,test_err_LR=selected_Model.evaluate_acc(y_test,yh)
    
    


    xlable=np.ones((1,X_train1.shape[1]))
    
    NB_model=NB.NaiveBayes(xlable)
    
    X_train1, y_train, X_test1, y_test=prePros.Data_spliting(df,df_test, x_headers,y_header, training_percent=0.90, shuffle=True, random_seed_value=42)

    start_time = time.time()
    CV_NB_acc,std_acc=ME.model_selection(X_train1, y_train, NB_model,Normalization=False,FinalModel=True, k=5)
    CV_NB_Time=time.time() - start_time  
    
    #X_train1,X_test1=prePros.Normalization(X_train1,X_test1)
    #evaluation phase
    start_time = time.time()
    NB_model.fit(X_train1, y_train)
    train_NB_Time=time.time() - start_time
    
    
    yh=NB_model.predict( X_train1)
    train_NB_acc,test_err_LR=NB_model.evaluate_acc(y_train,yh)
    

    
    start_time = time.time()
    yh=NB_model.predict( X_test1)
    test_NB_Time=time.time() - start_time
    
    test_NB_acc,test_err_LR=NB_model.evaluate_acc(yh,y_test)

    results_df = pd.DataFrame(None)
    Parameters = pd.Series([])
    Logistic_Regression=pd.Series([])
    NaiveBayse=pd.Series([])
    for i in range(0,9):
        
        if i==0:
            Parameters[i]="CV_ACC"     
            Logistic_Regression[i]=CV_LR_acc
            NaiveBayse[i]=CV_NB_acc          
        elif i==1:
            Parameters[i]="CV_TIME"     
            Logistic_Regression[i]=CV_LR_Time
            NaiveBayse[i]=CV_NB_Time           
        elif i==2:
            Parameters[i]="train_ACC"     
            Logistic_Regression[i]=train_LR_acc
            NaiveBayse[i]=train_NB_acc          
        elif i==3:
            Parameters[i]="train_TIME"     
            Logistic_Regression[i]=training_LR_Time
            NaiveBayse[i]=train_NB_Time          
        elif i==4:            
            Parameters[i]="test_ACC"     
            Logistic_Regression[i]=test_LR_acc
            NaiveBayse[i]=test_NB_acc           
        elif i==5:    
            Parameters[i]="test_TIME"     
            Logistic_Regression[i]=test_LR_Time
            NaiveBayse[i]=test_NB_Time           
        elif i==6:
            Parameters[i]="LR_LEARNINGRATE"     
            Logistic_Regression[i]=selected_Model.lr
            NaiveBayse[i]=0           
        elif i==7:
            Parameters[i]="LR_Lambda"     
            Logistic_Regression[i]=selected_Model.LxReg
            NaiveBayse[i]=0          
        elif i==8:
            Parameters[i]="LR_eps"     
            Logistic_Regression[i]=selected_Model.eps
            NaiveBayse[i]=0
         
    
    results_df.insert(0, "NaiveBayse", NaiveBayse) 
    results_df.insert(0, "Logistic_Regression", Logistic_Regression)
    results_df.insert(0, "Parameter", Parameters)
    print("_____experment 1 parkinsons_____")
    print(results_df)
    
    
    print(X_train1.shape)
    print(X_test1.shape)
    clf=Sklean_LogisticRegression.LogisticRegression()
    clf = clf.fit(X_train1, y_train)
    yh=clf.predict(X_test1)
    sklearn_acc_LR=Sklean_accuracy_score(y_test, yh)
    print("sklearn_acc_LR="+str(sklearn_acc_LR))
    
    nbb=GaussianNB()
    nbb = nbb.fit(X_train1, y_train)
    yh=nbb.predict(X_test1)
    sklearn_acc_LR=Sklean_accuracy_score(y_test, yh)
    print("sklearn_acc_GaussianNB="+str(sklearn_acc_LR))

def exp_1_adult():
    output='y'
    df=prePros.get_adult_data()
    df_test= pd.DataFrame(None)
    
    # Preparing data
    output=['y']
    df=prePros.get_cleaned_data_adult_2(df, [''],[''],corr_limite=.1, corr_flag=False,limite=.80,delete_outliers=False)
    #print(df.head(5))
    df=prePros.deleteCorelatedFeatures(df)
    #print(df.head(5))

   
    x_headers,y_header=prePros.define_variables(df,output)
    y_header=output
    X_train1, y_train, X_test1, y_test=prePros.Data_spliting_2(df,df_test, x_headers,y_header, training_percent=0.80, shuffle=True, random_seed_value=42)
    
    #prepare data for LR
    X_train,X_test=prePros.prepXforLR(X_train1,X_test1)
    
    # define the machine learning model=ml_model

 
    temp_acc=-1
    temp_time=np.inf
    num_iters=2000

    eps_=np.array([1e-5 ])
    #eps_=eps_[:,None,None]
    lr_=np.array([.1 ])
    #lr_=lr_[None,:,None]
    LxReg_=np.array([0])
    #LxReg_=LxReg_[None,None,:]
    for i in range(0,len(lr_)):
        for j in range(0,len(LxReg_)): 
            for k in range(0,len(eps_)):
                eps  =eps_[k]
                lr=lr_[i]
                LxReg  =LxReg_[j]
                ml_model=LR.LogisticRegression(lr, LxReg, num_iters, eps)
    # append results for comparison
                start_time = time.time()
                avg_acc,std_acc=ME.model_selection(X_train, y_train, ml_model,Normalization=False,FinalModel=True, k=5)
                training_Time=time.time() - start_time 
                if temp_acc < avg_acc:
                    selected_Model=LR.LogisticRegression(lr, LxReg, num_iters, eps)
                elif temp_acc == avg_acc:
                    if temp_time>training_Time:
                        selected_Model=LR.LogisticRegression(lr, LxReg, num_iters, eps)
   
    
   #evaluation phase: parameters acc, complexity
    start_time = time.time()
    CV_LR_acc,std_acc=ME.model_selection(X_train, y_train, selected_Model,Normalization=False,FinalModel=True, k=5)
    CV_LR_Time=time.time() - start_time  
    
     # Normalization
    #X_train,X_test=prePros.Normalization(X_train,X_test)
    #evaluation phase
    start_time = time.time()
    selected_Model.fit(X_train, y_train)
    training_LR_Time=time.time() - start_time
    
    
    yh=selected_Model.predict( X_train)
    train_LR_acc,test_err_LR=selected_Model.evaluate_acc(y_train,yh)
    

    
    start_time = time.time()
    yh=selected_Model.predict( X_test)
    test_LR_Time=time.time() - start_time
    
    test_LR_acc,test_err_LR=selected_Model.evaluate_acc(y_test,yh)
    
    


    xlable=np.ones((1,X_train1.shape[1]))
    
    NB_model=NB.NaiveBayes(xlable)
    
    #X_train1, y_train, X_test1, y_test=prePros.Data_spliting_2(df,df_test, x_headers,y_header, training_percent=0.80, shuffle=True, random_seed_value=42)

    start_time = time.time()
    CV_NB_acc,std_acc=ME.model_selection(X_train1, y_train, NB_model,Normalization=False,FinalModel=True, k=5)
    CV_NB_Time=time.time() - start_time  
    
    #X_train1,X_test1=prePros.Normalization(X_train1,X_test1)
    #evaluation phase
    start_time = time.time()
    NB_model.fit(X_train1, y_train)
    train_NB_Time=time.time() - start_time
    
    
    yh=NB_model.predict( X_train1)
    train_NB_acc,test_err_LR=NB_model.evaluate_acc(y_train,yh)
    

    
    start_time = time.time()
    yh=NB_model.predict( X_test1)
    test_NB_Time=time.time() - start_time
    
    test_NB_acc,test_err_LR=NB_model.evaluate_acc(yh,y_test)

    results_df = pd.DataFrame(None)
    Parameters = pd.Series([])
    Logistic_Regression=pd.Series([])
    NaiveBayse=pd.Series([])
    for i in range(0,9):
        
        if i==0:
            Parameters[i]="CV_ACC"     
            Logistic_Regression[i]=CV_LR_acc
            NaiveBayse[i]=CV_NB_acc          
        elif i==1:
            Parameters[i]="CV_TIME"     
            Logistic_Regression[i]=CV_LR_Time
            NaiveBayse[i]=CV_NB_Time           
        elif i==2:
            Parameters[i]="train_ACC"     
            Logistic_Regression[i]=train_LR_acc
            NaiveBayse[i]=train_NB_acc          
        elif i==3:
            Parameters[i]="train_TIME"     
            Logistic_Regression[i]=training_LR_Time
            NaiveBayse[i]=train_NB_Time          
        elif i==4:            
            Parameters[i]="test_ACC"     
            Logistic_Regression[i]=test_LR_acc
            NaiveBayse[i]=test_NB_acc           
        elif i==5:    
            Parameters[i]="test_TIME"     
            Logistic_Regression[i]=test_LR_Time
            NaiveBayse[i]=test_NB_Time           
        elif i==6:
            Parameters[i]="LR_LEARNINGRATE"     
            Logistic_Regression[i]=selected_Model.lr
            NaiveBayse[i]=0           
        elif i==7:
            Parameters[i]="LR_Lambda"     
            Logistic_Regression[i]=selected_Model.LxReg
            NaiveBayse[i]=0          
        elif i==8:
            Parameters[i]="LR_eps"     
            Logistic_Regression[i]=selected_Model.eps
            NaiveBayse[i]=0
         
    
    results_df.insert(0, "NaiveBayse", NaiveBayse) 
    results_df.insert(0, "Logistic_Regression", Logistic_Regression)
    results_df.insert(0, "Parameter", Parameters)
    print("_____experment 1 Adult_____")
    print(results_df)



    print(X_train1.shape)
    print(X_test1.shape)
    clf=Sklean_LogisticRegression.LogisticRegression()
    clf = clf.fit(X_train1, y_train)
    yh=clf.predict(X_test1)
    sklearn_acc_LR=Sklean_accuracy_score(y_test, yh)
    print("sklearn_acc_LR="+str(sklearn_acc_LR))
    
    nbb=GaussianNB()
    nbb = nbb.fit(X_train1, y_train)
    yh=nbb.predict(X_test1)
    sklearn_acc_Nb=Sklean_accuracy_score(y_test, yh)
    print("sklearn_acc_GaussianNB="+str(sklearn_acc_Nb))

def exp_1_WineQualityRed():
    output='y'
    df=prePros.get_WineQualityRed_data()
    df_test= pd.DataFrame(None)
    
    # Preparing data
    output=['y']
    df=prePros.get_cleaned_data_WineQualityRed(df, [''],[''],corr_limite=.0, corr_flag=False,limite=.95,delete_outliers=True)
   
    #print(df.head(5))
    df=prePros.deleteCorelatedFeatures(df)
    #print(df.head(5))
    
    x_headers,y_header=prePros.define_variables(df,output)
    y_header=output
    
    X_train1, y_train, X_test1, y_test=prePros.Data_spliting(df,df_test, x_headers,y_header, training_percent=0.78, shuffle=True, random_seed_value=42)
    
    #prepare data for LR

    X_train,X_test=prePros.prepXforLR(X_train1,X_test1)
    # define the machine learning model=ml_model

    temp_acc=-1
    temp_time=np.inf
    num_iters=2000

    eps_=np.array([1e-5 ])
    #eps_=eps_[:,None,None]
    lr_=np.array([.0001 ])
    #lr_=lr_[None,:,None]
    LxReg_=np.array([0])
    #LxReg_=LxReg_[None,None,:]
    for i in range(0,len(lr_)):
        for j in range(0,len(LxReg_)): 
            for k in range(0,len(eps_)):
                eps  =eps_[k]
                lr=lr_[i]
                LxReg  =LxReg_[j]
                ml_model=LR.LogisticRegression(lr, LxReg, num_iters, eps)
    # append results for comparison
                start_time = time.time()
                avg_acc,std_acc=ME.model_selection(X_train, y_train, ml_model,Normalization=False,FinalModel=True, k=5)
                training_Time=time.time() - start_time 
                if temp_acc < avg_acc:
                    selected_Model=LR.LogisticRegression(lr, LxReg, num_iters, eps)
                elif temp_acc == avg_acc:
                    if temp_time>training_Time:
                        selected_Model=LR.LogisticRegression(lr, LxReg, num_iters, eps)
   
    
   #evaluation phase: parameters acc, complexity
    start_time = time.time()
    CV_LR_acc,std_acc=ME.model_selection(X_train, y_train, selected_Model,Normalization=False,FinalModel=True, k=5)
    CV_LR_Time=time.time() - start_time  
    
     # Normalization
    #X_train,X_test=prePros.Normalization(X_train,X_test)
    #evaluation phase
    start_time = time.time()
    selected_Model.fit(X_train, y_train)
    training_LR_Time=time.time() - start_time
    
    
    yh=selected_Model.predict( X_train)
    train_LR_acc,test_err_LR=selected_Model.evaluate_acc(y_train,yh)
    

    
    start_time = time.time()
    yh=selected_Model.predict( X_test)
    test_LR_Time=time.time() - start_time
    
    test_LR_acc,test_err_LR=selected_Model.evaluate_acc(y_test,yh)
    
    


    xlable=np.ones((1,X_train1.shape[1]))
    
    NB_model=NB.NaiveBayes(xlable)
    X_train1,X_test1=prePros.Normalization(X_train1,X_test1)
    #X_train1, y_train, X_test1, y_test=prePros.Data_spliting_2(df,df_test, x_headers,y_header, training_percent=0.90, shuffle=True, random_seed_value=42)
    start_time = time.time()
    CV_NB_acc,std_acc=ME.model_selection(X_train1, y_train, NB_model,Normalization=False,FinalModel=True, k=5)
    CV_NB_Time=time.time() - start_time  
    
    #X_train1,X_test1=prePros.Normalization(X_train1,X_test1)
    #evaluation phase
    start_time = time.time()
    NB_model.fit(X_train1, y_train)
    train_NB_Time=time.time() - start_time
    
    
    yh=NB_model.predict( X_train1)
    train_NB_acc,test_err_LR=NB_model.evaluate_acc(y_train,yh)
    

    
    start_time = time.time()
    yh=NB_model.predict( X_test1)
    test_NB_Time=time.time() - start_time
    
    test_NB_acc,test_err_LR=NB_model.evaluate_acc(yh,y_test)

    results_df = pd.DataFrame(None)
    Parameters = pd.Series([])
    Logistic_Regression=pd.Series([])
    NaiveBayse=pd.Series([])
    for i in range(0,9):
        
        if i==0:
            Parameters[i]="CV_ACC"     
            Logistic_Regression[i]=CV_LR_acc
            NaiveBayse[i]=CV_NB_acc          
        elif i==1:
            Parameters[i]="CV_TIME"     
            Logistic_Regression[i]=CV_LR_Time
            NaiveBayse[i]=CV_NB_Time           
        elif i==2:
            Parameters[i]="train_ACC"     
            Logistic_Regression[i]=train_LR_acc
            NaiveBayse[i]=train_NB_acc          
        elif i==3:
            Parameters[i]="train_TIME"     
            Logistic_Regression[i]=training_LR_Time
            NaiveBayse[i]=train_NB_Time          
        elif i==4:            
            Parameters[i]="test_ACC"     
            Logistic_Regression[i]=test_LR_acc
            NaiveBayse[i]=test_NB_acc           
        elif i==5:    
            Parameters[i]="test_TIME"     
            Logistic_Regression[i]=test_LR_Time
            NaiveBayse[i]=test_NB_Time           
        elif i==6:
            Parameters[i]="LR_LEARNINGRATE"     
            Logistic_Regression[i]=selected_Model.lr
            NaiveBayse[i]=0           
        elif i==7:
            Parameters[i]="LR_Lambda"     
            Logistic_Regression[i]=selected_Model.LxReg
            NaiveBayse[i]=0          
        elif i==8:
            Parameters[i]="LR_eps"     
            Logistic_Regression[i]=selected_Model.eps
            NaiveBayse[i]=0
         
    
    results_df.insert(0, "NaiveBayse", NaiveBayse) 
    results_df.insert(0, "Logistic_Regression", Logistic_Regression)
    results_df.insert(0, "Parameter", Parameters)
    print("_____experment 1 WineQualityRed_____")
    print(results_df)
    
    
    print(X_train1.shape)
    print(X_test1.shape)
    clf=Sklean_LogisticRegression.LogisticRegression()
    clf = clf.fit(X_train, y_train)
    yh=clf.predict(X_test)
    sklearn_acc_LR=Sklean_accuracy_score(y_test, yh)
    print("sklearn_acc_LR="+str(sklearn_acc_LR))
    
    nbb=GaussianNB()
    nbb = nbb.fit(X_train1, y_train)
    yh=nbb.predict(X_test1)
    sklearn_acc_LR=Sklean_accuracy_score(y_test, yh)
    print("sklearn_acc_GaussianNB="+str(sklearn_acc_LR))
    
    
def exp_1_Iris():
    output='y'
    df=prePros.get_Iris_data()
    df_test= pd.DataFrame(None)
    
    df=prePros.get_cleaned_data_Iris(df, [''],[''],corr_limite=.0, corr_flag=False,limite=.7,delete_outliers=False)
    x_headers,y_header=prePros.define_variables(df,output)
    X_train1, y_train, X_test1, y_test=prePros.Data_spliting(df,df_test, x_headers,y_header, training_percent=0.85, shuffle=True, random_seed_value=42)
    #prepare data for LR
    # Preparing data
    #binary_columns=["Re1","Im1"]

    X_train,X_test=prePros.prepXforLR(X_train1,X_test1)
    
    # define the machine learning model=ml_model

    temp_acc=-1
    temp_time=np.inf
    num_iters=2000

    eps_=np.array([1e-5 ])
    #eps_=eps_[:,None,None]
    lr_=np.array([.01 ])
    #lr_=lr_[None,:,None]
    LxReg_=np.array([0])
    #LxReg_=LxReg_[None,None,:]
    for i in range(0,len(lr_)):
        for j in range(0,len(LxReg_)): 
            for k in range(0,len(eps_)):
                eps  =eps_[k]
                lr=lr_[i]
                LxReg  =LxReg_[j]
                ml_model=LR.LogisticRegression(lr, LxReg, num_iters, eps)
    # append results for comparison
                start_time = time.time()
                avg_acc,std_acc=ME.model_selection(X_train, y_train, ml_model,Normalization=False,FinalModel=True, k=5)
                training_Time=time.time() - start_time 
                if temp_acc < avg_acc:
                    selected_Model=LR.LogisticRegression(lr, LxReg, num_iters, eps)
                elif temp_acc == avg_acc:
                    if temp_time>training_Time:
                        selected_Model=LR.LogisticRegression(lr, LxReg, num_iters, eps)
   
    
   #evaluation phase: parameters acc, complexity
    start_time = time.time()
    CV_LR_acc,std_acc=ME.model_selection(X_train, y_train, selected_Model,Normalization=False,FinalModel=True, k=5)
    CV_LR_Time=time.time() - start_time  
    
     # Normalization
    #X_train,X_test=prePros.Normalization(X_train,X_test)
    #evaluation phase
    start_time = time.time()
    selected_Model.fit(X_train, y_train)
    training_LR_Time=time.time() - start_time
    
    
    yh=selected_Model.predict( X_train)
    train_LR_acc,test_err_LR=selected_Model.evaluate_acc(y_train,yh)
    

    
    start_time = time.time()
    yh=selected_Model.predict( X_test)
    test_LR_Time=time.time() - start_time
    
    test_LR_acc,test_err_LR=selected_Model.evaluate_acc(y_test,yh)
    
    


    xlable=np.ones((1,X_train1.shape[1]))
    
    NB_model=NB.NaiveBayes(xlable)
    
    #X_train1, y_train, X_test1, y_test=prePros.Data_spliting(df,df_test, x_headers,y_header, training_percent=0.80, shuffle=True, random_seed_value=42)

    start_time = time.time()
    CV_NB_acc,std_acc=ME.model_selection(X_train1, y_train, NB_model,Normalization=False,FinalModel=True, k=5)
    CV_NB_Time=time.time() - start_time  
    
    #X_train1,X_test1=prePros.Normalization(X_train1,X_test1)
    #evaluation phase
    start_time = time.time()
    NB_model.fit(X_train1, y_train)
    train_NB_Time=time.time() - start_time
    
    
    yh=NB_model.predict( X_train1)
    train_NB_acc,test_err_LR=NB_model.evaluate_acc(y_train,yh)
    

    
    start_time = time.time()
    yh=NB_model.predict( X_test1)
    test_NB_Time=time.time() - start_time
    
    test_NB_acc,test_err_LR=NB_model.evaluate_acc(yh,y_test)

    results_df = pd.DataFrame(None)
    Parameters = pd.Series([])
    Logistic_Regression=pd.Series([])
    NaiveBayse=pd.Series([])
    for i in range(0,9):
        
        if i==0:
            Parameters[i]="CV_ACC"     
            Logistic_Regression[i]=CV_LR_acc
            NaiveBayse[i]=CV_NB_acc          
        elif i==1:
            Parameters[i]="CV_TIME"     
            Logistic_Regression[i]=CV_LR_Time
            NaiveBayse[i]=CV_NB_Time           
        elif i==2:
            Parameters[i]="train_ACC"     
            Logistic_Regression[i]=train_LR_acc
            NaiveBayse[i]=train_NB_acc          
        elif i==3:
            Parameters[i]="train_TIME"     
            Logistic_Regression[i]=training_LR_Time
            NaiveBayse[i]=train_NB_Time          
        elif i==4:            
            Parameters[i]="test_ACC"     
            Logistic_Regression[i]=test_LR_acc
            NaiveBayse[i]=test_NB_acc           
        elif i==5:    
            Parameters[i]="test_TIME"     
            Logistic_Regression[i]=test_LR_Time
            NaiveBayse[i]=test_NB_Time           
        elif i==6:
            Parameters[i]="LR_LEARNINGRATE"     
            Logistic_Regression[i]=selected_Model.lr
            NaiveBayse[i]=0           
        elif i==7:
            Parameters[i]="LR_Lambda"     
            Logistic_Regression[i]=selected_Model.LxReg
            NaiveBayse[i]=0          
        elif i==8:
            Parameters[i]="LR_eps"     
            Logistic_Regression[i]=selected_Model.eps
            NaiveBayse[i]=0
         
    
    results_df.insert(0, "NaiveBayse", NaiveBayse) 
    results_df.insert(0, "Logistic_Regression", Logistic_Regression)
    results_df.insert(0, "Parameter", Parameters)
    print("_____experment 1 Iris_____")
    print(results_df)  
    X_train1, y_train, X_test1, y_test
    
    clf=Sklean_LogisticRegression.LogisticRegression()
    clf = clf.fit(X_train1, y_train)
    yh=clf.predict(X_test1)
    sklearn_acc_LR=Sklean_accuracy_score(y_test, yh)
    print("sklearn_acc_LR="+str(sklearn_acc_LR))
    
    nbb=GaussianNB()
    nbb = nbb.fit(X_train1, y_train)
    yh=nbb.predict(X_test1)
    sklearn_acc_LR=Sklean_accuracy_score(y_test, yh)
    print("sklearn_acc_GaussianNB="+str(sklearn_acc_LR))
    
def main():
     #exp_1_ionosphere()
#     exp_1_Iris()
#     exp_1_WineQualityRed()
     exp_1_adult()
     
def experment1():
     exp_1_ionosphere()
     exp_1_Iris()
     exp_1_WineQualityRed()
     exp_1_adult()
if __name__ == "__main__":
    main()
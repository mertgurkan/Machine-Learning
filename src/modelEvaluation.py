import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import preProcessing  as prePros


def model_selection(x_train, y_train, ml_model,Normalization=True,FinalModel=False, k=5):
    """
    This method is used for model selection, it applies K-Fold cross validation for LR ONLY .
    """
   

   

        

    if FinalModel:
        itrations=1
        tot_err =  np.zeros((k,itrations))
        tot_acc = np.zeros((k,itrations))
        x_k_fold = np.array_split(x_train, k)
        y_k_fold = np.array_split(y_train, k)
        for i in range(0, k):
            x_k_fold_tr = x_k_fold.copy()
            y_k_fold_tr = y_k_fold.copy()

        
            x_k_fold_v = x_k_fold_tr[i]
            y_k_fold_v = y_k_fold_tr[i]        

        
            del(x_k_fold_tr[i])
            del(y_k_fold_tr[i])
            x_k_fold_tr = np.concatenate(x_k_fold_tr)
            y_k_fold_tr = np.concatenate(y_k_fold_tr)
        
        
        # Normalization 
            if  Normalization:            
                X_train,X_test=prePros.Normalization(x_k_fold_tr,x_k_fold_v)
        
            ml_model.fit(x_k_fold_tr, y_k_fold_tr)
            yh = ml_model.predict(x_k_fold_v)
            tot_acc[i,0],tot_err[i,0]  = ml_model.evaluate_acc(y_k_fold_v, yh)
        
    else:
        itrations=ml_model.num_iters
        tot_err =  np.empty([0,itrations])
        tot_acc = np.empty([0,itrations])
        x_k_fold = np.array_split(x_train, k)
        y_k_fold = np.array_split(y_train, k)
        
        
    
        for i in range(0, k):
            x_k_fold_tr = x_k_fold.copy()
            y_k_fold_tr = y_k_fold.copy()

        
            x_k_fold_v = x_k_fold_tr[i]
            y_k_fold_v = y_k_fold_tr[i]        

        
            del(x_k_fold_tr[i])
            del(y_k_fold_tr[i])
            x_k_fold_tr = np.concatenate(x_k_fold_tr)
            y_k_fold_tr = np.concatenate(y_k_fold_tr)
        
        
        # Normalization 
            if  Normalization:
                X_train,X_test=prePros.Normalization(x_k_fold_tr,x_k_fold_v)
                

        
            ml_model.fit(x_k_fold_tr, y_k_fold_tr)
        
        
            acc=np.zeros((1,itrations))
            err=np.zeros((1,itrations))

            
            for j in range(5,itrations):
                ml_model.set_w_final(ml_model.get_w_itration(j))
                yh = ml_model.predict(x_k_fold_v)
                acc[0,j],err[0,]  = ml_model.evaluate_acc(y_k_fold_v, yh)
            #print(acc)    
            tot_err=np.concatenate((tot_err,err),axis=0) 
            tot_acc=np.concatenate((tot_acc,acc),axis=0) 
          
   

                
    avg_err = np.sum(tot_err, axis = 0)/tot_err.shape[0]
    avg_acc = np.sum(tot_acc, axis = 0)/tot_acc.shape[0]
    std_cv=np.std(tot_acc, axis = 0)             

    return avg_acc,std_cv 


    
def model_selection_2(x_train, y_train, ml_model,Normalization=True, k=5):
    """
    This method is used for model selection, it applies K-Fold cross validation for NB only.

    """

    itrations=1
    tot_err =  np.empty([0,itrations])
    tot_acc = np.empty([0,itrations])
    x_k_fold = np.array_split(x_train, k)
    y_k_fold = np.array_split(y_train, k)
    for i in range(0, k):
        x_k_fold_tr = x_k_fold.copy()
        y_k_fold_tr = y_k_fold.copy()

        
        x_k_fold_v = x_k_fold_tr[i]
        y_k_fold_v = y_k_fold_tr[i]        

        
        del(x_k_fold_tr[i])
        del(y_k_fold_tr[i])
        x_k_fold_tr = np.concatenate(x_k_fold_tr)
        y_k_fold_tr = np.concatenate(y_k_fold_tr)
        
        
        # Normalization 
        if  Normalization:
            
            x_k_fold_v = x_k_fold_v - x_k_fold_v.mean(axis=0)
            x_k_fold_v = x_k_fold_v / np.abs(x_k_fold_v).max(axis = 0)
            x_k_fold_v=np.nan_to_num(x_k_fold_v)
            x_k_fold_v[x_k_fold_v == 0] = 1
            x_k_fold_tr = x_k_fold_tr - x_k_fold_tr.mean(axis=0)
            x_k_fold_tr = x_k_fold_tr / np.abs(x_k_fold_tr).max( axis = 0)
            x_k_fold_tr=np.nan_to_num(x_k_fold_tr)
            x_k_fold_tr[x_k_fold_tr == 0] = 1
        
        ml_model.fit(x_k_fold_tr, y_k_fold_tr)
        yh = ml_model.predict(x_k_fold_v)
        tot_acc[0,i],tot_err[0,i]  = ml_model.evaluate_acc(y_k_fold_v, yh)
        

          
   

                
    avg_err = np.sum(tot_err, axis = 0)/tot_err.shape[0]
    avg_acc = np.sum(tot_acc, axis = 0)/tot_acc.shape[0]
    std_cv=np.std(tot_acc, axis = 0)             

    return avg_acc,std_cv 



def plotResults(model, X_test, y_test, our_prediction, sklearn_predection, figuretitle):
    """
    plots the predictions of our custom classifier versus
    the ones from the sklean library, as well as the actual
    y values
    """
    fig = plt.figure(figsize=(8,5))

    pred = our_prediction.flatten()
    tp, tn, fp, fn = [], [], [], []

    for i in range(X_test.shape[0]):
        if y_test[i] == 1: 
            # Positive class
            if pred[i] == 1:
                tp.append((X_test[i,1], X_test[i,2])) # True positive
            else:
                fn.append((X_test[i,1], X_test[i,2])) # False negative
        else:
            # Negative class
            if pred[i] == 1:
                fp.append((X_test[i,1], X_test[i,2])) # False positive
            else:
                tn.append((X_test[i,1], X_test[i,2])) # True negative

    # Convert the lists of tupples into numpy arrays for visualization
    tp = np.array(tp, dtype='float')
    tn = np.array(tn, dtype='float')
    fp = np.array(fp, dtype='float')
    fn = np.array(fn, dtype='float')

    sns.scatterplot(x=tp[:,0], y=tp[:,1])
    sns.scatterplot(x=fp[:,0], y=fp[:,1])
    sns.scatterplot(x=tn[:,0], y=tn[:,1])
    sns.scatterplot(x=fn[:,0], y=fn[:,1])

    # Show probability threshold area
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes', norm=colors.Normalize(0., 1.), zorder=0)
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='white')

    fig.legend(labels=['True Positives', 'False Positives', 'True Negatives', 'False Negatives'], loc='upper left')
    plt.title('Linear Discriminant Analysis')
    plt.savefig("../img/models/%s" % figuretitle)
    plt.close()
    print("Created ../img/models/%s" % figuretitle)
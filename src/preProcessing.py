# coding: utf-8
import numpy as np
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer

import seaborn as sns
sns.set()
def isFloat(string):
  
    #checks floot
   
    try:
        float(string)
        return True
    except ValueError:
        return False
def get_Iris_data():
    file_path = '../data/Iris/Iris.csv'
    headers_Names= ["sepal_length","sepal_width", "petal_length","petal_width","y"]
    df  = pd.read_csv(file_path,delimiter=',',header=None,names=headers_Names)  
    return df
def get_ionosphere_data():
    file_path = '../data/ionosphere/ionosphere.csv'
    headers_Names= ["Re1","Im1","Re2","Im2","Re3","Im3","Re4","Im4","Re5","Im5","Re6","Im6","Re7","Im7","Re8","Im8","Re9","Im9","Re10","Im10","Re11","Im11","Re12","Im12","Re13","Im13","Re14","Im14","Re15","Im15","Re16","Im16","Re17","Im17","y"]
    df  = pd.read_csv(file_path,delimiter=',',header=None,names=headers_Names) 
    return df

def get_parkinsons_data():
    file_path = '../data/parkinsons/parkinsons.csv'
    headers_Names= ["name","MDVP_Fo(Hz)","MDVP_Fhi(Hz)","MDVP_Flo(Hz)","MDVP_Jitter(%)","MDVP_Jitter(Abs)","MDVP_RAP","MDVP_PPQ","Jitter_DDP","MDVP_Shimmer","MDVP_Shimmer(dB)","Shimmer_APQ3","Shimmer_APQ5","MDVP_APQ","Shimmer_DDA","NHR","HNR","y","RPDE","DFA","spread1","spread2","D2","PPE"]
    df  = pd.read_csv(file_path,delimiter=',',header=None,names=headers_Names) 
    return df

def get_adult_data():
    file_path = '../data/adult/adult.csv'
    df  = pd.read_csv(file_path, delimiter=',',header=None,names=["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","y"]) 
    return df

def get_WineQualityRed_data():
    file_path = '../data/WineQualityRed/WineQualityRed.csv'
    df  = pd.read_csv(file_path,delimiter=';') 
    return df

def flagNotFloat(df, column):
#flag non float
    for i in df.index:
        if not isFloat(df[column][i]):
            df.at[i, 'flag'] = 'not_float'
    return df


def lebelEncoder(df,df_index,df_lables):
    #encode using oneLabelEncoder
    df[df_index] = df[df_index].map(df_lables)
    return df

def createFlag(df):
    # create flag column and make it = Ok
    df['flag'] = 'ok'
    return df

def remove_malform_features(df,delatedFeatures):
    #delete un needed features
    df2=df.copy()
    for i in delatedFeatures:
        del df2[i]
    return df2
def delRows(df,flag_to_keep):
    # remove un needed instances
    print("Remaining instances:"+str(len(df[df['flag'] == 'ok'])) ) 
    return df[(df['flag']==flag_to_keep)]

def cleanSpecialChar(df):
    #replrase missing data by Nan
    df= df.replace(' ', '',regex=True)
    return  df.replace('\?',np.nan,regex=True)

def flagNan(df):
    #flag missing data
    nan_col = df.isnull().any(axis=1)
    j=0
    for i in nan_col.index:
        if nan_col.at[i]:
            df.at[i,'flag']='Missing'
            j+=1
    print("Instances with missing values:"+str(len(df[df['flag'] == 'Missing'])) )       
    return df

def labelEncoder(df,column_names): 
    #labelencodeer
    for i in column_names:
        oneHotData=df[i].unique()
        oneHotData.sort()       
        for k in df[i].index:
            if df.at[k,i]==oneHotData[0]:
                df.at[k,i]=1
            else:
                df.at[k,i]=0
    return df

def oneHotEncoder(df,column_names):
    #one hot encoder
    for i in column_names:
        oneHotData=df[i].unique()
        oneHotData.sort()
        for j in range(len(oneHotData)):
            newcolumn=i+'_'+str(oneHotData[j])
            df.insert(2,newcolumn, 0)
            for k in df[i].index:
                    if df.at[k,i]==oneHotData[j]:
                        df.at[k,newcolumn]=1
    for i in column_names:
        del df[i]        
    return df
def oneHotEncoder_2(df,column_names):
    #one hot encoder
    for i in column_names:
        oneHotData=df[i].unique()
        oneHotData.sort()
        for j in range(len(oneHotData)-1):
            newcolumn=i+'_'+str(oneHotData[j])
            df.insert(2,newcolumn, 0)
            for k in df[i].index:
                    if df.at[k,i]==oneHotData[j]:
                        df.at[k,newcolumn]=1
    for i in column_names:
        del df[i]        
    return df

def flagDuplicates(df):
# identify dublicated columns
    duplicate_rows_df = df[df.duplicated()]
    if duplicate_rows_df.shape[0] == 0:
        print("No duplicates")
        return df
    # If we reach this part of the function then we do have duplicates in the data
    for i in duplicate_rows_df.index:
        df.at[i, 'flag'] = 'duplicated'
    print(" Duplicated Instances:"+str(len(df[df['flag'] == 'duplicated'])) )
    return df


def flagOutliersReal(df, realColumns, n_std=2):
#identify outlires for real columns
    for I in realColumns:
        # sumamry statistics
        I_mean, I_std = np.mean(df[I]), np.std(df[I])
        # identify outliers
        I_off = I_std * n_std
        # this is the range of acceptable data
        lower, upper = I_mean - I_off, I_mean + I_off
        # outliers
        outliers = df[(df[I]<lower) | (df[I]>upper)]
        j=0
        for i in outliers.index:
            df.at[i, 'flag'] = 'outlier'
            j+=1
    print("outlier instances:"+str(len(df[df['flag'] == 'outlier'])) )
    return df


def flagData(raw_data):
    """
        The different flags will be:
        - 'ok' (no issue found with row)
        - 'not a float'
        - 'duplicate'
        - 'potential outlier'
    """ 
    # by default, flag everything as OK, this may change as problems are found
    columns_to_clean = raw_data.columns
    raw_data['flag'] = 'ok'

    # Test that all columns are made of float values
    for column in columns_to_clean:
        raw_data = flag_not_float(raw_data, column)

    # Flag duplicated data if any
    raw_data = flag_duplicates(raw_data)
    # Flag potential outliers
    raw_data = flag_outliers(raw_data, columns_to_clean)
    return raw_data


def getCleanedData(df, real_columns,binary_columns,delete_outliers=False):
    """
    Flag potential errors and delete them. 
    Deleting outliers is optionnal
    """
    df=createFlag(df)
    df=cleanSpecialChar(df)
    df=flagNan(df)
    df=flagDuplicates(df)
    if delete_outliers:
        df=flagOutliersReal(df, real_columns)
        
    df=delRows(df,'ok')
    binary_columns=removeBinFeature(df,binary_columns)
    df=deleteFeatures(df,binary_columns)
    #real_columns=removeReFeature(df,real_columns)
    #df=deleteFeatures(df,real_columns)
    # Drop rows containing missing values first


    # Drop the flag column before returning the clean data
    return df.drop('flag', axis=1)






def define_malform_features(df,featuresNames,limite):
    listelements= list()
    for i in featuresNames:
        count = df[i].to_numpy()
        max_m, counts_elements = np.unique(count, return_counts=True)
        counts_elements=counts_elements/len(count)
        if counts_elements.max()>=limite:
            listelements.append(i)
    return listelements

def removeReFeature(df,featuresNames,threshould=.01):# not completed
    listelements= list()
    for i in featuresNames:
        np_var = np.var(df[i])
        np_mean = np.mean(df[i])
        np_std=np.std(df[i])
        print(listelements)
        print(i)
        print(np_mean)
        print(np_std)
        if np_var<threshould:
            listelements.append(i)
    return listelements

def remove_Re_malform_Feature(df,threshould=.001):# not completed
    removed_features= []
    for i in df.columns:
        np_std=np.std(df[i])
        if np_std<threshould:
            removed_features.append(i)
    return removed_features

def computeYRange(df,limite):
    """
    Transform 'quality'[0,1,...,10] => 'y' [0,1]
    Wine dataset only
    """
    # this is to skip SettingWithCopyWarning from Pandas
    # Create the binary y column
    df['y'] = np.where(df['y'] >= limite, 1.0, 0.0)
    # Drop the 'quality' column
    return df



def prepData(df,column_names):
    for col in column_names:
        temp=df[col].unique()
        temp.sort()
        for j in range(len(temp)):
            df[col] = df[col].replace(temp[j],j)
    del df["flag"] 
    return df

def covertRealToCategoral(df,column,criteria,values):
    df[column] = np.select(criteria, values, 0)
    return df
    
def createCriteria(df2,initial,final):
    print(type(df2))
    criteria=[]
    for i in range(0,len(initial)):
        criteria.append(df2.between(initial(i),final(i))) 
        values=np.linspace(0,len(initial)-1)
    return criteria,values
                
# Boxplot (outliers)

def draw_boxplots(raw_data, columns_to_clean):
    plt.figure(figsize=(4, 4))
    raw_data[columns_to_clean].boxplot()
    plt.show()
#draw_boxplots(raw_data, columns_to_clean)


# Histogram (Frequency)

def draw_histogram(dataset, bins=50, width=12, height=5):
    dataset.hist(bins=bins, figsize=(width,height))

def Data_spliting(dataset,data_test, x_inputs, y_output, training_percent=0.85, shuffle=True, random_seed_value=42):
    
    if data_test.empty:
        np.random.seed(random_seed_value)
        dataset = dataset.sample(frac=1) #shuffling ,frac=1 returns all row in random order
    
    #Spliting on the shuffled data
        row_nom          = len(dataset.index)
        splitting_index  = int(training_percent*row_nom)
    #Data Spliting
        training_dataset = dataset[:splitting_index]
        testing_dataset  = dataset[splitting_index:]
        X_train = training_dataset[x_inputs].to_numpy()
        y_train = training_dataset[y_output].to_numpy()
    
    
        X_test  = testing_dataset[x_inputs].to_numpy()
        y_test  = testing_dataset[y_output].to_numpy()
        print(dataset.shape)
        print(X_train.shape)
        print(X_test.shape)
    else:
        training_dataset = dataset
        testing_dataset  = data_test            
        X_train = training_dataset[x_inputs].to_numpy()
        y_train = training_dataset[y_output].to_numpy()
    
    
        X_test  = testing_dataset[x_inputs].to_numpy()
        y_test  = testing_dataset[y_output].to_numpy()    
    #returns

    
#    X_test = X_test - X_test.mean(axis=0)
#    X_test = X_test / np.abs(X_test).max( axis = 0)
#    X_test=np.nan_to_num(X_test)
#    X_test[X_test == 0] = 1    
#    
#    X_train = X_train - X_train.mean(axis=0)
#    X_train = X_train / np.abs(X_train).max( axis = 0)
#    X_train=np.nan_to_num(X_train)
#    X_train[X_train == 0] = 1 
#    
    return X_train, y_train, X_test, y_test

def Data_spliting_2(dataset,data_test, x_inputs, y_output, training_percent=0.85, shuffle=True, random_seed_value=42):
    
    if data_test.empty:
        np.random.seed(random_seed_value)
        dataset = dataset.sample(frac=1) #shuffling ,frac=1 returns all row in random order
    
    #Spliting on the shuffled data
        row_nom          = len(dataset.index)
        splitting_index  = int(training_percent*row_nom)
    #Data Spliting
        training_dataset = dataset[:splitting_index]
        testing_dataset  = dataset[splitting_index:]
    else:
        training_dataset = dataset
        testing_dataset  = data_test            
    
    #returns
    X_train = training_dataset[x_inputs].to_numpy()
    y_train = training_dataset[y_output].to_numpy()
    
    
    X_test  = testing_dataset[x_inputs].to_numpy()
    y_test  = testing_dataset[y_output].to_numpy()
   
    

    
    return X_train, y_train, X_test, y_test

def get_cleaned_data_ionosphere(df, real_columns,binary_columns,corr_limite=.3, corr_flag=False,limite=.8,delete_outliers=False):
    """
    Flag potential errors and delete them. 
    Deleting outliers is optionnal
    """
    encodingMap={'b': 0, 'g': 1}
    lebelEncoder(df,"y",encodingMap)
    df=createFlag(df)
    df=cleanSpecialChar(df)
    df=flagNan(df)
    df=flagDuplicates(df)
    
    real_columns= ["Re2","Im2","Re3","Im3","Re4","Im4","Re5","Im5","Re6","Im6","Re7","Im7","Re8","Im8","Re9","Im9","Re10","Im10","Re11","Im11","Re12","Im12","Re13","Im13","Re14","Im14","Re15","Im15","Re16","Im16","Re17","Im17"]
 
    if delete_outliers:
        df=flagOutliersReal(df, real_columns)
        
    df=delRows(df,'ok')
    removed_features=define_malform_features(df,binary_columns,.9)
    df=remove_malform_features(df,removed_features)


    del df['flag']

    corr_flag=True
    if corr_flag:
        removed_features=findWeakfeatures(df,.1)
        df=remove_malform_features(df,removed_features)
    return df
# adult data cleaning 
def get_cleaned_data_adult(df, real_columns,binary_columns,corr_limite=.5, corr_flag=False,limite=.8,delete_outliers=False):
    del df["education"]
    del df["fnlwgt"]     
    df=createFlag(df)
    df=cleanSpecialChar(df)
    
    df['workclass'] = df['workclass'].replace(['Without-pay', 'Never-worked'], 'Unemployed')
    df['workclass'] = df['workclass'].replace(['State-gov', 'Local-gov'], 'Gov')
    df['workclass'] = df['workclass'].replace(['Self-emp-inc', 'Self-emp-not-inc'], 'Private')

    df['marital-status'] = df['marital-status'].replace(['Divorced', 'Separated', 'Widowed'], 'Not_Married')
    df['marital-status'] = df['marital-status'].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'],'Married')
    
    df['native-country'] = df['native-country'].replace(['Canada', 'Cuba', 'Dominican-Republic', 'El-Salvador','Guatemala','Haiti','Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Puerto-Rico', 'Trinadad&Tobago','United-States'], 'n_Amrica')
    df['native-country'] = df['native-country'].replace(['Cambodia', 'China', 'Hong', 'India', 'Iran', 'Japan', 'Laos','Philippines','Taiwan', 'Thailand', 'Vietnam'], 'Asia')
    df['native-country'] = df['native-country'].replace(['England', 'France', 'Germany', 'Greece', 'Holand-Netherlands','Hungary','Ireland', 'Italy', 'Poland', 'Portugal', 'Scotland','Yugoslavia'], 'europe')
    df['native-country'] = df['native-country'].replace(['Columbia', 'Ecuador', 'Peru'], 's_Amrica')
    df['native-country'] = df['native-country'].replace(['South', '?'], 'Other')
    
    
    df=flagNan(df)
    df=flagDuplicates(df)
    if delete_outliers:
        df=flagOutliersReal(df, real_columns)
        
    df=delRows(df,'ok')
    z=["workclass","marital-status","occupation","relationship","race","sex","native-country","y"]
    df=prepData(df,z)

        
    initial=np.array([0,26,51])
    final=np.array([25,50,100])
    column=["age"]
    
    criteria=[]
    for i in range(0,len(initial)):
        criteria.append(df['age'].between(initial[i],final[i])) 
        values=np.linspace(0,len(initial)-1,len(initial))
   
    #criteria,values=createCriteria(df[column],initial,final)
    df=covertRealToCategoral(df,column,criteria,values)

    initial=np.array([0,35,46])
    final=np.array([34,45,100])

    column=["hours-per-week"]
    criteria=[]
    for i in range(0,len(initial)):
        criteria.append(df['hours-per-week'].between(initial[i],final[i])) 
        values=np.linspace(0,len(initial)-1,len(initial))    
    
    df=covertRealToCategoral(df,column,criteria,values)

    
    criteria=[]
    column=["capital-gain"]
    median=df[column].median()
    initial=np.array([0,2])
    final=np.array([1,99999])
    for i in range(0,len(initial)):
        criteria.append(df['capital-gain'].between(initial[i],final[i])) 
        values=np.linspace(0,len(initial)-1,len(initial))  
        

    #criteria,values=createCriteria(df,column,initial,final)
    df=covertRealToCategoral(df,column,criteria,values)
 
    criteria=[]
    column=["capital-loss"]
    median=df[column].median()
    initial=np.array([0,2])
    final=np.array([1,50000])
    for i in range(0,len(initial)):
        criteria.append(df['capital-loss'].between(initial[i],final[i])) 
        values=np.linspace(0,len(initial)-1,len(initial)) 
    

    #criteria,values=createCriteria(df,column,initial,final)
    df=covertRealToCategoral(df,column,criteria,values)
    
    
    removed_features=findWeakfeatures(df,corr_limite)
    df=remove_malform_features(df,removed_features)
    
    y_temps = ["y","flag"]
    x_temps = [var for var in df.columns.tolist() if not var in y_temps]
    removed_features=define_malform_features(df,x_temps,limite)
    df=remove_malform_features(df,removed_features)

    
    y_temps = ["y","flag","sex","capital-loss","capital-gain"]
    x_temps = [var for var in df.columns.tolist() if not var in y_temps]
    df=oneHotEncoder(df,x_temps)
        
    #print(df)
    #real_columns=removeReFeature(df,real_columns)
    #df=deleteFeatures(df,real_columns)
    # Drop rows containing missing values first
    #del df["flag"]
    # remove features with weak correlation
    #for i in df.columns:
        #if i != 'y':    
            #df[i]=  (df[i]-df[i].mean())/df[i].std()
    #return df

def define_variables(df,output):
    y_vars = [output]
    x_vars = [var for var in df.columns.tolist() if not var in y_vars]
    return x_vars,y_vars
def prepXforLR(X_train1,X_test1):
    X_train=X_train1.copy()
    X_test=X_test1.copy()
    b = np.ones((X_train.shape[0],X_train.shape[1]+1))
    b[:,1:] = X_train
    X_train=b
    b=np.ones((X_test.shape[0],X_test.shape[1]+1))
    b[:,1:] = X_test
    X_test=b
    return X_train,X_test
  
def findWeakfeatures(df,corr_limite):
    y_vars = ["y"]
    x_vars = [var for var in df.columns.tolist() if not var in y_vars]
    corr = df[x_vars + ['y']].corr()
    corr.style.background_gradient(cmap='coolwarm').set_precision(2)
    cor=np.array(corr)
    temp=cor[:,-1]
    temp2=temp[:-1]
    names=['features','Correlation']
    df3=pd.DataFrame(columns=['features','Correlation'])
    df3['features']=x_vars
    df3['Correlation']=temp2
    delatedFeatures=[]
    for i in df3['features'].index:
        if abs(df3.at[i,'Correlation'])< corr_limite:
               delatedFeatures.append(df3.at[i,'features'])
    return delatedFeatures
def get_cleaned_data_adult_2(df, real_columns,binary_columns,corr_limite=.5, corr_flag=False,limite=.8,delete_outliers=False):
    del df["education"]
    del df["fnlwgt"]     
    df=createFlag(df)
    df=cleanSpecialChar(df)
    
    df['workclass'] = df['workclass'].replace(['Without-pay', 'Never-worked'], 'Unemployed')
    df['workclass'] = df['workclass'].replace(['State-gov', 'Local-gov'], 'Gov')
    df['workclass'] = df['workclass'].replace(['Self-emp-inc', 'Self-emp-not-inc'], 'Private')

    df['marital-status'] = df['marital-status'].replace(['Divorced', 'Separated', 'Widowed'], 'Not_Married')
    df['marital-status'] = df['marital-status'].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'],'Married')
    
    df['native-country'] = df['native-country'].replace(['Canada', 'Cuba', 'Dominican-Republic', 'El-Salvador','Guatemala','Haiti','Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Puerto-Rico', 'Trinadad&Tobago','United-States'], 'n_Amrica')
    df['native-country'] = df['native-country'].replace(['Cambodia', 'China', 'Hong', 'India', 'Iran', 'Japan', 'Laos','Philippines','Taiwan', 'Thailand', 'Vietnam'], 'Asia')
    df['native-country'] = df['native-country'].replace(['England', 'France', 'Germany', 'Greece', 'Holand-Netherlands','Hungary','Ireland', 'Italy', 'Poland', 'Portugal', 'Scotland','Yugoslavia'], 'europe')
    df['native-country'] = df['native-country'].replace(['Columbia', 'Ecuador', 'Peru'], 's_Amrica')
    df['native-country'] = df['native-country'].replace(['South', '?'], 'Other')    
    
    
    df=flagNan(df)
    df=flagDuplicates(df)
    real_columns=["age","hours-per-week"]
    if delete_outliers:
        df=flagOutliersReal(df, real_columns)
        
    df=delRows(df,'ok')
    z=["workclass","marital-status","occupation","relationship","race","sex","native-country","y"]
    df=prepData(df,z)

        
    initial=np.array([0,26,51])
    final=np.array([25,50,100])
    column=["age"]
    
    criteria=[]
    for i in range(0,len(initial)):
        criteria.append(df['age'].between(initial[i],final[i])) 
        values=np.linspace(0,len(initial)-1,len(initial))
   
    #criteria,values=createCriteria(df[column],initial,final)
    df=covertRealToCategoral(df,column,criteria,values)

    initial=np.array([0,35,46])
    final=np.array([34,45,100])

    column=["hours-per-week"]
    criteria=[]
    for i in range(0,len(initial)):
        criteria.append(df['hours-per-week'].between(initial[i],final[i])) 
        values=np.linspace(0,len(initial)-1,len(initial))    
    
    df=covertRealToCategoral(df,column,criteria,values)

    
    criteria=[]
    column=["capital-gain"]
    median=df[column].median()
    initial=np.array([0,2])
    final=np.array([1,99999])
    for i in range(0,len(initial)):
        criteria.append(df['capital-gain'].between(initial[i],final[i])) 
        values=np.linspace(0,len(initial)-1,len(initial))  
        

    #criteria,values=createCriteria(df,column,initial,final)
    df=covertRealToCategoral(df,column,criteria,values)
 
    criteria=[]
    column=["capital-loss"]
    median=df[column].median()
    initial=np.array([0,2])
    final=np.array([1,50000])
    for i in range(0,len(initial)):
        criteria.append(df['capital-loss'].between(initial[i],final[i])) 
        values=np.linspace(0,len(initial)-1,len(initial)) 
    

    #criteria,values=createCriteria(df,column,initial,final)
    df=covertRealToCategoral(df,column,criteria,values)
    
    
    #removed_features=findWeakfeatures(df,corr_limite)
    #df=remove_malform_features(df,removed_features)
    
    y_temps = ["y","flag"]
    x_temps = [var for var in df.columns.tolist() if not var in y_temps]
    removed_features=define_malform_features(df,x_temps,.9)
    df=remove_malform_features(df,removed_features)

    
    y_temps = ["y","flag","sex","capital-loss","capital-gain"]
    x_temps = [var for var in df.columns.tolist() if not var in y_temps]
    
    df=oneHotEncoder(df,x_temps)
    #df = encode_onehot(df,x_temps)

    df=createFlag(df)
    df=flagDuplicates(df)
    df=delRows(df,'ok')
    del df['flag']
    
    
    
    
    #print(df)
    #real_columns=removeReFeature(df,real_columns)
    #df=deleteFeatures(df,real_columns)
    # Drop rows containing missing values first
    #del df["flag"]
    # remove features with weak correlation
    df.to_csv('mm.csv', encoding='utf-8')

    return df

def get_cleaned_data_Parkinsons(df, real_columns,binary_columns,corr_limite=.5, corr_flag=False,limite=.8,delete_outliers=False):
    """
    Flag potential errors and delete them. 
    Deleting outliers is optionnal
    """
    #encodingMap={'b': 0, 'g': 1}
    #lebelEncoder(df,"y",encodingMap)
    del df['name']

    df=createFlag(df)
    df=cleanSpecialChar(df)
    df=flagNan(df)
 
    if delete_outliers:
        df=flagOutliersReal(df, real_columns)
        
    df=delRows(df,'ok')
    
    df=flagDuplicates(df)    
    #removed_features=define_malform_features(df,binary_columns,limite)
    #df=remove_malform_features(df,removed_features)
    #real_columns=removeReFeature(df,real_columns)
    #df=deleteFeatures(df,real_columns)
    # Drop rows containing missing values first
    del df['flag']
    # remove features with weak correlation
    if corr_flag:
        removed_features=findWeakfeatures(df,corr_limite)
        df=remove_malform_features(df,removed_features)
    #for i in df.columns:
        #if i != 'y':    
            #df[i]=  (df[i]-df[i].mean())/df[i].std()
    return df

def get_cleaned_data_Skin(df, real_columns,binary_columns,corr_limite=.5, corr_flag=False,limite=.8,delete_outliers=False):
    """
    Flag potential errors and delete them. 
    Deleting outliers is optionnal
    """
    #encodingMap={'b': 0, 'g': 1}
    #lebelEncoder(df,"y",encodingMap)
    #del df['name']

    df=createFlag(df)
    df=cleanSpecialChar(df)
    df=flagNan(df)
 
    if delete_outliers:
        df=flagOutliersReal(df, real_columns)
        
    df=delRows(df,'ok')
    
    df=flagDuplicates(df)    
    #removed_features=define_malform_features(df,binary_columns,limite)
    #df=remove_malform_features(df,removed_features)
    #real_columns=removeReFeature(df,real_columns)
    #df=deleteFeatures(df,real_columns)
    # Drop rows containing missing values first
    del df['flag']
    # remove features with weak correlation
    if corr_flag:
        removed_features=findWeakfeatures(df,corr_limite)
        df=remove_malform_features(df,removed_features)

def get_cleaned_data_Iris(df, real_columns,binary_columns,corr_limite=.5, corr_flag=False,limite=.8,delete_outliers=False):
    """
    Flag potential errors and delete them. 
    Deleting outliers is optionnal
    """
    #encodingMap={'b': 0, 'g': 1}
    #lebelEncoder(df,"y",encodingMap)
    #del df['name']
    
    df['y'] = df['y'].replace(['Iris-setosa'], 0)
    df['y'] = df['y'].replace(['Iris-virginica', 'Iris-versicolor'],1)
    #encodingMap={'Iris-s': 0, 'Iris-v': 1}
    #lebelEncoder(df,"y",encodingMap)    
    
    
    df=createFlag(df)
    df=cleanSpecialChar(df)
    df=flagNan(df)
 
    if delete_outliers:#remove outlires
        df=flagOutliersReal(df, real_columns)
        
    
    
    df=flagDuplicates(df) 
    df=delRows(df,'ok')# remove FLAGED ROWS 
  
    del df['flag']
    # remove features with weak correlation
    if corr_flag:
        removed_features=findWeakfeatures(df,corr_limite)
        df=remove_malform_features(df,removed_features)
    # normalization    

    return df
def encode_onehot(df, cols):
    """
    One-hot encoding is applied to columns specified in a pandas DataFrame.
    
    Modified from: https://gist.github.com/kljensen/5452382
    
    Details:
    
    http://en.wikipedia.org/wiki/One-hot
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    vec = DictVectorizer()
    
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(outtype='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df
def get_cleaned_data_WineQualityRed(df, real_columns,binary_columns,corr_limite=.5, corr_flag=False,limite=.8,delete_outliers=False):
    """
    Flag potential errors and delete them. 
    Deleting outliers is optionnal
    """
    df['y']=df['y'].replace([0, 1, 2, 3,4,5], 0)
    df['y']=df['y'].replace([6,7,8], 1)
    #encodingMap={'b': 0, 'g': 1}
    #lebelEncoder(df,"y",encodingMap)
    #del df['name']

    df=createFlag(df)
    df=cleanSpecialChar(df)
    df=flagNan(df)
    real_columns=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"	]
    if delete_outliers:
        df=flagOutliersReal(df, real_columns)
        
    #df=flagDuplicates(df)
    df=delRows(df,'ok')
    del df['flag']    
    #
    removed_features=remove_Re_malform_Feature(df)
    df=remove_malform_features(df,removed_features)

    

    #real_columns=removeReFeature(df,real_columns)
    #df=deleteFeatures(df,real_columns)
    # Drop rows containing missing values first
    
    #df=deleteCorelatedFeatures(df)
    # remove features with weak correlation
    if corr_flag:
        removed_features=findWeakfeatures(df,corr_limite)
        df=remove_malform_features(df,removed_features)
    #for i in df.columns:
        #if i != 'y':    
            #df[i]=  (df[i]-df[i].mean())/df[i].std()
    return df
def Normalization(X_train,X_test):
    mean_=X_train.mean( axis = 0)
    #std_=X_train.std( axis = 0)
    max_=np.abs(X_train).max( axis = 0)
    X_train = (X_train -mean_)/max_
    #std_=np.abs(X_train).max( axis = 0)
    
    #X_train = X_train / np.abs(X_train).max( axis = 0) 
    #X_train=np.nan_to_num(X_train)
    X_train[:, 0] =  1 
    
              
    #mean_=X_train.mean( axis = 0)
    #std_=X_train.std( axis = 0)
    X_test = (X_test -mean_)
    X_test = X_test / max_
    #X_test=np.nan_to_num(X_test)
    X_test[:, 0] =  1 
    return X_train,X_test

def deleteCorelatedFeatures(df):
    corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8) ]
    df.drop(df[to_drop], axis=1)
    return df

def plot_dist_by_category(df, x_vars, y_cat, y_cont, img_name):
    y_cat='y'
    j=0
    fig = plt.figure()
    for x in x_vars:
        j+=1
        
        if j==1:
            plt.subplot(2, 2, 1)
            
        elif j==2:
            plt.subplot(2, 2, 2)
        elif j==3:
            plt.subplot(2, 2, 3)
        elif j==4: 
            plt.subplot(2, 2, 4)
        
    #------------------------------------------------------------
    # First plot : P(Xi|Y=0) versus P(Xi|Y=1)
    #------------------------------------------------------------
        
    
        pos_class   = df[df[y_cat] ==1] # positive class 
        neg_class   = df[df[y_cat] ==0] # negative class
        pos_mean    = float(np.mean(pos_class[[x]])) # mean, positive class
        neg_mean    = float(np.mean(neg_class[[x]])) # mean, negative class
    
        sns.distplot(pos_class[[x]], color='#7282ff')
        sns.distplot(neg_class[[x]], color='#e56666')
        plt.axvline(pos_mean, color='#7282ff')
        plt.axvline(neg_mean, color='#e56666')
        if j==1:
            fig.legend(labels=['%s for y=1' % x,'%s for y=0' % x], loc='upper left')
            
        elif j==2:
            fig.legend(labels=['%s for y=1' % x,'%s for y=0' % x], loc='upper right')
        elif j==3:
            fig.legend(labels=['%s for y=1' % x,'%s for y=0' % x], loc='lower left')
        elif j==4: 
            fig.legend(labels=['%s for y=1' % x,'%s for y=0' % x], loc='lower right')

    
 
    plt.savefig("../img/Preprocessing/%s.png" % img_name)
    plt.show()
    plt.close()
    print("Created %s.png" % img_name)
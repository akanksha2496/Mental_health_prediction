#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[102]:


# to do linear algebra
import numpy as np 
from numpy import mean
from numpy import std

# to do data processing,CSV file input/output
import pandas as pd 

# to plot graphs and figures
import matplotlib.pyplot as plt # to plot graphs
import seaborn as sns

# to get statistics calculation like min, max, inverse and all
from scipy import stats
from scipy.stats import randint

# to perform clustering methods
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

#to perform feature selection for reducing dimensions 
from sklearn.feature_selection import SelectKBest, chi2,f_classif,mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# to do preprocessing of dataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler

# To use machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB 
from mlxtend.classifier import StackingClassifier
from sklearn import svm

# to calculate validation score from predicted and actual labels
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,recall_score,precision_score,f1_score

#To use neural network models
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV

#to use bagging machine learning models
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

# to parse the arguments from the commandline
from argparse import ArgumentParser
import argparse
from argparse import RawTextHelpFormatter

# to reduse the dimension of the data
from sklearn.decomposition import PCA

# to load and save the models build during trainig
from joblib import dump, load

# for grid search on diffrent model paramters
from sklearn.model_selection import GridSearchCV

# deep learning realeted libraries from keras
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input
from keras.models import Sequential
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers import Input, Dense,LSTM,Dropout,BatchNormalization
from keras.models import Model
from keras.optimizers import SGD,Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import keras
from keras.regularizers import l2
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Bidirectional,Conv1D,Conv2D,Dense,Dropout,Embedding,Flatten,GlobalMaxPool1D,LSTM,MaxPooling1D,MaxPooling2D,Reshape,Input,Lambda
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras import backend as K
from keras.initializers import GlorotNormal

# to perfrom cross validation using techniques like KFold
from sklearn.model_selection import KFold

# to supress the warnings
import warnings
warnings.filterwarnings("ignore")


# # CommandLine Arguments Passing

# In[3]:


# argument parser to read cmammndline inputs files and output files

parser=argparse.ArgumentParser(description='Please provide following arguments',formatter_class=RawTextHelpFormatter) # creating object of argumentParser to parse all inputs given via commmand Line
parser.add_argument("-i","-I","--input", type=str, nargs='+',required=True, help="Input: Dataset to work on and the model on which the dataset is to be evaluated ") # parsing and adding the argumnet value to variables inorder to access later
parser.add_argument("-o","-O","--output",type=str, help="Output: sample file")
args = parser.parse_args() # 

input_file=args.input
output_file=args.output
data_file=input_file[0] #  the dataset path from which the data is to be loaded into a dataframe later
model_loaded=input_file[1] # the model on which the data needs to be trained


# # Data Exploration

# In[5]:


# reading the csv file from the directory into a dataframe using read_csv  of pandas

data=pd.read_csv(data_file) 
data.index=range(data.shape[0])

# displaying the shape of the dataframe using shape() of pandas

print("Shape of the data:") 
print(data.shape)


# In[6]:


# displaying the head of the dataframe using head() of pandas

data.head()


# In[7]:


# producing the statsistics like minimum, maximum, statistics of the data using describe of pandas
   
data.describe()


# In[8]:


# giving information about the data types of each feature of the dataframe using info() of pandas
   
data.info()


# In[9]:


# replacing all NAN(Not a number) or missing values in categorical features with the default values

#deafult value for string
string_default='NaN'

# replcaing the NAN with default values
for feature in data.columns:
    if(data[feature].dtype!=np.int64): # checking data-type of the feature for NAN replacement
        data[feature]=data[feature].fillna(string_default) # using fillna of pandas to replace the NAN values
data.head(5)   


# In[10]:


#preprocessing the Gender feature to male,female and transgender values only

#converting all values to lower case for easy implementation using lower() of pandas
gender=data['Gender'].str.lower()

#getting all the unique values from the Gender feature  using unique() of pandas
gender=data['Gender'].unique()

#sorting the values with respect to male, female and transgender category
male=["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
transgender=["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]           
female=["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

#iteratiing over each row to replace noisy values with categorical values
for (row, col) in data.iterrows(): # iterating over rows using iterrows() of pandas
    if str.lower(col.Gender) in male:
        data['Gender'].replace(to_replace=col.Gender, value='male', inplace=True) # using replace() to replce a world with other in a dataframe of pandas
    if str.lower(col.Gender) in female:
        data['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)
    if str.lower(col.Gender) in transgender:
        data['Gender'].replace(to_replace=col.Gender, value='transgender', inplace=True)

# removing other nosiy data  by dropping them
rubbish_list = ['A little about you', 'p']
data=data[~data['Gender'].isin(rubbish_list)] # isin() is used to test if an item is in the column/row of dataframe.
print(data['Gender'].unique())


# In[11]:


# replcing all missing age with the median age for better assumption

data['Age'].fillna(data['Age'].median(), inplace = True) # using fillna() of pandas to fill the NAN value with the median using median() on feature

# for mor data exploration task adding a new column age specifying the age range in which the person lies 
temp=pd.Series(data['Age']) # converting to a Series data type using Series()
temp[temp<18]=data['Age'].median()
data['Age']=temp


temp=pd.Series(data['Age'])
temp[temp>120]=data['Age'].median()
data['Age']=temp
data['age_range']=pd.cut(data['Age'], [0,20,30,65,100], labels=["0-20", "21-30", "31-65", "66-100"], include_lowest=True)


# In[12]:


#replacing NAN of self employed column to No for better assumption and understanding
data['self_employed']=data['self_employed'].replace([string_default],'No') # using replace() to replace the NAN values with "No"
print(data['self_employed'].unique()) # using unique() to print the unique results


# In[13]:


#replacing NAN of work_interferecolumn to Don\'t know for better assumption and understanding
data['work_interfere']=data['work_interfere'].replace([string_default], 'Don\'t know' )# using replace() to replace the NAN values with "Don\'t know"
print(data['work_interfere'].unique())# using unique() to print the unique results


# In[14]:


#Converting all the categorical data to numeric data for machine learning alogorithms to work

labelDict = {} # intializing a dictionary 
for feature in data:
    le=preprocessing.LabelEncoder() # instantiating a LabelEncoder() object
    le.fit(data[feature]) # fitting the data to encode all categorical data to numbers using fit()
    data[feature]=le.transform(data[feature]) # transforming each feature into numbers using transfrom()
data=data.drop(['Country'], axis= 1) # dropping the Country featurs as it is irrevalant to the problem
data=data.drop(['Timestamp'], axis= 1)# dropping the Timestamp featurs as it is unique and can lead to noisy element in data
data.head()


# In[15]:


# correlation between the numerical values

corrmat=data.corr() # calcualting the correlation or similarity between the numerical features using the corr() of pandas
f,ax=plt.subplots(figsize=(12, 9)) # subplots helps in plotting figure for a given size
sns.heatmap(corrmat, vmax=.8, square=True) # using the seaborn heatmap() to show the correlation obatin using pandas corr()
plt.show() # used to display the plot created using matplolib show()

# correlation between the target label for top 15 features

k=15 # K is the number of top features to be  considered
cols=corrmat.nlargest(k,'treatment')['treatment'].index # nlargest() of corr() of pandas give sthe top 10 features with max correlation with a given feature
cm=np.corrcoef(data[cols].values.T) # the coefficients are obtained used numpy corrcoef()
sns.set(font_scale=1.25) 
hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 10},yticklabels=cols.values,xticklabels=cols.values) # displaying the heatmap using seaborn heatmap()
plt.show() # used to display the plot created using matplolib show()


# In[16]:


# correlation between the categorical values with target label

data_temp=data[data.columns[2:-1]] # getting the data for all categorical feature
chi_scores=chi2(data_temp,data['treatment']) # using chi2() of skelearn to get the correlation/similarity betwen the features
p_values=pd.Series(chi_scores[1],index=data_temp.columns) # obtaining correlation score as p-value
p_values.sort_values(ascending = False , inplace = True) # sorrting the p-values obtained  to get the least correlated features
plt.xlabel("Features") 
plt.ylabel("P-values") 
plt.title("Chi2 Test") 
p_values.plot.bar()


# In[17]:


# Distribiution and density by Age

plt.figure(figsize=(12,8)) # setting the figure size to be plotted using the figure() of matplolib
sns.distplot(data["Age"], bins=24) # using the distplot() of seaborn to show the distribution of age group present in the dataframe. Bins is used to divide age of person into a group in order to calulate the frequency
plt.title("Age Distribuition and Density ") # setting the title of the plot using matplotlib title()
plt.xlabel("Age") #setting label of x_axis  using xlabel of matplotlib
plt.ylabel("Distribuition and Density") #setting label of y_axis  using xlabel of matplotlib


# In[18]:


# exploration on the count of feamale and male who underwent mental illnes treatment

plt.figure(figsize=(12,8))# setting the figure size to be plotted using the figure() of matplolib
labels=['male','female'] # setting labels for x-axis
g=sns.countplot(x="treatment",data=data) # using countplot() of seaborn to plot histogram for categorical feature
g.set_xticklabels(labels) # using set_xticklabels() of seaborn to mark explicitly each value of x 
plt.title('Total Distribuition of Male v/s Female')# setting the title of the plot using matplotlib title()


# In[19]:


# exploration of age range for female, male and transgenders who underwent treatement

labels=["0-20", "21-30", "31-65", "66-100"] # intializing the labels

# factorplot() of seaborn helps in draw a categorical plot 
g=sns.factorplot(x="age_range",y="treatment",hue="Gender",data=data,kind="bar",ci=None,size=5,aspect=2,legend_out=True) 
g.set_xticklabels(labels)# using set_xticklabels() of seaborn to mark explicitly each value of x 
plt.title('Probability of mental health condition')
plt.ylabel('Probability x 100') # setting labels for y-axis using ylabel from matplotlib
plt.xlabel('Age')
new_labels=['female','male','transgender'] # intializing the Gender labels

for t,l in zip(g._legend.texts, new_labels): t.set_text(l) # _legend of seaborn helps to plot the legend key
g.fig.subplots_adjust(top=0.9,right=0.8) # plotting using subplot()
plt.show()


# In[20]:


# exploration of female, male and transgenders who underwent treatement and has family medicle history

labels=['No','Yes']# intializing the labels

# factorplot() of seaborn helps in draw a categorical plot 
g = sns.factorplot(x="family_history", y="treatment", hue="Gender", data=data, kind="bar", ci=None, size=5, aspect=2, legend_out = True)
g.set_xticklabels(labels)# using set_xticklabels() of seaborn to mark explicitly each value of x 
plt.title('Probability of mental health condition')
plt.ylabel('Probability x 100')
plt.xlabel('Family History')  # setting labels for x-axis using xlabel from matplotlib

new_labels=['female','male','transgender'] 
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)# _legend of seaborn helps to plot the legend key
g.fig.subplots_adjust(top=0.9,right=0.8) #positioning the legend using subplots_adjust() of matplotlib
plt.show()


# In[21]:


# exploration of female, male and transgenders who underwent treatement and had proper care options available to them

labels=['No','Not sure Care Options','Yes'] # intializing the labels

# factorplot() of seaborn helps in draw a categorical plot 
g = sns.factorplot(x="care_options", y="treatment", hue="Gender", data=data, kind="bar", ci=None, size=5, aspect=2, legend_out = True)
g.set_xticklabels(labels)# using set_xticklabels() of seaborn to mark explicitly each value of x 
plt.title('Probability of mental health condition')
plt.ylabel('Probability x 100')
plt.xlabel('Care options')
new_labels=['female','male','transgender'] 
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)# _legend of seaborn helps to plot the legend key
g.fig.subplots_adjust(top=0.9,right=0.8) #positioning the legend using subplots_adjust() of matplotlib
plt.show()


# In[22]:


# exploration of female, male and transgenders who underwent treatement and have impact due to work pressure

labels=["Don't know",'Never','Often','Rarely','Sometimes' ]# intializing the labels

# factorplot() of seaborn helps in draw a categorical plot 
g = sns.factorplot(x="work_interfere", y="treatment", hue="Gender", data=data, kind="bar", ci=None, size=5, aspect=2, legend_out = True)
g.set_xticklabels(labels)# using set_xticklabels() of seaborn to mark explicitly each value of x
plt.title('Probability of mental health condition')
plt.ylabel('Probability x 100')
plt.xlabel('Work interfere')
new_labels=['female','male','transgender'] 
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)# _legend of seaborn helps to plot the legend key
g.fig.subplots_adjust(top=0.9,right=0.8)#positioning the legend using subplots_adjust() of matplotlib
plt.show()


# # Data Preprocessing 

# In[23]:


# normalizing the values of age feature between 0 and 1

scaler=MinMaxScaler() # using MinMaxScaler() from skelarn to bring the values in each column between 0 to 1. Hence normalizing the value to avoid bias.
data['Age']=scaler.fit_transform(data[['Age']]) # using fit() to train the model and transform() to alter the values passed to a new set of values
data.head()


# In[24]:


# rearranging the columns for better exploration

temp=data['treatment']
data.drop(columns=['treatment'],inplace=True) # drop() of pandas help in dropping a particular column
data['treatment']=temp # maming a new column in dataframe
data.head()


# # Train and Test Split

# In[25]:


# Spllitting the dataset into train and test for having data for building the model and for testing the model on unseen data

X=data[data.columns[:-1]]# columns() give the list of all column in the dataframe
y=data.treatment # the target label of the dataset as treatment column

train_X,test_X,train_y,test_y=train_test_split(X, y, test_size=0.20, random_state=0) # train_test_split of sklearn helps in spliiting the dataset into train and test set using a split ratio
print("Train data shape:",train_X.shape)
print("Test data shape:",test_X.shape)
print("Train label data shape:",len(train_y))
print("Test label data shape:",len(test_y))


# # Feature Selection

# In[26]:


# Selecting the fetaures which can best represent the dataset using feature selection. Also reducing the dimension in order to avoid overfitting(remembering of train data) by the model.

model=VarianceThreshold() # VarianceThreshold of skelarn removes the features which has varinace belwow a givem threshold. Here Threshold is 0
model.fit(train_X,train_y) # fit() to train the model 
train_X=model.transform(train_X) # transform() to alter the values of train or test dataset accordingly
test_X=model.transform(test_X)
print("Train data shape:",train_X.shape)
print("Test data shape:",test_X.shape)
print("Train label data shape:",len(train_y))
print("Test label data shape:",len(test_y))


# In[27]:


# seleect k best

model=SelectKBest(chi2,k=20) # SelectKBest of skelarn use statitcs to compare the correlation of features with target label. The more the correlation the important the feature.
# model=SelectKBest(f_classif,k=20) # the statistics used could be chi2, anova, mutual information
# model=SelectKBest(mutual_info_classif,k=20)
model.fit(train_X,train_y) # fit() to train the model 
train_X=model.transform(train_X) # transform() to alter the values of train or test dataset accordingly
test_X=model.transform(test_X)
print("Train data shape:",train_X.shape)
print("Test data shape:",test_X.shape)
print("Train label data shape:",len(train_y))
print("Test label data shape:",len(test_y))


# In[28]:


# lda=LinearDiscriminantAnalysis() # LinearDiscriminantAnalysis() of skelarm is an embbeded method of feature selection. It selects the important feature using model trainig
# lda.fit(train_X,train_y) # fit() to train the model 
# train_X= lda.transform(train_X) # transform() to alter the values of train or test dataset accordingly
# test_X= lda.transform(test_X)
# print("Train data shape:",train_X.shape)
# print("Test data shape:",test_X.shape)
# print("Train label data shape:",len(train_y))
# print("Test label data shape:",len(test_y))


# # Clustering

# In[29]:


def fit(X,c): # training the  Kmean model
    kmeans=KMeans(n_clusters=c) # initialzing the Kmean clustering object using Kmeans()
    kmeans=kmeans.fit(X) # fitting the data using fit()
    return kmeans


# In[30]:


def predict(X_test,kmeans): # using the Kmeans model for predcition
    y_predicted = kmeans.predict(X_test) # predicting the data using the predict()
    return y_predicted


# In[31]:


def elbow_method(X): # using the elbow methods to ge the right value of number of cluters
    error = []
    for i in range(1,11): 
        kmeans_model = fit(X, i)# fitting the data using fit()
        error.append(kmeans_model.inertia_) # loading the interia_ paramter

    plt.plot(range(1,11),error,'b', marker = 'o', linewidth = 2, markersize = 7) # plotting the graph using plot() of matplotlib
    plt.title('The Elbow Method Graph')
    plt.xlabel('Number of clusters')
    plt.ylabel('Error (Sum of squared distances)')
    plt.show()


# In[32]:


def visualize_clusters(X, optimal_cl, X_pca): # visulaizing the clusters obtained
    kmeans = fit(X_pca, optimal_cl)
    dump(kmeans, 'Kmeans_Model.joblib') # saving the model using joblib's dump() method
    kmeans = load('Kmeans_Model.joblib') # loading the model using joblib's load() method

    y_kmeans = predict(X_pca, kmeans) # predicting the data using the predict()
    plt.scatter(X_pca[y_kmeans==0, 0], X_pca[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1') # plotting the scatter graph using scatter() of matplotlib
    plt.scatter(X_pca[y_kmeans==1, 0], X_pca[y_kmeans==1, 1], s=100, c='skyblue', label ='Cluster 2')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='purple',marker='*', label = 'Centroids')
    plt.title('Clusters of Iris data')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(loc='upper left')
    plt.show()
    return kmeans, y_kmeans


# In[33]:


def calc_accuracy(y_test, y_pred): # calculating the accuracy of the model
    y_test=np.asarray(y_test)
    match = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            match +=1
    return match/len(y_test)


# In[34]:


def reduce_features(X, X_test): # using the PCA trasnformation for features redcution
    pca = PCA(n_components=2) # defining PCA object for appling PCA
    new_X = pca.fit_transform(X) # fiiting and transforming the train and test to required dimension using fit_transform()
    new_X_test = pca.transform(X_test)
    return new_X, new_X_test


# In[35]:


def prediction_on_optimal(X_train, X_test, y_train, y_test, optimal_cl, kmeans_model, y_train_pred): # making predcition and evaluating results
    y_test_pred = predict(X_test, kmeans_model)# predicting the data using the predict()
    train_accuracy = calc_accuracy(y_train, y_train_pred)# calculating the accuracy of the model
    validation_accuracy = calc_accuracy(y_test, y_test_pred)# calculating the accuracy of the model
    print("Train accuracy: ", train_accuracy)
    print("Validation accuracy: ", validation_accuracy)


# In[38]:


elbow_method(train_X) #using the elbow methods to ge the right value of number of cluters
optimal_cl = 2
print("Optimal number of clusters found is %s \nAs at %s this is an elbow point"% (optimal_cl, optimal_cl))
X_pca, X_test_pca = reduce_features(train_X,test_X) # using the PCA trasnformation for features redcution
model, y_train_pred = visualize_clusters(np.array(train_X), optimal_cl, X_pca)# visulaizing the clusters obtained
prediction_on_optimal(train_X, X_test_pca,train_y,test_y, optimal_cl, model, y_train_pred)# making predcition and evaluating results




# # Machine Learning Model Training

# In[60]:


# saving the predcitons made to a file

def savingPredictions(y_pred,length):
    results=pd.DataFrame({'Index':range(length), 'Treatment': y_pred})
    results.to_csv(output_file,index=False)


# In[81]:


# evaluations of the model based on the predcited labels and the target labels

def evaluation(model,y_test,y_pred):
    print('Accuracy:', metrics.accuracy_score(y_test, y_pred)) # using accuracy_score() of skelearn to get the accuracy of teh precitions made.
    
    # evaluatiing the confusion matris using skelarn confusion_matrix()
    confusion = metrics.confusion_matrix(y_test, y_pred)
    tp=confusion[1, 1]
    tn=confusion[0, 0]
    fp=confusion[0, 1]
    fn=confusion[1, 0]
    sns.heatmap(confusion,annot=True,fmt="d") 
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # calculating the specificity
    specificity=tn / (tn + fp)
    print('Specificity:',specificity)
    
    # calculating the f1_score
    f1_value=f1_score(y_test,y_pred)
    print('F1-Score:',f1_value)
    
    # calculating the precision
    precision=precision_score(y_test,y_pred)
    print('Precision:',precision)
    
    # calculating the recall/sensitivity
    recall=recall_score(y_test,y_pred)
    print('Recall/Sensitivity:',recall)
    
    # calculating the Receiver Operating Characteristic area under curve score
    print('AUC Score:', metrics.roc_auc_score(y_test, y_pred))
    
    # calculate cross-validated AUC
    print('Cross-validated AUC:', cross_val_score(model,train_X,train_y,cv=5,scoring='roc_auc').mean())

    # plotting the ROC AUC Curve
    y_pred_prob=model.predict_proba(test_X)[:, 1]
    y_pred_prob=y_pred_prob.reshape(-1,1) 
    y_pred_class=binarize(y_pred_prob, 0.3)[0]
    roc_auc=metrics.roc_auc_score(y_test, y_pred_prob)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve ')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()


# In[ ]:


def gridSearch(model,param_grid):
    grid = GridSearchCV(model, param_grid, cv=10, scoring='accuracy')# instantiate the grid object using the GRIDSearchCV
    grid.fit(X, y) # fiiting the data for training using fit()
    grid.cv_results_#viewing the resulst from grid search using grid_scores_
    print('GridSearch best score', grid.best_score_) # dispalying  the best model from grid Search
    print('GridSearch best params', grid.best_params_)
    print('GridSearch best estimator', grid.best_estimator_)


# In[40]:


# defining model for Logistic Regression

def logisticRegression():
    logreg = LogisticRegression() # instantiating object for LogisticRegressio
    logreg.fit(train_X,train_y) # to train a logistic regression model on the training set using fit()
    y_pred= logreg.predict(test_X) # to make class predictions for the testing set using predict
    print('===================Logistic Regression===============')
    accuracy_score=evaluation(logreg,test_y,y_pred)
    savingPredictions(y_pred,len(test_X))
    
# logisticRegression()


# In[64]:


# defining model for Naive Bayes

def naiveBayes():
    logreg=GaussianNB() # instantiating object for Naive Bayes
    logreg.fit(train_X,train_y) # to train a Naive Bayes model on the training set using fit()
    y_pred=logreg.predict(test_X) # to make class predictions for the testing set using predict
    print('===================Naive Bayes===============')
    accuracy_score=evaluation(logreg,test_y,y_pred)
    savingPredictions(y_pred,len(test_X))
    
# naiveBayes()


# In[46]:


# defining model for Support Vector Machine

def supportVectorMachine():
    logreg=svm.SVC(kernel='rbf', class_weight='balanced',C=1,gamma=0.005,random_state=9,probability=True) # instantiating object for SVM
    logreg.fit(train_X,train_y) # to train a svm model on the training set using fit()
    y_pred=logreg.predict(test_X) # to make class predictions for the testing set using predict
    print('===================Support Vector Machine===============')
    accuracy_score=evaluation(logreg,test_y,y_pred)
    savingPredictions(y_pred,len(test_X))
    
# supportVectorMachine()


# In[123]:


# defining model for Decision Tree Classifier

def decisionClassifier():
#     tree = DecisionTreeClassifier() # instantiating the model
#     featuresSize=20
#     creating a parameter grid to map the parameter names to the values that should be searched
#     param_grid = {"max_depth":[3, None],
#               "max_features":[5,10,15,20],
#               "min_samples_split":[2,5,9],
#               "min_samples_leaf":[2,5,9],
#               "criterion": ["gini", "entropy"]}
#     gridSearch(tree ,param_grid) # performing the gridSarch
    logreg=DecisionTreeClassifier(max_depth=3, min_samples_split=5, max_features=15, criterion='entropy', min_samples_leaf=5)# instantiating object for Decison Tree Classifier
    logreg.fit(train_X,train_y) # to train a svm model on the training set using fit()
    y_pred=logreg.predict(test_X) # to make class predictions for the testing set using predict
    print('===================Decision Tree Classifier===============')
    accuracy_score=evaluation(logreg,test_y,y_pred)
    savingPredictions(y_pred,len(test_X))
    
# decisionClassifier()


# In[127]:


# defining model for K Nearest Neighbours

def kNN():
#     knn=KNeighborsClassifier() # instantaiting the kNN model
#     k_range = list(range(1, 31)) # define the k parameter values that need be searched
#     param_grid = dict(n_neighbors=k_range)# create a parameter grid: map the parameter names to the values that should be searched
#     gridSearch(knn,param_grid) # performing the gridSarch
    logreg=KNeighborsClassifier(n_neighbors=27, weights='uniform')# instantiating object for K Nearest Neighbours
    logreg.fit(train_X,train_y) # to train a knn model on the training set using fit()
    y_pred=logreg.predict(test_X) # to make class predictions for the testing set using predict
    print('===================K Nearest Neighbours===============')
    accuracy_score=evaluation(logreg,test_y,y_pred)
    savingPredictions(y_pred,len(test_X))
    
# kNN()


# In[50]:


# defining model for  Bagging Decision Tree Classifier
def bagging():
    logreg=BaggingClassifier(DecisionTreeClassifier(), max_samples=1.0, max_features=1.0, bootstrap_features=False)# instantiating object for Bagging Decision Tree Classifier
    logreg.fit(train_X,train_y) # to train a Bagging Decision Tree Classifier model on the training set using fit()
    y_pred=logreg.predict(test_X) # to make class predictions for the testing set using predict
    print('===================Bagging Decision Tree Classifier===============')
    accuracy_score=evaluation(logreg,test_y,y_pred)
    savingPredictions(y_pred,len(test_X))
    
# bagging()


# In[129]:


# defining model for Random Forest
def randomForest():
#     forest = RandomForestClassifier(n_estimators = 20) # instantiating the model
#     featuresSize=20 
# #     creating a parameter grid to map the parameter names to the values that should be searched
#     param_grid = {"max_depth":[3, None],
#               "max_features":[5,10,15,20],
#               "min_samples_split":[2,5,9],
#               "min_samples_leaf":[2,5,9],
#               "criterion": ["gini", "entropy"]}
#     gridSearch(forest,param_grid) # performing the gridSarch
    logreg=RandomForestClassifier(max_depth = None, min_samples_leaf=8, min_samples_split=2, n_estimators = 20, random_state = 1)# instantiating object for Random Forest
    logreg.fit(train_X,train_y) # to train a  Random Forest model on the training set using fit()
    y_pred=logreg.predict(test_X) # to make class predictions for the testing set using predict
    print('===================Random Forest===============')
    accuracy_score=evaluation(logreg,test_y,y_pred)
    savingPredictions(y_pred,len(test_X))
    
# randomForest()


# In[66]:


# defining model for Boosting Decision Tree Classifier
def boosting():
    logreg=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=1), n_estimators=500)# instantiating object for Boosting of Decsion Tree Classifier
    logreg.fit(train_X,train_y) # to train Boosting Decision Tree Classifier model on the training set using fit()
    y_pred=logreg.predict(test_X) # to make class predictions for the testing set using predict
    print('===================Boosting Decision Tree Classifier===============')
    accuracy_score=evaluation(logreg,test_y,y_pred)
    savingPredictions(y_pred,len(test_X))
    
# boosting()


# In[54]:


def stacking():
    clf1=KNeighborsClassifier(n_neighbors=1)
    clf2=RandomForestClassifier(random_state=1)
    clf3=GaussianNB()
    lr=LogisticRegression()
    logreg=StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)# instantiating object for Stacking Classifier
    logreg.fit(train_X,train_y) # to train Stacking Classifier model on the training set using fit()
    y_pred=logreg.predict(test_X) # to make class predictions for the testing set using predict
    print('===================Stacking Classifier===============')
    accuracy_score=evaluation(logreg,test_y,y_pred)
    savingPredictions(y_pred,len(test_X))
    
# stacking()


# # Deep Learning Model

# In[73]:


def define_model():# Defining the model fuction 
    model=Sequential()  # defining a squential model using keras Sequential()
    model.add(Dense(10,input_dim=20,activation='relu'))  # adding three dense layers with 100, 50 and 50 neuraons using Dense()
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) #  using sigmoid as the activation function 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # compiling the model  using model.complie()
    return model


# In[126]:


# K-Fold Cross validation
def deepLearning():
    num_folds=5  # number of folds
    epochs=200 # number of epochs
    batch_size=64 # batch size
    cv_scores=[]  # storing all folds validation accuracy
    cv_scores_loss=[]
    kf=KFold(n_splits=num_folds,shuffle=True, random_state=1)  # intializig the kfold validation using the KFold()
    for train_index, test_index in kf.split(train_X,train_y):
        X_train, X_test = train_X[train_index],train_X[test_index]  # getting the train data for each fold
        y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]  # getting teh test data for each fold
        model=define_model()   # define the model to train on 
        model_info = model.fit(np.array(X_train), np.array(y_train), 
                                    verbose = False, batch_size=batch_size, 
                                    epochs =epochs)   # fitting the model using fit()
        scores=model.evaluate(X_test,y_test,verbose=0)  # evaluating on test of each fold using evaluate()

        cv_scores.append(scores[1]*100)
        cv_scores_loss.append(scores[0]*100)
    # printing the results for cross validation loss and accuracy
    print("Cross-validated Accuracy = %.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
    print("Cross-validated Loss = %.2f%% (+/- %.2f%%)" % (np.mean(cv_scores_loss), np.std(cv_scores_loss)))
    
    model=define_model()  # define the model to train on 
    model.fit(np.array(train_X),np.array(train_y),batch_size=batch_size,epochs =200)  # fitting the model using fit()
    print("Train Loss",(model.evaluate(test_X,test_y,verbose=0))[0])  # evaluating on train loss of each fold using evaluate()
    print("Test Loss",(model.evaluate(train_X,train_y,verbose=0))[0])  # evaluating on test  loss of each fold using evaluate()
    y_pred=(model.predict(test_X) > 0.5).astype("int32") # to make class predictions for the testing set using predict
    print('===================Deep Neural Network===============')
    
    print('Accuracy:', metrics.accuracy_score(test_y, y_pred)) # using accuracy_score() of skelearn to get the accuracy of teh precitions made.
    
    # evaluatiing the confusion matris using skelarn confusion_matrix()
    confusion = metrics.confusion_matrix(test_y, y_pred)
    tp=confusion[1, 1]
    tn=confusion[0, 0]
    fp=confusion[0, 1]
    fn=confusion[1, 0]
    sns.heatmap(confusion,annot=True,fmt="d") 
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # calculating the specificity
    specificity=tn / (tn + fp)
    print('Specificity:',specificity)
    
    # calculating the f1_score
    f1_value=f1_score(test_y,y_pred)
    print('F1-Score:',f1_value)
    
    # calculating the precision
    precision=precision_score(test_y,y_pred)
    print('Precision:',precision)
    
    # calculating the recall/sensitivity
    recall=recall_score(test_y,y_pred)
    print('Recall/Sensitivity:',recall)
    
    # calculating the Receiver Operating Characteristic area under curve score
    print('AUC Score:', metrics.roc_auc_score(test_y, y_pred))
#     print(len(y_pred.tolist()))
    savingPredictions(len(y_pred.tolist()),len(test_X))

# deepLearning()


# In[ ]:


if model_loaded=='LR': # for Logistic regression model
    logisticRegression()
elif model_loaded=='NB':# for Naive Bayes model
    naiveBayes()
elif model_loaded=='SVM':# for Support Vector Machine model
    supportVectorMachine()
elif model_loaded=='DT':# for Decision Tree model
    decisionClassifier()
elif model_loaded=='KNN':# for K Nearest Neigbour model
    kNN()
elif model_loaded=='Bagging':# for Bagging model
    bagging()
elif model_loaded=='RF':# for Random Forest model
    randomForest()
elif model_loaded=='Boosting':# for Boosting model
    boosting()
elif model_loaded=='Stacking':# for Stacking model
    stacking()
elif model_loaded=='DL':# for Deep Learning model
    deepLearning()
elif model_loaded=='ALL': # to run all the models
    logisticRegression()
    naiveBayes()
    supportVectorMachine()
    decisionClassifier()
    kNN()
    bagging()
    randomForest()
    boosting()
    stacking()
    deepLearning()


# In[203]:


# accuracy=[77.77,75.79,80.55,81.34,79.36,75.39,76.98,74.2,76.58,83.571]
# f1_score=[78.29,77.49,82.56,83.03,79.99,76.15,77.86,76.01,76.67,79.69]
# models=['LR','KNN','DT','RF','BAG','BT','ST','SVM','NB','DNN']



# ax = plt.subplot(111)
# ax.bar(x-0.2, accuracy, width=0.3, color='b', align='center')
# ax.bar(x, f1_score, width=0.3, color='g', align='center')
# # ax.xaxis_date()
# colors = {'Accuracy':'Blue', 'F1_Score':'green'}        
# labels = list(colors.keys())
# handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
# # plt.xticks()
# plt.ylabel("Percentage")
# plt.title("Model Comparison Based On F1-Score_Accuracy")
# plt.legend(handles, labels)
# plt.show()


# In[199]:


# f1_score=[78.29,77.49,82.56,83.03,79.99,76.15,77.86,76.01,76.67,79.69]
# recall=[82.11,85.36,94.3,93.49,84.55,80.48,82.92,83.73,78.86,86.217]
# precision=[74.81,70.94,73.41,74.67,75.91,72.26,73.38,69.59,74.61,74.12]
# models=['LR','KNN','DT','RF','BAG','BT','ST','SVM','NB','DNN']
# x=np.asarray([0,1,2,3,4,5,6,7,8,9])

# ax = plt.subplot(111)
# ax.bar(x-0.2, recall, width=0.3, color='b', align='center')
# ax.bar(x, precision, width=0.3, color='g', align='center')
# ax.bar(x+0.2, f1_score, width=0.3, color='r', align='center')
# # ax.xaxis_date()
# colors = {'Recall':'Blue', 'Precision':'green','F1_Score':'red'}         
# labels = list(colors.keys())
# handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
# # plt.xticks()
# plt.ylabel("Percentage")
# plt.title("Model Comparison Based On F1-Score_Recall_Precision")
# plt.legend(handles, labels)
# plt.show()


# In[ ]:





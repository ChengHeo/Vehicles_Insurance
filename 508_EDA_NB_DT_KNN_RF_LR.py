#!/usr/bin/env python
# coding: utf-8

# In[1]:


#package for read data visualization
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly as pt


# ## EDA

# In[2]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[3]:


print(train.shape)


# In[4]:


train.info()


# In[5]:


train.head()


# In[6]:


dup=train.duplicated()
train[dup]


# In[7]:


train.isnull().sum()


# In[8]:


train.Response.value_counts(normalize=True)


# In[9]:


for i in train.columns:
    if train[i].dtype=="object":
        print(train[i].value_counts())


# Check which of the values in categorial form and number form

# In[10]:


cat=[]
num=[]
for i in train.columns:
    if train[i].dtype=="object":
           cat.append(i)
            
    else:
            num.append(i)


# In[11]:


print(cat)
print(num)


# Checking the Data in more depth for the Numerical and Categorical Data.

# In[12]:


train[num].describe().T


# In[13]:


train[cat].describe().T


# In[14]:


#Select feature column names and target variable we are going to use for training
Gender  = {'Male': 1,'Female': 0} 
  
# traversing through dataframe 
# Gender column and writing 
# values where key matches 
train.Gender = [Gender[item] for item in train.Gender] 
print(train)


# In[15]:


#Select feature column names and target variable we are going to use for training
Vehicle_Age  = {'> 2 Years': 0,'1-2 Year': 1,'< 1 Year': 2} 
  
# traversing through dataframe 
# Vehicle_Age column and writing 
# values where key matches 
train.Vehicle_Age = [Vehicle_Age[item] for item in train.Vehicle_Age] 
print(train)


# In[16]:


#Select feature column names and target variable we are going to use for training
Vehicle_Damage  = {'Yes': 0,'No': 1} 
  
# traversing through dataframe 
# Vehicle_Age column and writing 
# values where key matches 
train.Vehicle_Damage = [Vehicle_Damage[item] for item in train.Vehicle_Damage] 
print(train)


# In[17]:


train.info()
train[0:10]


# In[18]:


#Select feature column names and target variable we are going to use for training
Gender  = {'Male': 1,'Female': 0} 
  
# traversing through dataframe 
# Gender column and writing 
# values where key matches 
test.Gender = [Gender[item] for item in test.Gender] 
print(test)


# In[19]:


#Select feature column names and target variable we are going to use for training
Vehicle_Damage  = {'Yes': 1,'No':0} 
  
# traversing through dataframe 
# Vehicle_Age column and writing 
# values where key matches 
test.Vehicle_Damage = [Vehicle_Damage[item] for item in test.Vehicle_Damage] 
print(test)


# In[20]:


#Select feature column names and target variable we are going to use for training
Vehicle_Age  = {'> 2 Years': 0,'1-2 Year': 1,'< 1 Year': 2} 
  
# traversing through dataframe 
# Vehicle_Age column and writing 
# values where key matches 
test.Vehicle_Age = [Vehicle_Age[item] for item in test.Vehicle_Age] 
print(test)


# ### Checking to make sure all values in number form

# In[21]:


test.info()
test[0:10]


# In[22]:


cat=[]
num=[]
for i in train.columns:
    if train[i].dtype=="object":
           cat.append(i)
            
    else:
            num.append(i)


# In[23]:


print(cat)
print(num)


# In[24]:


print("Any missing sample in training set:",train.isnull().values.any())
print("Any missing sample in test set:",test.isnull().values.any(), "\n")


# In[25]:


#Frequency distribution of classes"
train_outcome = pd.crosstab(index=train["Response"],  # Make a crosstab
                              columns="count")      # Name the count column

train_outcome


# In[26]:


train.Response.value_counts(normalize=True)


# ## Data Visualisation

# In[27]:


#Select feature column names and target variable we are going to use for training
features=['Gender','Age','Driving_License','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Annual_Premium',
'Policy_Sales_Channel','Vintage']
target = 'Response'


# In[28]:


#This is input which our classifier will use as an input.
train[features].head(10)


# In[29]:


#Display first 10 target variables
train[target].head(10).values


# In[30]:


train[num].describe().T


# In[31]:


for i in train.columns:
    if train[i].dtype=="object":
        print(train[i].value_counts())


# In[32]:


plt.figure(figsize=(15,15))
sns.lineplot(train["Age"],train["Annual_Premium"])


# Age and premium is sure to have a relation as we can see that with the increase in the age the premium increases. The lowest premium is of the age group range of 20-40 and there is a rise in premium after 50 and a steady rise till 80 where there are peaks of the premium .Premium and age are very important factor when it comes to analysing insurance data

# In[33]:


sns.countplot(train["Previously_Insured"], hue=train["Vehicle_Age"])
plt.show()


# In[34]:


sns.scatterplot(train["Annual_Premium"],train["Age"],hue=train["Region_Code"])


# Region doesn't shows a very strong connection to the Premium

# In[35]:


sns.pairplot(train)


# In[36]:


plt.figure(figsize=(10,10))
train.boxplot(vert=0)
plt.show()


# In[37]:


plt.figure(figsize=(10,8))
sns.heatmap(train.corr(),annot=True)
plt.show()


# As you can see above, we obtain the heatmap of correlation among the variables. The color palette in the side represents the amount of correlation among the variables. The lighter shade represents a high correlation.

# In[38]:


#visualisation the data
sns.countplot(x='Response', data=train, palette='bwr')
plt.show()


# In[39]:


#then defining attributes and target variable of the data
X = train.drop(['Response'], axis=1)
y = train['Response']


# In[40]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25 , random_state=10)


# ## Naive Bayes Model

# Naive Bayes is a classification technique based on the Bayes theorem. It is a simple but powerful algorithm for predictive modeling under supervised learning algorithms. The technique behind Naive Bayes is easy to understand. Naive Bayes has higher accuracy and speed when we have large data points.

# In[41]:


from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


# In[42]:


NB_model = GaussianNB()
NB_model.fit(X_train, y_train)


# In[43]:


y_train_predict1 = NB_model.predict(X_train)
model_score = NB_model.score(X_train, y_train)
print(model_score)
print(metrics.confusion_matrix(y_train, y_train_predict1))
print(metrics.classification_report(y_train, y_train_predict1))


# In[44]:


y_test.value_counts()


# In[45]:


y_test_predict1 = NB_model.predict(X_test)
model_scoreNB = NB_model.score(X_test, y_test)
print(model_scoreNB)
print(metrics.confusion_matrix(y_test, y_test_predict1))
print(metrics.classification_report(y_test, y_test_predict1))


# In[46]:


colors = ['#F93822','#FDD20E']

NB_model = GaussianNB()
NB_model.fit(X_train, y_train)
y_pred = NB_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Labelling the matrix
names = ['True Neg','False Pos','False Neg','True Pos']

# Counts of the test data and labelling it on the matrix
counts = [value for value in cm.flatten()]

# Get the % of the grand total on the matrix
percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]

labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm,annot = labels,cmap = colors,fmt ='')

# Classification Report
print(classification_report(y_test, y_pred))
print('GaussianNB Accuracy= {:.10f}'.format(accuracy_score(y_test, y_pred)))


# ## Decision Tree

# In[47]:


#import the decision tree package, Machine learning Algorithm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[48]:


#the we need splitting the data for doing modeling into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)


# In[49]:


DT_model = DecisionTreeClassifier()
DT_model.fit(X_train, y_train)


# In[50]:


y_train_predict2 = DT_model.predict(X_train)
model_score = DT_model.score(X_train, y_train)
print(model_score)
print(metrics.confusion_matrix(y_train, y_train_predict2))
print(metrics.classification_report(y_train, y_train_predict2))


# In[51]:


y_test.value_counts()


# In[52]:


y_test_predict2 = DT_model.predict(X_test)
model_scoreNB = DT_model.score(X_test, y_test)
print(model_score)
print(metrics.confusion_matrix(y_test, y_test_predict2))
print(metrics.classification_report(y_test, y_test_predict2))


# In[53]:


colors = ['#F93822','#FDD20E']


dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

    
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
    
# Labelling the matrix
names = ['True Neg','False Pos','False Neg','True Pos']
    
# Counts of the test data and labelling it on the matrix
counts = [value for value in cm.flatten()]
    
# Get the % of the grand total on the matrix
percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm,annot = labels,cmap = colors,fmt ='')
    
# Classification Report
print(classification_report(y_test, y_pred))
print('DT Accuracy= {:.10f}'.format(accuracy_score(y_test, y_pred)))


# In[54]:


dtc.score(X_test, y_test)*100


# ## KNN

# In[55]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


# In[56]:


Ks = 21
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print ("Listed below are the accuracy of the model respective to the value of K:")
mean_acc


# In[57]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.legend('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()


# In[58]:


error = []
# Calculating error for K values between 1 and 20
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    
    #finding the mean of the scenario when results of test 
    #not identical to value predicted by the model
    
    error.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(12, 6))
plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
print("Minimum error:-",min(error),"at K =",error.index(min(error))+1)


# In[59]:


KNN_model = KNeighborsClassifier(n_neighbors=i)
KNN_model.fit(X_train, y_train)


# In[60]:


y_train_predict3 = KNN_model.predict(X_train)
model_score = KNN_model.score(X_train, y_train)
print(model_score)
print(metrics.confusion_matrix(y_train, y_train_predict3))
print(metrics.classification_report(y_train, y_train_predict3))


# In[61]:


y_test.value_counts()


# In[62]:


y_test_predict3 = KNN_model.predict(X_test)
model_scoreNB = KNN_model.score(X_test, y_test)
print(model_score)
print(metrics.confusion_matrix(y_test, y_test_predict3))
print(metrics.classification_report(y_test, y_test_predict3))


# In[63]:


colors = ['#F93822','#FDD20E']


def model_evaluation(classifier):
    
    
    knn = KNeighborsClassifier(n_neighbors=classifier)    
    y_proba_knn = knn.fit(X_train, y_train).predict_proba(X_test)
    knn_predict = knn.predict(X_test)
    #roc = roc_auc_score(Y_test, y_proba_knn[:,1])
    
    # Confusion Matrix
    cm = confusion_matrix(y_test,knn_predict)
    
    # Labelling the matrix
    names = ['True Neg','False Pos','False Neg','True Pos']
    
    # Counts of the test data and labelling it on the matrix
    counts = [value for value in cm.flatten()]
    
    # Get the % of the grand total on the matrix
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm,annot = labels,cmap = colors,fmt ='')
    
    # Classification Report
    print(classification_report(y_test,knn_predict))
    print('KNN Accuracy= {:.10f}'.format(accuracy_score(y_test, knn_predict)))


# In[64]:


selected_k = mean_acc.argmax()+1
print ("Evaluate the model for K = ", selected_k)
model_evaluation(selected_k)


# ## Random forest

# Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction

# In[65]:


from sklearn.ensemble import RandomForestClassifier

RF_model=RandomForestClassifier(n_estimators=100,random_state=10)
RF_model.fit(X_train, y_train)


# In[66]:


y_train_predict4 = RF_model.predict(X_train)
model_score =RF_model.score(X_train, y_train)
print(model_score)
print(metrics.confusion_matrix(y_train, y_train_predict4))
print(metrics.classification_report(y_train, y_train_predict4))


# In[67]:


y_test_predict4 = RF_model.predict(X_test)
model_scoreRF = RF_model.score(X_test, y_test)
print(model_scoreRF)
print(metrics.confusion_matrix(y_test, y_test_predict4))
print(metrics.classification_report(y_test, y_test_predict4))


# In[68]:


colors = ['#F93822','#FDD20E']


rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

    
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
    
# Labelling the matrix
names = ['True Neg','False Pos','False Neg','True Pos']
    
# Counts of the test data and labelling it on the matrix
counts = [value for value in cm.flatten()]
    
# Get the % of the grand total on the matrix
percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm,annot = labels,cmap = colors,fmt ='')
    
# Classification Report
print(classification_report(y_test, y_pred))
print('RF Accuracy= {:.10f}'.format(accuracy_score(y_test, y_pred)))


# ## Logistic Regression

# Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression)

# In[69]:


from sklearn.linear_model import LogisticRegression


# In[70]:


model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
model_score = model.score(X_test, y_test)
print(model_score)


# In[71]:


y_train_predict5 = model.predict(X_train)
model_score =model.score(X_train, y_train)
print(model_score)
print(metrics.confusion_matrix(y_train, y_train_predict5))
print(metrics.classification_report(y_train, y_train_predict5))


# In[72]:


y_test_predict1 = model.predict(X_test)
model_scoreLR = model.score(X_test, y_test)
print(model_scoreLR)
print(metrics.confusion_matrix(y_test, y_test_predict1))
print(metrics.classification_report(y_test, y_test_predict1))


# In[73]:


colors = ['#F93822','#FDD20E']


lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

    
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
    
# Labelling the matrix
names = ['True Neg','False Pos','False Neg','True Pos']
    
# Counts of the test data and labelling it on the matrix
counts = [value for value in cm.flatten()]
    
# Get the % of the grand total on the matrix
percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm,annot = labels,cmap = colors,fmt ='')
    
# Classification Report
print(classification_report(y_test, y_pred))
print('LR Accuracy= {:.10f}'.format(accuracy_score(y_test, y_pred)))


# In[ ]:





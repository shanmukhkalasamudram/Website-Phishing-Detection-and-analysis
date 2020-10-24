#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, cross_val_score, KFold
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, precision_score, recall_score,  accuracy_score, precision_recall_curve
import warnings
warnings.filterwarnings('always')


# In[2]:


dataset =pd.read_csv(r"phishing.csv")


# In[3]:


dataset.head()


# In[4]:


dataset.columns


# In[5]:


dataset.shape


# In[6]:


dataset.isnull().sum()


# In[7]:


from sklearn.model_selection import train_test_split,cross_val_score


# In[8]:


X= dataset.drop(columns='class')
X.head()


# In[9]:


Y=dataset['class']
Y=pd.DataFrame(Y)
Y.head()


# In[10]:


train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.3,random_state=2)


# In[11]:


print(train_X.shape)# Features for training  
print(test_X.shape) # to find accuracy 
print(train_Y.shape) 
print(test_Y.shape)


# In[12]:


#1.Logistic Regression
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[13]:


logreg=LogisticRegression()
model_1=logreg.fit(train_X,train_Y)


# In[14]:


logreg_predict= model_1.predict(test_X)
logreg_predict


# In[15]:


accuracy_score(logreg_predict,test_Y)


# In[16]:


print(classification_report(logreg_predict,test_Y))


# In[17]:


def plot_confusion_matrix(test_Y, predict_y):
 C = confusion_matrix(test_Y, predict_y)
 A =(((C.T)/(C.sum(axis=1))).T)
 B =(C/C.sum(axis=0))
 plt.figure(figsize=(20,4))
 labels = [1,2]
 cmap=sns.light_palette("blue")
 plt.subplot(1, 3, 1)
 sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
 plt.xlabel('Predicted Class')
 plt.ylabel('Original Class')
 plt.title("Confusion matrix")
 plt.subplot(1, 3, 2)
 sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
 plt.xlabel('Predicted Class')
 plt.ylabel('Original Class')
 plt.title("Precision matrix")
 plt.subplot(1, 3, 3)
 sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
 plt.xlabel('Predicted Class')
 plt.ylabel('Original Class')
 plt.title("Recall matrix")
 plt.show()


# In[18]:


plot_confusion_matrix(test_Y, logreg_predict)


# In[19]:


#2. KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier


# In[20]:


knn=KNeighborsClassifier(n_neighbors=3)
model_2= knn.fit(train_X,train_Y)


# In[21]:


knn_predict=model_2.predict(test_X)


# In[22]:


accuracy_score(knn_predict,test_Y)


# In[23]:


print(classification_report(test_Y,knn_predict))


# In[24]:


plot_confusion_matrix(test_Y, knn_predict)


# In[25]:


#3. DecisionTree
from sklearn.tree import DecisionTreeClassifier


# In[27]:


decisiontree=DecisionTreeClassifier()
model_3=decisiontree.fit(train_X,train_Y)


# In[28]:


decisiontree_predict=model_3.predict(test_X)


# In[29]:


accuracy_score(decisiontree_predict,test_Y)


# In[30]:


print(classification_report(decisiontree_predict,test_Y))


# In[31]:


plot_confusion_matrix(test_Y, decisiontree_predict)


# In[32]:


#4 .RandomForest 
from sklearn.ensemble import RandomForestClassifier


# In[33]:


random=RandomForestClassifier()
model_4=random.fit(train_X,train_Y)


# In[34]:


random_predict=model_4.predict(test_X)


# In[35]:


accuracy_score(random_predict,test_Y)


# In[36]:


print(classification_report(random_predict,test_Y))


# In[37]:


plot_confusion_matrix(test_Y, random_predict)


# In[38]:


#5. Support Vector Machine 
from sklearn.svm import SVC


# In[39]:


svc=SVC()
model_5=svc.fit(train_X,train_Y)


# In[40]:


svm_predict=model_5.predict(test_X)


# In[41]:


accuracy_score(svm_predict,test_Y)


# In[42]:


print(classification_report(svm_predict,test_Y))


# In[43]:


plot_confusion_matrix(test_Y, svm_predict)


# In[44]:


#6.Gradient boosting Classifier
gbc=GradientBoostingClassifier()
model_6=gbc.fit(train_X,train_Y)


# In[45]:


gbc_predict=model_6.predict(test_X)
gbc_predict


# In[46]:


accuracy_score(gbc_predict,test_Y)


# In[47]:


print(classification_report(gbc_predict,test_Y))


# In[48]:


plot_confusion_matrix(test_Y, gbc_predict)


# In[49]:


#7.Adaboost Classifier
abc=AdaBoostClassifier()
model_7=abc.fit(train_X,train_Y)


# In[50]:


abc_predict=model_7.predict(test_X)
abc_predict


# In[51]:


accuracy_score(abc_predict,test_Y)


# In[52]:


print(classification_report(abc_predict,test_Y))


# In[53]:


plot_confusion_matrix(test_Y, abc_predict)


# In[54]:


#8. Naive Bayes
nbc = GaussianNB()


# In[55]:


model_8=nbc.fit(train_X,train_Y)


# In[56]:


nbc_predict=model_8.predict(test_X)
nbc_predict


# In[57]:


accuracy_score(nbc_predict,test_Y)


# In[58]:


print(classification_report(nbc_predict,test_Y))


# In[59]:


plot_confusion_matrix(test_Y, nbc_predict)


# In[60]:


#9.Stocastic Gradient
sgdc = SGDClassifier()
model_9=sgdc.fit(train_X,train_Y)


# In[61]:


sgdc_predict=model_9.predict(test_X)
sgdc_predict


# In[62]:


accuracy_score(sgdc_predict,test_Y)


# In[63]:


print(classification_report(sgdc_predict,test_Y))


# In[64]:


plot_confusion_matrix(test_Y, sgdc_predict)


# In[65]:


print('Logistic Regression Accuracy:',accuracy_score(logreg_predict,test_Y))
print('K-Nearest Neighbour Accuracy:',accuracy_score(knn_predict,test_Y))
print('Decision Tree Classifier Accuracy:',accuracy_score(decisiontree_predict,test_Y))
print('Random Forest Classifier Accuracy:',accuracy_score(random_predict,test_Y))
print('support Vector Machine Accuracy:',accuracy_score(svm_predict,test_Y))
print('Gradient Booster Classifier Accuracy:' , accuracy_score (gbc_predict, test_Y))
print('Ada Boost Classifier Accuracy:' , accuracy_score (abc_predict, test_Y))
print('Naive Bayes Classifier Accuracy:' , accuracy_score (nbc_predict, test_Y))
print('Stochastic Gradient Descent Classifier Accuracy:', accuracy_score (sgdc_predict, test_Y))


# In[66]:


# PREDICTION ON GIVEN INPUT USING RANDOM FOREST AS IT HAS THE HIGHEST ACCURACY
#1 represents it is phishing website and -1 represents it is not phishing website
index=input("Input number:")
ip=input("Ip:")
long=input("Long url:")
short=input("Short url:")
symbol=input("Symbol:")
redirect=input("Redirecting:")
fix=input("Prefix/Suffix:")
sub=input("Sub Domain:")
http=input("Https : ")
rl=input("Domain Reg len:")
fav=input("Favicon:")
port=input("Non Std Port:")
hturl=input("Https Domain Url:")
req=input("Request Url:")
acc=input("Anchor Url:")
links=input("Script tag links:")
server=input("Server Form Handler:")
info=input("Info Email:")
abnorm=input("Abnormal Url:")
forward=input("Website Forwarding:")
status=input("Status Bar Cust:")
right=input("Disable Right Click:")
pop=input("Using Pop up window?:")
ir=input("IFramed redirection:")
age=input("Age of domain:")
dns=input("DNS recording:")
traffic=input("Web Traffic:")
rank=input("Page Rank:")
gi=input("Google Index:")
point=input("Links pointing towards:")
stat=input("Stats Report:")
inp=[[int(index),int(ip),int(long),int(short),int(symbol),int(redirect),int(fix),int(sub),int(http),int(rl),int(fav),int(port),int(hturl),int(req),int(acc),int(links),int(server),int(info),int(abnorm),int(forward),int(status),int(right),int(pop),int(ir),int(age),int(dns),int(traffic),int(rank),int(gi),int(point),int(stat)]]
a=random.predict(inp)
print("Is it a phishing website:",a[0])


# In[67]:


#So it is not a phishing Website 


# In[ ]:





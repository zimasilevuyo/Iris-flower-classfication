import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.offline as pyo
import cufflinks as cf
from plotly.offline import init_notebook_mode,plot,iplot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import chart_studio.plotly as py

import graphviz
import os

pyo.init_notebook_mode(connected=True)
cf.go_offline()

iris=pd.read_csv('Iris.csv')


iris
iris.shape
iris.drop('Id',axis=1,inplace=True)

iris
#   Visualizing the data
px.scatter(iris,x='Species',y='PetalWidthCm',size='PetalWidthCm')
plt.bar(iris['Species'],iris['PetalWidthCm'])
px.bar(iris,x='Species',y='PetalWidthCm')
iris.iplot(kind='bar',x=['Species'],y=['PetalWidthCm'])
px.line(iris,x='Species',y='PetalWidthCm')
iris.rename(columns={'SepalLengthCm':'SepalLength','SepalWidthCm':'SepalWidth','PetalWidthCm':'PetalWidth','PetalLengthCm':'PetalLength'},inplace=True)


iris
px.scatter_matrix(iris,color='Species',title='Iris',dimensions=['SepalLength','SepalWidth','PetalWidth','PetalLength'])


#   Data PreProcessing
iris
X=iris.drop(['Species'],axis=1)
X
y=iris['Species']
y
le=LabelEncoder()
y=le.fit_transform(y)
y
X
X=np.array(X)
X
y
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
X_test
X_test.size

iris

#   Training the Decision Tree

#   Decision Tree
DT=tree.DecisionTreeClassifier()
DT.fit(X_train,y_train)


y_train.size
prediction_DT=DT.predict(X_test)
accuracy_DT=accuracy_score(y_test,prediction_DT)*100
accuracy_DT

y_test
prediction_DT

vis_data=tree.export_graphviz(DT,out_file=None, feature_names=iris.drop(['Species'],axis=1).keys(),class_names=iris['Species'].unique(),filled=True,rounded=True,special_characters=True)
graphviz.Source(vis_data)


Catagory=['Iris-Setosa','Iris-Versicolor','Iris-Virginica']

#   Lets predict on custom input value


X_DT=np.array([[1 ,1, 1, 1]])
X_DT_prediction=DT.predict(X_DT)


X_DT_prediction[0]
print(Catagory[int(X_DT_prediction[0])])

#   KNN Algorithm
#   Preprocessing for KNN

sc = StandardScaler().fit(X_train)  # Load the standard scaler
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_std,y_train)

predict_knn=knn.predict(X_test_std)
accuracy_knn=accuracy_score(y_test,predict_knn)*100

accuracy_knn

#   Lets predict on custom input value
X_knn=np.array([[7.7 ,3.5, 4.6, 4]])
X_knn_std=sc.transform(X_knn)
X_knn_std

X_knn_prediction=knn.predict(X_knn_std)
X_knn_prediction[0]
print(Catagory[int(X_knn_prediction[0])])

#   Finding Best K Value

k_range=range(1,26)
scores={}
scores_list=[]

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std,y_train)
    prediction_knn=knn.predict(X_test_std)
    scores[k]=accuracy_score(y_test,prediction_knn)
    scores_list.append(accuracy_score(y_test,prediction_knn))

scores_list

plt.plot(k_range,scores_list)
#   K MEANS Clustering

y
colormap=np.array(['Red','green','blue'])
fig=plt.scatter(iris['PetalLength'],iris['PetalWidth'],c=colormap[y],s=50)

iris
X

km=KMeans(n_clusters=3,random_state=2,n_jobs=4)
km.fit(X)

centers=km.cluster_centers_
print(centers)

km.labels_
Catagory_kmeans=['Iris-Versicolor', 'Iris-Setosa', 'Iris-Virginica']
Catagory_kmeans

colormap=np.array(['Red','green','blue'])
fig=plt.scatter(iris['PetalLength'],iris['PetalWidth'],c=colormap[km.labels_],s=50)

new_labels=km.labels_
fig,axes=plt.subplots(1,2,figsize=(16,8))
axes[0].scatter(X[:,2],X[:,3],c=y,cmap='gist_rainbow',edgecolor='k',s=150)
axes[1].scatter(X[:,2],X[:,3],c=y,cmap='jet',edgecolor='k',s=150)
axes[0].set_title('Actual',fontsize=18)
axes[1].set_title('Predicted',fontsize=18)

#   Lets predict on custom input value


X_km=np.array([[1 ,1, 1, 1]])



X_km_prediction=km.predict(X_km)
X_km_prediction[0]
print(Catagory_kmeans[int(X_km_prediction[0])])


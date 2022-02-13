from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
y=iris.target
feature_names=iris.feature_names
target_names=iris.target_names
print("feature names:",feature_names)
print("target names:",target_names)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=1)
print("x_train:",x_train.shape)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
#training the classifier
knn.fit(x_train,y_train)
#predict by test data
y_pred=knn.predict(x_test)
#calculating the accuracy of the clasifier
from sklearn import metrics
accuracy=metrics.accuracy_score(y_pred,y_test)
print('accuracy:',accuracy)
#to predict any out of sample target using this data and model
sample=[[1,3,4,5],[2,3,2,4]]
preds=knn.predict(sample)
pred_species=[iris.target_names[p] for p in preds]
print(pred_species)
#save the model
import joblib
joblib.dump(knn,'iris_knn.pkl')
#to load the model
knn=joblib.load('iris_knn.pkl')

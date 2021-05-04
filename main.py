import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

other_voices = pd.read_csv("voices.csv")
voices = pd.concat([other_voices, my_voices])
#%%
male = voices[voices.label =='male']
male = male.drop(['label'],axis=1)

female = voices[voices.label=='female']
female = female.drop(['label'],axis=1)

# dropping label 
x_data = voices.drop(['label','skew','kurt','meandom','maxdom','dfrange','mindom'],axis=1)
# x_data = voices[['sd','Q25','meanfun','IQR','sp.ent']]


voices.label = [0 if each == "male" else 1 for each in voices.label]
y = voices.label

# normalization
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))

#%% train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


#%% KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

k = 3
knn = KNeighborsClassifier(n_neighbors=k) #n_neighbors = k
 
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("knn score (k={}): {}".format(k, knn.score(x_test, y_test)))

cm_knn = confusion_matrix(y_test, prediction)
print("knn confusion matrix: {}".format(cm_knn))

tp_knn = cm_knn[0,0]
fp_knn = cm_knn[0,1]
tn_knn = cm_knn[1,1]
fn_knn = cm_knn[1,0]
precision_knn = tp_knn/(tp_knn+fp_knn)
recall_knn = tp_knn/(tp_knn+tn_knn)
f1score_knn = 2*(precision_knn*recall_knn)/(precision_knn+recall_knn)
print("knn precision: ",precision_knn)
print("knn recall: ",recall_knn)
print("knn f1 score: ",f1score_knn)

#%% k-fold cross validation knn
from sklearn.model_selection import cross_val_score
accuracies_knn = cross_val_score(estimator=knn, X=x_train, y=y_train, cv=5) # k-fold

print("knn cv average accuracy: ", np.mean(accuracies_knn)) #5 accuracy degerinin ortalaması

#%% Score and Error Rate
score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train, y_train)
    score_list.append(knn2.score(x_test, y_test))
plt.figure(1)
plt.plot(range(1,15),score_list)
plt.title('Score')
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.show()
plt.show()

error_list = []
for i in range(1, 20):
    knn3 = KNeighborsClassifier(n_neighbors = i)
    knn3.fit(x_train, y_train)
    prediction = knn3.predict(x_test)
    error_list.append(np.mean(prediction != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 20), error_list, color='red', linestyle='dashed', marker='o',
          markerfacecolor='blue', markersize=10)
plt.title('K vs Error Rate')
plt.xlabel('K')
plt.ylabel('Error Rate')

#%% SVM
from sklearn.svm import SVC
svm = SVC(C=100, gamma=0.1)
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
print("svm score: ", svm.score(x_test, y_test))
cm_svm = confusion_matrix(y_test, prediction)
print("svm confusion matrix: {}".format(cm_svm))
tp_svm = cm_svm[0,0]
fp_svm = cm_svm[0,1]
tn_svm = cm_svm[1,1]
fn_svm = cm_svm[1,0]
precision_svm = tp_svm/(tp_svm+fp_svm)
recall_svm = tp_knn/(tp_svm+tn_svm)
f1score_svm = 2*(precision_svm*recall_svm)/(precision_svm+recall_svm)
print("svm precision: ",precision_svm)
print("svm recall: ",recall_svm)
print("svm f1 score: ",f1score_svm)

#%% k-fold cross validation svm
from sklearn.model_selection import cross_val_score
accuracies_svm = cross_val_score(estimator=svm, X=x_train, y=y_train, cv=5) # k-fold

print("svm cv average accuracy: ", np.mean(accuracies_svm)) #5 accuracy degerinin ortalaması



#%% Grid Search
# from sklearn.model_selection import GridSearchCV 
# from sklearn.metrics import classification_report
# # defining parameter range 
# param_grid = {'C': [0.1, 1, 10, 100],  
#               'gamma': [1, 0.1, 0.01, 0.001], 
#               'kernel': ['rbf']}  
  
# grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
  
# # fitting the model for grid search 
# grid.fit(x_train, y_train) 
# # print best parameter after tuning 
# print(grid.best_params_) 
  
# # print how our model looks after hyper-parameter tuning 
# print(grid.best_estimator_) 
# grid_predictions = grid.predict(x_test) 
  
# # print classification report 
# print(classification_report(y_test, grid_predictions)) 

#%% # Evaluating the ANN
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score
# from keras.models import Sequential # initialize neural network library
# from keras.layers import Dense # build our layers library
# def build_classifier():
#     classifier = Sequential() # initialize neural network
#     classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
#     classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
#     classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#     classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#     return classifier
# classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
# accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)
# mean = accuracies.mean()
# variance = accuracies.std()
# print("Accuracy mean: "+ str(mean))
# print("Accuracy variance: "+ str(variance))

import sklearn
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from matplotlib import pyplot
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
import scipy
from sklearn import tree
from sklearn import model_selection

print("loading the dataset")
data = datasets.fetch_openml(data_id=344)
data.data.info()
print("We are performing one hot encoding")
ohe = OneHotEncoder(sparse=False)
ct = ColumnTransformer([("encoder", OneHotEncoder(sparse=False), [2,6,7])], remainder="passthrough")
new_data = ct.fit_transform(data.data)
ct.get_feature_names()
print("type of the new data")
type(new_data)
print("convert the type from numpy to dataframe")
lis_new_data = pd.DataFrame(new_data, columns = ct.get_feature_names(), index = data.data.index)
lis_new_data.info()

print("----------Decission Tree--------------")

dt = tree.DecisionTreeRegressor(random_state=0)
parameters = [{"min_samples_leaf":[2,4,6,8,10]}]
tuned_dt = model_selection.GridSearchCV(dt, parameters, scoring="neg_root_mean_squared_error", cv=10)
train_sizes_dt, train_scores_dt, test_scores_dt, fit_times_dt, score_times_dt = learning_curve(tuned_dt, lis_new_data, data.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
print("Computational time for training of last point",fit_times_dt[4].mean())
print("Computational time for test of last point",score_times_dt[4].mean())
rsme_dt = 0-test_scores_dt
print("RMSE for decision tree",rsme_dt)
dt_rsme = rsme_dt[4].mean()
print("mean RMSE for last point in Decision tree to choose best model",dt_rsme)

pyplot.plot(train_sizes_dt,  rsme_dt.mean(axis=1))
pyplot.xlabel("Number of training examples_dt")
pyplot.ylabel("RMSE_dt")
pyplot.show()

print("----------K nearest Neighbours--------------")

knn = KNeighborsRegressor(n_neighbors=2)
train_sizes_knn, train_scores_knn, test_scores_knn, fit_times_knn, score_times_knn = learning_curve(knn, lis_new_data, data.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
print("Computational time for training of last point",fit_times_knn[4].mean())
print("Computational time for test of last point",score_times_knn[4].mean())
rsme_knn = 0-test_scores_knn
print("RMSE for decision tree",rsme_knn)
knn_rsme = rsme_knn[4].mean()
print("mean RMSE for last point in K nearest Neighbours to choose best model",knn_rsme)

pyplot.plot(train_sizes_knn,  rsme_knn.mean(axis=1))
pyplot.xlabel("Number of training examples_knn")
pyplot.ylabel("RMSE_knn")
pyplot.show()

print("----------Linear Regression--------------")

lr = LinearRegression()
train_sizes_lr, train_scores_lr, test_scores_lr, fit_times_lr, score_times_lr = learning_curve(lr, lis_new_data, data.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
print("Computational time for training of last point",fit_times_lr[4].mean())
print("Computational time for test of last point",score_times_lr[4].mean())
rsme_lr = 0-test_scores_lr
print("RMSE for decision tree",rsme_lr)
lr_rsme = rsme_lr[4].mean()
print("mean RMSE for last point in Linear Regression to choose best model",lr_rsme)

pyplot.plot(train_sizes_lr,  rsme_lr.mean(axis=1))
pyplot.xlabel("Number of training examples_lr")
pyplot.ylabel("RMSE_lr")
pyplot.show()

print("----------Support Vector Machine Regressor--------------")

svm = SVR()
train_sizes_svm, train_scores_svm, test_scores_svm, fit_times_svm, score_times_svm = learning_curve(svm, lis_new_data, data.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
print("Computational time for training of last point",fit_times_svm[4].mean())
print("Computational time for test of last point",score_times_svm[4].mean())
rsme_svm = 0-test_scores_svm
print("RMSE for decision tree",rsme_svm)
svm_rsme = rsme_svm[4].mean()
print("mean RMSE for last point in Support Vector Machine Regressor- to choose best model",svm_rsme)

pyplot.plot(train_sizes_svm,  rsme_svm.mean(axis=1))
pyplot.xlabel("Number of training examples_svm")
pyplot.ylabel("RMSE_svm")
pyplot.show()


print("----------Bagged  Regressor--------------")


br = BaggingRegressor()
train_sizes_br, train_scores_br, test_scores_br, fit_times_br, score_times_br = learning_curve(br, lis_new_data, data.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
print("Computational time for training of last point",fit_times_br[4].mean())
print("Computational time for test of last point",score_times_br[4].mean())
rsme_br = 0-test_scores_br
print("RMSE for decision tree",rsme_br)
br_rsme = rsme_br[4].mean()
print("mean RMSE for last point in Bagged  Regressor to choose best model",br_rsme)

pyplot.plot(train_sizes_br,  rsme_br.mean(axis=1))
pyplot.xlabel("Number of training examples_br")
pyplot.ylabel("RMSE_br")
pyplot.show()

print("----------Dummy Regressor--------------")


dr = DummyRegressor()
train_sizes_dr, train_scores_dr, test_scores_dr, fit_times_dr, score_times_dr = learning_curve(dr, lis_new_data, data.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
print("Computational time for training of last point",fit_times_dr[4].mean())
print("Computational time for test of last point",score_times_dr[4].mean())
rsme_dr = 0-test_scores_dr
print("RMSE for decision tree",rsme_dr)
dr_rsme = rsme_dr[4].mean()
print("mean RMSE for last point in Dummy Regressor to choose best model",dr_rsme)
pyplot.plot(train_sizes_dr,  rsme_dr.mean(axis=1))
pyplot.xlabel("Number of training examples_dr")
pyplot.ylabel("RMSE_dr")
pyplot.show()


print("-------------Comparision of all models learning curves-------------")

pyplot.plot(train_sizes_dt,  rsme_dt.mean(axis=1),linestyle = '--',label = "Decision Tree")
pyplot.plot(train_sizes_knn,  rsme_knn.mean(axis=1),linestyle = '-',label = "KNN")
pyplot.plot(train_sizes_lr,  rsme_lr.mean(axis=1),linestyle = 'solid',label = "Linear regression")
pyplot.plot(train_sizes_svm,  rsme_svm.mean(axis=1),linestyle = 'dotted',label = "Support vector machine")
pyplot.plot(train_sizes_br,  rsme_br.mean(axis=1),linestyle = 'dashdot',label = "Bagged regression")
pyplot.plot(train_sizes_dr,  rsme_dr.mean(axis=1),linestyle = ':',label = "Dummy regreesor")
pyplot.xlabel("Number of training examples")
pyplot.ylabel("RMSE")
pyplot.legend()
pyplot.show()

print("-------------Statistical Significance--------------")
dt_ss=scipy.stats.ttest_rel(rsme_br.mean(axis=1),rsme_dr.mean(axis=1) )
knn_ss=scipy.stats.ttest_rel(rsme_br.mean(axis=1),rsme_knn.mean(axis=1) )
lr_ss=scipy.stats.ttest_rel(rsme_br.mean(axis=1),rsme_lr.mean(axis=1) )
svm_ss=scipy.stats.ttest_rel(rsme_br.mean(axis=1),rsme_svm.mean(axis=1) )
dr_ss=scipy.stats.ttest_rel(rsme_br.mean(axis=1),rsme_dr.mean(axis=1) )
print("decision tree",dt_ss)
print("knn",knn_ss)
print("lr",lr_ss)
print("svm",svm_ss)
print("dr",dr_ss)


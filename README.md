# Regression-ML-project


***ANUHYA KALVAKALA****
compare multiple machine learning methods on a regression task using learning curves and computational times using SKLEARN

**Dataset Description:**

•	This data set is an artificial dataset developed using the methods as below.
•	X1: uniformly distributed over [-5,5] 
•	X2: uniformly distributed over [-15, -10] 
•	X3: IF (X1 > 0) THEN X3 = green ELSE X3 = red with probability 0.4 
•	 X4=brown with prob. 0.6 X4: IF (X3=green) THEN X4=X1+2X2 ELSE X4=X1/2 with prob. 0.3, and X4=X2/2 with prob. 0.7 
•	X5: uniformly distributed over [-1,1] 
•	X6: X6=X4*[epsilon], where [epsilon] is uniformly distribute over [0,5] 
•	X7: X7=yes with prob. 0.3 and X7=no with prob. 0.7 
•	X8: IF (X5 < 0.5) THEN X8 = normal ELSE X8 = large 
•	X9: uniformly distributed over [100,500] 
•	X10: uniformly distributed integer over the interval [1000,1200]
•	The target variable Y using the rules: IF (X2 > 2) THEN Y = 35 - 0.5 X4 ELSE IF (-2 <= X4 <= 2) THEN Y = 10 - 2 X1 ELSE IF (X7 = yes) THEN Y = 3 -X1/X4 ELSE IF (X8 = normal) THEN Y = X6 + X1 ELSE Y = X1/2
•	The data had the numeric and nominal values at the x3, x6, x7 which are converted into numeric values using the one hot encoder value.
•	The converted values are fit and the type of the data we observe after fitting is NumPy which is converted to data frame data which is easy to calculate further.
Methods:
•	We are working with 6 methods as below 
Decision Tree Regressor:
•		In this we used sklearn and imported tree, model_selection and DecisionTreeRegressor.
•		In this we have tunned our parameters using min leaf sample 2,4,6,8,10 and tuned the tree using the scoring as "neg_root_mean_squared_error" and cross validation value of 10.
•		Using the tunned tree, we have made a learning curve using the train sizes as 0.2,0.4,0.6,0.8,1 with cross validation 10.
•	K Nearest Neighbors Regressor:
•	In this we used sklearn and imported KNeighborsRegressor
•	In this we have tunned our parameters using n_neighbors=2 and tuned the tree using the scoring as "neg_root_mean_squared_error" and cross validation value of 10.
•	Using the tunned tree, we have made a learning curve using the train sizes as 0.2,0.4,0.6,0.8,1 with cross validation 10

•	Linear Regression:
•		In this we used sklearn and imported Linear Regression
•		In this we have tuned the tree using the scoring as "neg_root_mean_squared_error" and cross validation value of 10.
•		Using the train sizes as 0.2,0.4,0.6,0.8,1 with cross validation 10 learning curve is calculated.
•	Support Vector Machine:
•		In this we used sklearn.svm and imported SVR
•		In this we have tuned the tree using the scoring as "neg_root_mean_squared_error" and cross validation value of 10.
•	Using the train sizes as 0.2,0.4,0.6,0.8,1 with cross validation 10 learning curve is calculated.
•	Bagging Regressor:
•		In this we used sklearn and imported BaggingRegressor
•	In this we have tuned the tree using the scoring as "neg_root_mean_squared_error" and cross validation value of 10.
•	Using the train sizes as 0.2,0.4,0.6,0.8,1 with cross validation 10 learning curve is calculated.
•	Dummy Regressor:
•		In this we used sklearn and imported DummyRegressor
•		In this we have tuned the tree using the scoring as "neg_root_mean_squared_error" and cross validation value of 10.
•		Using the train sizes as 0.2,0.4,0.6,0.8,1 with cross validation 10 learning curve is calculated.

**Graphs:**
**Decision Tree Regressor:
 <img width="226" alt="image" src="https://user-images.githubusercontent.com/96926526/170374753-1a253492-041e-45b4-998c-a72baf3d4487.png">

**K nearest Neighbors:**
 <img width="226" alt="image" src="https://user-images.githubusercontent.com/96926526/170374780-14460ef8-bece-4ee0-817a-bede65a95cf1.png">

**Linear regression:**
 
  <img width="222" alt="image" src="https://user-images.githubusercontent.com/96926526/170374829-a58a11a5-94c9-48a8-9e4d-f959e063345e.png">

**Support Vector Machine:**
     
   <img width="248" alt="image" src="https://user-images.githubusercontent.com/96926526/170374857-faa0c76b-7f91-42cd-873c-3b51f3d72be8.png">

**Bagged Decision tree:**
  
   <img width="251" alt="image" src="https://user-images.githubusercontent.com/96926526/170374908-03910930-7079-4e25-801f-be68b8fb5176.png">

    
**Dummy Regressor:**
     
   <img width="254" alt="image" src="https://user-images.githubusercontent.com/96926526/170374926-a3c0e0ad-d552-47b6-ae43-a3ea4d436f38.png">

**All models graph comparison:**

   <img width="337" alt="image" src="https://user-images.githubusercontent.com/96926526/170374963-d875d935-5f91-4e33-8b92-c16dfb9f985d.png">

                                                         
Here while comparing all the learning curves we can see a very close relationship between the decision tree and bagged Regressor but bagged regressor which works as an ensemble performs better than the decision tree.

Comparing bagged and decision tree regressors we can observe a banana shape in their learning curves from 15000 to 32000.
      
<img width="243" alt="image" src="https://user-images.githubusercontent.com/96926526/170375272-5b73f3a5-1136-485d-a774-f512df60dd0d.png">


**Tables:**
**Comparing all models with the best model**

![image](https://user-images.githubusercontent.com/96926526/170375228-c94f451c-f746-44c5-aa02-487dd0fae3b5.png)


**Comparing computational training and test times of all models**

![image](https://user-images.githubusercontent.com/96926526/170375067-0304dcf3-9301-47b1-ab10-c73c698db968.png)



**Discussion:**

•	Comparing all above models, we can compare that the bagged regression model performs better than other models with an RMSE value of 0.0891.
•	Comparing the graphs, we can say that decision tree and bagging regressor are very close but the bagging regressor model performs better than the remaining models where SVM performs bad compared to dummy regressor
•	Comparing the best model with all other models to get the statistical significance we imported SciPy and used stats. ttest_rel () we found all models are statistically significantly different as compared with p<0.01
•	Comparing the models computational time highest training and testing time was taken by SVM and least training time was by k nearest, and testing was by decision tree


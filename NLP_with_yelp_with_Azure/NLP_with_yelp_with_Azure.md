* This project directly reterives data from Yelp website and stores it in the Azure SQL database 
* Runs multiple ML classification models 
* Uses Grid search CV to hyperparameter tuning 
* accuracy of the model is stored again in the sql database
* Azure Key Vault & SQL database from Azure is used

Dataset:
------------------------------------
text                     | Stars
------------------------------------
restaurant was good,.... |    5
Not so good              |    2
amazing food             |    4
...                      |    4
...                      |    5
...                      |    1
_____________________________________



Feature creation:
* Count Vectorizer
* TF-IDF 

Models:
* SGD classifier
* Linear Regression
* MultinomialNB 
* Decision Tree Classifier
* Support Vector Machine

References:
https://scikit-learn.org/stable/modules/classes.html?highlight=svm#module-sklearn.svm
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decisiontreeclassifier#sklearn.tree.DecisionTreeClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html?highlight=multinomial#sklearn.naive_bayes.MultinomialNB
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html?highlight=sgdclassifier#sklearn.linear_model.SGDClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html?highlight=linear%20regression#sklearn.linear_model.LinearRegression



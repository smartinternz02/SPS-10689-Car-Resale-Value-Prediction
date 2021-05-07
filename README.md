 For Project Demonstration video click on the link given :-  
OVERVIEW:

  The prices of the new manufactured cars in the automobile industry by the manufacturer with some extra cost which are incurred by the Government in the form of taxes. 
  So, customers buying a new car can be assured of the money which they are investing to be worthy. But due to the increased price of new cars and the problems of customers to buy new cars due to improper financial status, used car sales are on increase in demand.
Due to the high raise in demand for the used cars, there is a need for a car resale price prediction system to effectively determine the quality and worth of the car using different features, which are very much required while assessing the price of resale car.

PURPOSE:

Client will be able to efficiently determine the price that the chosen resale vehicle must cost. So, that people can invest money according to their preferences.
In this project, we suggest a solution using Machine Learning algorithms implemented in JAVA. 

EXISTING PROBLEM:

Existing models use Linear Regression algorithm, Decision Tree algorithms as the basic models. Problem with these models is that, it takes a huge amount of time for training the model, and if any attributes are categorical, proper encoding techniques must be used, and yet due to high correlated data, this technique produces mid-level percentage of accuracy (70 - 80%), and if over fitting took place if we tried to remove the duplicate values. 

PROPOSED SOLUTION:

Proposed model is built on Random Forest regressor algorithm, instead of using conventional Linear Regression, and Decision Trees models. This model takes less amount of training time, and the correlation coefficient is high, and low error metrics, thus making suitable for modelling.
  
EXPERIMENTAL INVESTIGATIONS

There are separate datasets, one for the training, and other for testing the built model. 

TRAINING DATASET:

Training data consist of 12 independent variables and 1 dependent variable. Many attributes consisted of much missing data. The following table depicts the details of train data and test data.

depicts the details of train data and test data.

DATA CLEANING:

As depicted in the train dataset properties table, there are many missing values in the dataset. These missing will affect the accuracy in building the model. So, there is a need to remove the missing data, or to fill the missing data. “RemoveMissingValue” filer has been used in-order to replace missing value with median. And in for categorical variables such as Seats, etc. Missing values were replaced by mode.

DATA VISUALIZATION:

Visualizing the data is the most crucial part in analyzing the model. We use WEKA explorer to visualize the data in the form of Histogram plots. Below figures shows the modified data after using the “RemoveMissingValue” filter and removing unwanted attributes such “name”,”Location”,”New_Price”.
  


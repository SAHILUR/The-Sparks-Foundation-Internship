# By Sahilur Rahaman


#Data Science and Business Analytics Internship  of The Sparks Foundation

#Task -- 1
##Prediction using Supervised ML
 
#Data Source: https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv

#problem statement --What will be predicted score if a student studies for 9.25 hrs/ day?
#Using Linear Regression to Predict the Score of a Student



## Source code


##import all the Required libaries
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt



#read data (csv file)
df = pd.read_csv(r'F:\WORKPROJ\Internship\Student.csv') 
#print(df.head())



##apply linear regression model on dataset
x=df[['Hours']]
y=df[['Scores']]
reg=linear_model.LinearRegression() 



##fit the best fit line
reg.fit(x,y)   


#name the x and Y axis
plt.xlabel('Hours')    
plt.ylabel('Scores')


#Visualize Data By scatter and Regression Line 
plt.scatter(x,y)       
plt.plot(x,reg.predict(x))



##predict the value
print('the predict score is:',reg.predict([[9.25]]))    



##show the graph
plt.show()    















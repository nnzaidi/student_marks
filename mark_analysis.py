import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('student_marks.csv')
# print(data.head(10))
# print(data.info())

# Check for null values
# print(data.isnull().sum())

# Check number of values for column - number_courses
# print(data['number_courses'].value_counts())        # Min:3; Max:8

# Scatter plot number_courses vs. Marks
# figure = px.scatter(data_frame=data, 
#                     x='number_courses',
#                     y='Marks',
#                     size='time_study',
#                     title='Number of Courses and Marks Scored')
# figure.show()
# figure.write_image("Number of Courses and Marks Scored.png")

# Relationship between time_study and Marks
# figure = px.scatter(data_frame=data,
#                     x='time_study',
#                     y='Marks',
#                     size='number_courses',
#                     title='Time Spent and Marks Scored')
# figure.show()
# figure.write_image("Time Spent and Marks Scored.png")

# Observe correlation between the marks scored and the other two columns
# correlation = data.corr()
# print(correlation['Marks'].sort_values(ascending=False))

# Student Marks Prediction Model
# Splitting the data into training and test sets
x = np.array(data[['time_study','number_courses']])
y = np.array(data['Marks'])
xtrain, xtest, ytrain, ytest = train_test_split(x,
                                                y,
                                                test_size=0.2,
                                                random_state=42)
# Train and test the ML algorithm
model = LinearRegression()
model.fit(xtrain,ytrain)
model.score(xtest,ytest)
# print(model.score(xtest,ytest)) # Shows accuracy of 94%

# Test with user input
# features = np.array([['time_study','number_courses']])
features = np.array([[4.508,5]])
model.predict(features)
print(model.predict(features))
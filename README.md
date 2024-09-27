# ML-Python Practice 2: Student Marks (Linear Regression)

The project is referred from below link for study purpose. Credit to the owner:
https://thecleverprogrammer.com/2022/04/26/student-marks-prediction-with-machine-learning/

Pre-analysis activities include:
Clean and prepare the datatset by handling missing values and removing outliers.

Identify the relationship between each column:
1. number_courses vs. Marks: The number of courses does not affect the marks they get.
2. time_study vs. Marks: There is a linear relationship between the two. THis means that the more time they spent for studies, the better they can score.

Correlation check for all columns:
Marks             1.000000
time_study        0.942254
number_courses    0.417335
*** The correlation data shows that time_study correlates heavily to Marks.

Students Marks Prediction Model
1. Splitting the training and test sets
2. Train the model using linear regression algorithm
3. The test set scores 94% accuracy, implies that the Prediction Model is successful
4. Test the model by using input from user:
 __________________________________
|                                  |
|features = np.array([[4.508, 3]]) |
|model.predict(features)           |
|__________________________________|

The model predicted that the Marks will be 22.3

Conclusion:
After two Linear Regression practices, I am pretty much clear on how and when to use this model for data analysis. I am able to understand the relationship between each variable in the dataset and how to make analysis based on that.

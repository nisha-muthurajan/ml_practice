import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error

data = {
    "Study Hours": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],

    "Attendance": [65, 70, 75, 80, 82, 85, 88, 90, 93, 95, 97, 98],

    "Assignments": [4, 5, 6, 7, 8, 8, 9, 10, 10, 10, 10, 10],

    "Previous Score": [50, 55, 60, 65, 68, 72, 75, 80, 85, 88, 92, 95],

    "Final Score": [52, 58, 64, 70, 74, 79, 84, 89, 94, 97, 99, 100]
}

df=pd.DataFrame(data)

x=df[["Study Hours","Attendance","Assignments","Previous Score"]]
y=df["Final Score"]

x_train,x_test,y_train,y_test=train_test_split(x,y)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("Actual values:",x_test)
print("Predicted values:",y_pred)

print("R2 score:",r2_score(y_test,y_pred))
print("MAE:",mean_absolute_error(y_test,y_pred))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = {
    "Performance Score": [
        45, 48, 50, 52, 55, 58,
        60, 62, 65, 68, 70, 72,
        75, 78, 80, 82, 85, 90
    ],

    "Promoted": [
        0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1
    ]
}

df=pd.DataFrame(data)
x=df[["Performance Score"]]
y=df["Promoted"]

x_train,x_test,y_train,y_test=train_test_split(x,y)

model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print("Actual Values:",y_test.values)
print("Predicted Values:",y_pred)

print("Accuracy Score:",accuracy_score(y_test,y_pred))

print("Probability:",model.predict_proba(x_test))
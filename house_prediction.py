import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

area=[500,700,900,1100,1300,1500,1700,1900,2100,2300]
prices=[25,32,40,48,55,63,70,78,85,93]

data=pd.DataFrame({"Area":area,"Prices":prices})

x=data[["Area"]]
y=data["Prices"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

print("predicted prices:",y_predict)
print("predict for area 2500:",model.predict([[2500]]))
print("R2 score:",r2_score(y_test,y_predict))

plt.scatter(x,y,color="blue")
plt.plot(x,model.predict(x),color="red")
plt.xlabel("Area in sq ft")
plt.ylabel("Price in $1000")
plt.title("Area vs Price")
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

data = {
    "Temperature": [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38],
    
    "Sales": [20, 25, 32, 42, 55, 72, 92, 115, 142, 172, 205, 240, 278, 318, 360]
}

df=pd.DataFrame(data)

x=df[["Temperature"]]
y=df["Sales"]

poly=PolynomialFeatures(degree=1)
x_poly=poly.fit_transform(x)

model=LinearRegression()
model.fit(x_poly,y)

y_pred=model.predict(x_poly)

print("Actual Values:",y.values)
print("Predicted Values:",y_pred)

print("R2 score:",r2_score(y,y_pred))
print("MAE:",mean_absolute_error(y,y_pred))

x_plot=np.linspace(
    x["Temperature"].min(),
    x["Temperature"].max(),
    100
).reshape(-1,1)

x_plot_poly=poly.transform(x_plot)

y_plot=model.predict(x_plot_poly)

plt.scatter(x,y)
plt.plot(x_plot,y_plot)
plt.xlabel("TEMPERATURE")
plt.ylabel("SALES")
plt.title("POLYNOMIAL REGRESSION")
plt.show()
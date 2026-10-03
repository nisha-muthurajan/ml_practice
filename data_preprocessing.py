import pandas as pd
import numpy as np

data = {
    "Age": [24, 29, np.nan, 27, 29, 150, 35, 24],
    "Department": [
        "IT", "HR", "IT", "Sales",
        "HR", "IT", "sales", "IT"
    ],
    "Salary": [
        30000, 40000, 45000, np.nan,
        40000, 50000, 35000, 30000
    ],
    "Performance Score": [
        65, 80, 75, 70, 80, 95, np.nan, 65
    ],
    "Promoted": [0, 1, 1, 0, 1, 1, 0, 0]
}

df = pd.DataFrame(data)

print(df)

print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.info)
print(df.describe)
 
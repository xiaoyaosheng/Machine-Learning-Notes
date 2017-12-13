import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
data_from = pd.read_fwf("brain_body.txt")

x_values = data_from[['Brain']]
y_value = data_from[['Body']]

body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_value)

plt.scatter(x_values, y_value)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()

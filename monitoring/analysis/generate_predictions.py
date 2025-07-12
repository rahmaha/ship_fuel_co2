import pandas as pd
import pickle
import numpy as np

from sklearn.multioutput import MultiOutputRegressor

#load the current data 
cur_data = pd.read_csv('data/current.csv')

#drop target
X_current = cur_data.drop(columns=['fuel_consumption', 'CO2_emissions'])

#load model and dv
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/dv.pkl', 'rb') as f:
    dv = pickle.load(f)

X = dv.transform(X_current.to_dict(orient='records'))

# predict
y_pred = model.predict(X)
y_pred = np.expm1(y_pred)

# add predictions coloumns to current data
cur_data['fuel_consumption_pred'] = y_pred[:, 0]
cur_data['CO2_emissions_pred'] = y_pred[:, 1]

# save it
cur_data.to_csv('data/current_with_preds.csv', index=False)
print('Saved to data/current_with_preds.csv')
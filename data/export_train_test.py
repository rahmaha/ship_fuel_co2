import pandas as pd
# from sklearn.model_selection import train_test_split

df = pd.read_csv('data/ship_fuel_efficiency.csv')

#drop ship_id
df = df.drop(columns='ship_id')

#split
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = df.iloc[:865]
test_df = df.iloc[865:]
# save
train_df.to_csv('data/reference.csv', index=False)
test_df.to_csv('data/current.csv', index=False)

print(len(train_df))
print(len(test_df))
print('saved train and test dataframe to reference and current')
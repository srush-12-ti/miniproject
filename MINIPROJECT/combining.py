import pandas as pd 
df1=pd.read_csv('preprocessed2015.csv')
df2=pd.read_csv("preprocessed2013.csv")
df_combined = pd.concat([df1, df2], ignore_index=True)
df_combined.to_csv("finaldataset.csv",sep=',', index=False)

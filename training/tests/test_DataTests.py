import pandas as pd
df=pd.read_csv('data/creditcard2023.csv')
df_reduccion = pd.DataFrame(index=df.columns[:-1])
df_reduccion["Visual"]=[0, 1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,0, 1,0, 0, 0, 0, 0, 1,0, 0]
import pandas as pd
import numpy as np
import chardet

with open('Pokemon.csv', 'rb') as f:
    result = chardet.detect(f.read())

df = pd.read_csv('Pokemon.csv', encoding=result['encoding'])
df = df.iloc[:-4]

#print(df['Type 2'].drop_duplicates().to_numpy())

valid_tp2 = ['Poison','Flying','Dragon' ,'Ground' ,'Fairy' ,'Grass', 'Fighting' ,
              'Psychic', 'Steel' ,'Ice' , 'Rock', 'Dark', 'Water' ,'Electric','Fire' 
              'Ghost' ,'Bug' ,'Normal']

df = df[df['Type 2'].isin(valid_tp2) | df['Type 2'].isnull()]

df.drop_duplicates(inplace=True)

df['Attack'] = pd.to_numeric(df['Attack'])
df = df[df['Attack'] < 500]

Gen = ['0','1','2','3','4','5','6']
Lgd = ['TRUE', 'FALSE']

df['Legendary'] = df['Legendary'].astype(str)
df['Generation'] = df['Generation'].astype(str)

mask = df['Generation'].isin(Lgd) & df['Legendary'].isin(Gen)
df.loc[mask, ['Generation', 'Legendary']] = df.loc[mask, ['Legendary', 'Generation']].values

df = df[df['Legendary'].isin(Lgd)]
df = df[df['Generation'].isin(Gen)]
df.to_csv('filtered.csv')
print(df)
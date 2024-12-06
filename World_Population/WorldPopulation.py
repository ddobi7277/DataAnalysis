import pandas as pd
import seaborn as sns
import matplotlib as plt

data= pd.read_csv("world_population.csv")
#info
print(data.info())

#cleaning an filtrering
print(data.corr())
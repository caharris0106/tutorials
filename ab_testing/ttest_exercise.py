from scipy import stats
import numpy as np
import pandas as pd

df = pd.read_csv("advertisement_clicks.csv")
A = df[df['advertisement_id'] == "A"]
B = df[df['advertisement_id'] == 'B']
a = np.array(A['action'])
b = np.array(B['action'])
N = b.shape[0]



var_a = a.var(ddof=1)
var_b = b.var(ddof=1)

s = np.sqrt((var_a + var_b) / 2)
t = ( a.mean() - b.mean() ) / ( s * np.sqrt( 2/ N ) )
print('t ',t)
df = 2*N - 2
p = 2*stats.t.cdf(t, df=df)
print('p ', p)
t2, p2 = stats.ttest_ind(a, b)
print("t2:\t", t2, "p2:\t", p2)

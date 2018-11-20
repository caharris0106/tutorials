import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, chi2_contingency
import pandas as pd

T = np.zeros((2,2))

df = pd.read_csv('advertisement_clicks.csv')

print(df.groupby('advertisement_id').sum())

ad_a = df[df['advertisement_id'] == "A"]
ad_b = df[df['advertisement_id'] == "B"]

a_clicks = ad_a.groupby('advertisement_id').sum()['action']['A']

T[0,0] = a_clicks
T[0,1] = len(ad_a) - a_clicks

b_clicks = ad_b.groupby('advertisement_id').sum()['action']['B']

T[1,0] = b_clicks
T[1,1] = len(ad_b) - b_clicks

def get_p_value(T):
    det = T[0,0] * T[1,1] - T[0,1]*T[1,0]
    c2 = float(det) / T[0].sum() * float(det) / T[1].sum() * T.sum() / T[:,0].sum() / T[:, 1].sum()
    # test = (float(det) **2)*T.sum()/ (T[1].sum() * T[0].sum() * T[:,0].sum() * T[:, 1].sum())
    p = 1 - chi2.cdf(x=c2, df=1)
    return p

if __name__ == "__main__":
    print(get_p_value(T))
    # Since the p_value is < 0.05 percent, the differece is significant

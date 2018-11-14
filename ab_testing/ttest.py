import numpy as np
from scipy import stats

N = 100
a = np.random.randn(N) + 2
b = np.random.randn(N)

# We want to use N-1, unbiased estimate
var_a = a.var(ddof=1)
var_b = b.var(ddof=1)

# Pooled standard Deviation
s = np.sqrt( (var_a + var_b)/2 )

# Test Statistic
t = (a.mean() - b.mean())/(s*np.sqrt(2/N))

# Degrees of freedom
# Two sample test
df = 2*N - 2

#p value
p = 1-stats.t.cdf(t, df=df)
print("t:\t", t, "p:\t", 2*p)

# compare with the stats ttest
t2, p2 = stats.ttest_ind(a, b)
print("t2:\t", t2, "p2:\t", p2)

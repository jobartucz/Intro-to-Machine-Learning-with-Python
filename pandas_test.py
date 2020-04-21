import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

if False:
    ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
    ts = ts.cumsum()
    ts.plot()
else:
    df = pd.DataFrame(np.random.randn(1000, 4), columns=['A','B','C','D'])
    pd.plotting.scatter_matrix(df, alpha=0.2)

plt.show()

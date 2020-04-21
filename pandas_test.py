import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(1000, 4), columns=['A','B','C','D'])
print(df)
pd.plotting.scatter_matrix(df, alpha=0.2)
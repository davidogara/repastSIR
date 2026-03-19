import subprocess
import numpy as np
import pandas as pd
import json
import os
from tempfile import NamedTemporaryFile

df = pd.DataFrame({'beta':np.random.default_rng(123).uniform(size=3)})
df['seed'] = range(0,len(df))

pars = df.to_dict(orient='records')

 




if __name__ == "__main__":
    query = f'python run_batch.py --pars test.json'
    res = subprocess.run(args=query,shell=True,capture_output=True)
    foo=1
    

"""Utility to generate a random walk dataset for testing."""
import pandas as pd
import random

close = list()
open = list()
high = list()
low = list()
volume = list()

close.append(5000)
open.append(4999)
high.append(5021)
low.append(4980)
volume.append(random.lognormvariate(3.0, 2))

for i in range(1, 1000):
    open.append((open[i - 1] * random.normalvariate(1.0, 0.001)))
    high.append(max(close[i - 1], open[i - 1]) * (1 + random.random()))
    low.append(min(close[i - 1], open[i - 1]) * (1 - random.random()))
    close.append(close[i - 1] * random.normalvariate(1.0, 0.001))
    volume.append(random.lognormvariate(3.0, 2))

df = pd.DataFrame({
    'time': range(1, 1001),
    'open': open,
    'high': high,
    'low': low,
    'close': close,
    'volume': volume
})

df.loc[
    :,
    [
        'time',
        'open',
        'high',
        'low',
        'close',
        'volume'
    ]
].to_csv(
    './data/random-walk-1000.csv',
    index=False
)

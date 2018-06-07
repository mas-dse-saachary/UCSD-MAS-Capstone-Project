import json
from StringIO import StringIO
import pandas as pd

def parse_columns(listings, cols):
    chars = "%$"
    for i in cols:
        listings[i] = listings[i].astype(str).map(lambda x: x.rstrip(chars))
        listings[i] = listings[i].astype(str).map(lambda x: x.lstrip(chars))
        listings[i] = listings[i].apply(pd.to_numeric, errors='coerce')
        listings[i].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
    return listings 
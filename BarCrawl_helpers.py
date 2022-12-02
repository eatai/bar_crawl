import numpy as np
import pandas as pd

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

from os.path import join, split

# Cartesian Accelerations and their ICA signals
xyz = ['x', 'y', 'z']
C = ['C1', 'C2']

# Spherical Accelerations and their ICA signals
rtp = ['rho', 'theta', 'phi']
S = ['S1', 'S2']

def cart_to_spherical(xyz):
    # INPUT
    # xyz - nx3 array of cartesian coordinates
    #
    # OUTPUT
    # rtp - nx3 array of spherical coordinates (radius, azimuth, elevation)
    eps = np.finfo(np.float32).eps
    
    x = xyz[:,0] + eps
    y = xyz[:,1] + eps
    z = xyz[:,2] + eps
    rho = np.linalg.norm(xyz, axis=1)
    theta = np.unwrap(np.arctan2(y,x))
    phi = np.unwrap(np.arccos(z/rho))
    
    return np.nan_to_num(np.stack([rho, theta, phi], axis=1), copy = False)


def null_count(x):
    for col in x.columns:
        print(f"{col}:\t{x[col].isna().sum()}")
        
# Quartiles for agg
def q25(x):
    return x.quantile(0.25)

def q50(x):
    return x.quantile(0.50)

def q75(x):
    return x.quantile(0.75)


# Frequency-domain functions for agg
def dc(x):
    dc = x[0]
    return dc

def power(x):
    power = np.linalg.norm(x, axis=0)
    return power

def peak(x):
    peak = x[1:].max(axis=0)
    return peak

def peak_freq(x):
    peak_freq = (np.argmax(x[1:],axis=0)+1)*0.1
    return peak_freq


# File compression and conversion

# The function reduce_mem_usage credited to:
# https://gist.github.com/fujiyuu75/748bc168c9ca8a49f86e144a08849893

def reduce_mem_usage(df, use_float16=False, verbose = True):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
    """
    
    start_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
        print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def csv2feather(csv, datafolder = '.'):
    feather_filename = split(csv)[1][:-3] + 'ftr'

    df = pd.read_csv(csv)
    df = reduce_mem_usage(df, verbose = False)
    
    save_file = join(datafolder, feather_filename)
    print(f'Saving: {save_file}')
    df.to_feather(save_file)
    return df



def rising_falling_edge(y):
    y_diff = np.diff(y, prepend=0)
    rise = np.where(y_diff>0)
    fall = np.where(y_diff<0)
    return np.hstack((rise, fall))



# Plotting
import matplotlib.pyplot as plt

TP_COLOR = '#AED581'
FP_COLOR = '#FFE082'
FN_COLOR = '#FF8A65'
TN_COLOR = '#DCEDC8'

def plot_confusion_time(t, y_true, y_pred, ax=plt.gca, ylims = [0,1], **kwaargs):

    # y_true in {0,0.5}, y_pred in {0, 1}
    # 0.5*y_true + y_pred in {0, 0.5, 1.5, 2}
    # 0: True Negative
    # 0.5: False Negative
    # 1.5: False Positive
    # 2: True Positive

    y_match = 0.5 * y_true + y_pred

    y_tp = (y_match==2)
    y_fp = (y_match==1.5)
    y_fn = (y_match==0.5)
    y_tn = (y_match==0)

    L1, = ax.fill_between(t, ylims[0], ylims[1], where = y_tp, color = TP_COLOR)
    L2, = ax.fill_between(t, ylims[0], ylims[1], where = y_fp, color = FP_COLOR)
    L3, = ax.fill_between(t, ylims[0], ylims[1], where = y_tn, color = TN_COLOR)
    L4, = ax.fill_between(t, ylims[0], ylims[1], where = y_fn, color = FN_COLOR)

    return L1, L2, L3, L4,


    

    



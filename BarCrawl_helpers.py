import numpy as np

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

    



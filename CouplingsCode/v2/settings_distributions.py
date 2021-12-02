import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as sstat
#from mpmath import mp
import math

from tensorflow_probability.substrates import numpy as tfp
tfd = tfp.distributions


n=1000
mean_obs = 43
sd_obs = 5 #recall that in tfp loc/scale convention is used, and scale in normal is the standard dev


y_obs = tfp.distributions.Normal(    
    mean_obs,
    sd_obs
    ).sample(n)



def lik(mu,n,y):
    med = 1/n*np.sum(y)
    yyy = []
    ssum = np.sum(yyy)
    for j in range(len(y)):
        yyy.append((y[j]-med)**2)
    res = math.e**(-1/50*(n*(mu-med)**2 + ssum))
    return res


mu0 = 33
def pi0(mu):
    ris2 = tfd.Normal(mu0,2).prob(mu)
    return ris2



##Target distribution: posterior proportional to lik*pi
def target(mu):
    return lik(mu,n,y_obs)*pi0(mu)
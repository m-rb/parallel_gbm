from numba import jit
import pandas as pd
import numpy as np

def normal_simulations(assets, number_of_simulations):
    """ Returns a dataframe with shape(#assets,#simulations)

    This function allows to generate normal distributed numbers such as N~(0,1)

    """
    df = pd.DataFrame(np.random.normal(0,1,(number_of_simulations,assets)),
                      columns=[i for i in range(1,assets+1)])
    return df

@jit(nopython=False, parallel=True)
def gbm(s0, sigma, r, d, time_step,w):
    """
    This function generates a geometric brownian motion.
    The usage of jit allows for parallel computation.
    """
    gbm = s0 * np.exp(r-d-(1/2*(sigma**2)) * time_step + sigma * np.sqrt(time_step) * w)
    return gbm


r = -0.005 #assuming 1y EUR Swap rate
d = [0.01,0.07,0.04,0.05,0.025] # random 1y dividend yield
sigma = [0.20,0.17,0.15,0.19,0.12] #random annual volatilies
time_step = 1 #1y

sims = normal_simulations(5,200000)

prices = [gbm(100, sigma[i-1],r,d[i-1],time_step,sims[i]) for i in sims.columns]

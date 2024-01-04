import numpy as np
import pandas as pd

import scipy.stats

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
 
import statsmodels.api as sm

import matplotlib.pyplot as plt

def rapid_linear_reg(x, y, summary=False, plot=False):
    ox = np.array(x).reshape(-1,1)
    oy = np.array(y).reshape(-1,1)

    not_nan = ~(np.isnan(ox)|np.isnan(oy))
    x, y = ox[not_nan].reshape(-1,1), oy[not_nan].reshape(-1,1)

    reg = LinearRegression().fit(x,y)
    X_with_constant = sm.add_constant(x)
    est = sm.OLS(y, X_with_constant)
    est2 = est.fit()
    if summary:
        print(est2.summary())
    print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

    pearsonr = scipy.stats.pearsonr(x.reshape(-1),y.reshape(-1))
    spearmanr = scipy.stats.spearmanr(x.reshape(-1),y.reshape(-1))
    
    print(pearsonr)
    print(spearmanr)
    py = np.full_like(ox, fill_value=np.nan)
    py[~np.isnan(ox)] = reg.predict(ox[~np.isnan(ox)].reshape(-1,1)).reshape(-1)

    if plot:
        plt.scatter(ox,oy)
        plt.plot(ox[not_nan], py[not_nan],c = 'red')
        plt.show()

    return py, pearsonr, spearmanr


def rapid_process_result(loss, r_df, path=True, plot=False):
    if path:
        loss = np.load(loss)
        r_df = pd.read_csv(r_df, index_col=0)
    print('epochs: {0}'.format(loss.shape[1]))

    r = r_df['r'].to_numpy().reshape(-1,1)
    p = r_df.to_numpy()[:,1:]
    ae = np.abs(p-r)

    df = pd.DataFrame(index=r_df.index, columns=['r','p','ae'], dtype=float)
    df['r'] = r_df['r']

    for i in range(loss.shape[0]):
        val_idx = list(range(ae.shape[0]))[i::loss.shape[0]]
        r_i = r[val_idx, :]
        p_i = p[val_idx, :]
        ae_i = np.abs(p_i-r_i)

        Min_loss_i = [np.where(loss[i,:] == np.min(loss[i,:]))[0][0]]
        Min_val_i = [np.where(np.mean(ae_i**2, axis=0) == np.min(np.mean(ae_i**2, axis=0)))[0][0]]
        
        df.loc[val_idx, 'p'] = p_i[:,Min_loss_i]
    
    final_p = df['p'].to_numpy().reshape(-1,1)
    final_ae = abs(final_p - r)

    print('Median Absolute Error:', np.median(final_ae))
    print('Mean Absolute Error:', np.mean(final_ae))

    Min_loss = np.argmin(np.mean(loss, axis=0))
    Min_mae = np.argmin(np.median(ae, axis=0))
    print('Min_loss:', Min_loss, np.mean(loss, axis=0)[Min_loss])

    if plot:
        plt.figure(dpi=100,figsize = (12,6))
        plt.plot(np.mean(loss, axis=0))
        plt.plot(np.mean(ae, axis=0))
        plt.plot(np.median(ae, axis=0))
        plt.scatter(Min_loss, np.mean(loss, axis=0)[Min_loss], c='r', zorder = 2)
        plt.scatter(np.argmin(np.mean(ae, axis=0)), np.min(np.mean(ae, axis=0)), c='r', zorder = 5)
        plt.scatter(Min_mae, np.median(ae, axis=0)[Min_mae], c='r', zorder = 2)

        plt.axhline(y=np.median(final_ae), linewidth=1,color='grey')
        plt.ylim(0, 10)
        plt.show()

    return r, final_p, final_ae
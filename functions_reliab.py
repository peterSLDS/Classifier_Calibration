def calc_prep(x, y, level, m, uncertainty, type):
    import numpy as np
    import pandas as pd
    from sklearn.isotonic import IsotonicRegression
    from scipy.stats import bernoulli
    from scipy.stats import norm

    # Sort inout data
    def combineXY(x, y):
        x = np.array(x)
        y = np.array(y)
        df = pd.DataFrame({'X': x, 'Y': y})
        df = df.sort_values('X', axis=0, ascending=True)
        return df

    df = combineXY(x=x, y=y)
    x = np.array(df.loc[:, 'X'])
    y = np.array(df.loc[:, 'Y'])

    # Run isotonic regression
    isoreg = IsotonicRegression()
    isoreg.fit(x, y)
    isofit = np.array(isoreg.predict(x))

    # Compute Values for Uncertainty regions
    # Resampling
    if uncertainty == "resampling":
        z = np.sort(np.unique(x))
        z_df = pd.DataFrame({'Z': z})
        x_df = pd.DataFrame({'X': x}).sort_values(by='X')
        for l in range(1, m + 1):
            y_resample = []
            iso_bs = IsotonicRegression()
            # Sample with Replacement
            x_resample = x_df.sample(n=len(x_df), replace=True)
            # Iterate over all units in the resampled data set
            for i in range(1, (len(x) + 1)):
                # Draw new label for current unit from resampled data
                if type == "consistency":
                    # Use initial score from classifier as parameter
                    bern = bernoulli(x_resample.iloc[i - 1, 0])
                    y_resample.append(bern.rvs(1)[0])
                if type == "confidence":
                    # Use isotonic fit as parameter
                    bern = bernoulli(isoreg.f_(x_resample.iloc[i - 1, 0]))
                    y_resample.append(bern.rvs(1)[0])
            x_resample = np.array(x_resample.loc[:, 'X'])
            y_resample = np.array(y_resample)
            # Fit PAVA on resampled data with new labels
            iso_bs.fit(x_resample, y_resample)
            # Obtain new CEP_hat at the unique scores of the resampled set
            x_uni = np.unique(x_resample)
            isofit_bs = np.array(iso_bs.predict(x_uni))
            # Linearly interpolate in between
            bs_inter = np.interp(x=z, xp=x_uni, fp=isofit_bs, left=0, right=1)
            # Append new CEP_hats to collection of bootstrap iterations
            bs_row = pd.DataFrame({'Z': z, 'fit_bs' + str(l): bs_inter})
            z_df = z_df.merge(bs_row, on='Z', how='left')
        z_df = z_df.T
        # Obtain quantiles for each unique forecast score z in the original data set over the m bootstrapped values
        z_df = z_df.rename(columns=z_df.iloc[0]).drop(z_df.index[0])
        z_df = z_df.quantile(q=[(level / 2), 1 - (level / 2)], axis=0, numeric_only=True, interpolation='nearest')
        z_df = z_df.T
        # Quantiles are lower and upper bounds of the uncertainty region
        z_df.columns = ['LB', 'UB']
        # Linearly interpolate the uncertainty region on the range of the initial scores
        # May change x=x to linspace of finer grained values on range of z
        range_z = np.linspace(min(z), max(z), num=100)
        z_df_inter = pd.DataFrame({'LB': np.interp(x=range_z, xp=z, fp=np.array(z_df['LB'])),
                                   'UB': np.interp(x=range_z, xp=z, fp=np.array(z_df['UB']))})
        quants = z_df_inter

    elif uncertainty == "discrete":
        # Discrete Asymptotics
        # Only consistency allowed, because confidence doesn't converege (Dimitiradis Online Appendix p.10)
        q_snd = norm.ppf(q=1 - (level / 2))
        z, c = np.unique(x, return_counts=True)
        uni_count = pd.DataFrame({'Unique': z, 'Count': c})
        uni_count['frac'] = np.sqrt(
            uni_count['Unique'] * (1 - uni_count['Unique']) / uni_count['Count']) * q_snd
        uni_count['LB'] = uni_count['Unique'] - uni_count['frac']
        uni_count['UB'] = uni_count['Unique'] + uni_count['frac']
        uni_count.loc[(uni_count.LB < 0), 'LB'] = 0
        uni_count.loc[(uni_count.UB > 1), 'UB'] = 1
        range_z = z
        quants = pd.DataFrame({'LB': np.interp(x=x, xp=z, fp=uni_count.loc[:, 'LB']),
                               'UB': np.interp(x=x, xp=z, fp=uni_count.loc[:, 'UB'])})
        #quants = uni_count.drop(['Unique', 'Count', 'frac'], axis=1)
    elif uncertainty == "continuous":
        print("Caution: Density Estimation is not bounded!")
        def density_estimation(x_eval):
            from sklearn.neighbors import KernelDensity
            x_reshape = x_eval.reshape(-1,1)
            kde = KernelDensity()
            kde.fit(x_reshape)
            density = kde.score_samples(x_reshape)
            return(np.exp(density))
        def qchern(q_eval):
            from scipy.interpolate import interp1d
            chern_x = np.array([0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,
                0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
                0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79,
                0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
                0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
                0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 0.9999])
            chern_y = np.array([0.000000, 0.013187, 0.026383, 0.039595, 0.052830,
                    0.066096, 0.079402, 0.092757, 0.106168, 0.119645,
                    0.133196, 0.146831, 0.160560, 0.174393, 0.188342,
                    0.202418, 0.216633, 0.230999, 0.245530, 0.260242,
                    0.275151, 0.290274, 0.305629, 0.321238, 0.337123,
                    0.353308, 0.369821, 0.386694, 0.403959, 0.421656,
                    0.439828, 0.458525, 0.477804, 0.497731, 0.518383,
                    0.539855, 0.562252, 0.585706, 0.610378, 0.636468,
                    0.664235, 0.694004, 0.726216, 0.761477, 0.800658,
                    0.845081, 0.896904, 0.960057, 1.043030, 1.171530,
                    1.189813, 1.209897, 1.232241, 1.257496, 1.286659,
                    1.321370, 1.364637, 1.423026, 1.516664, 1.784955])
            chern_fct = interp1d(x=chern_x, y=chern_y)
            q_chern = chern_fct(q_eval)
            return q_chern
        z, c = np.unique(x, return_counts=True)
        density = density_estimation(x_eval=z)
        uni_count = pd.DataFrame({'Unique': z, 'Count': c, 'Density': density})
        qchern = qchern(q_eval=1-(level/2))
        uni_count['frac'] = (pow(uni_count['Unique'] * (1 - uni_count['Unique']) / (2*len(z)*uni_count['Density']), (1/3)))*qchern
        uni_count['LB'] = uni_count['Unique'] - uni_count['frac']
        uni_count['UB'] = uni_count['Unique'] + uni_count['frac']
        uni_count.loc[(uni_count.LB < 0), 'LB'] = 0
        uni_count.loc[(uni_count.UB > 1), 'UB'] = 1
        quants = pd.DataFrame({'LB': np.interp(x=x, xp=z, fp=uni_count.loc[:, 'LB']),
                               'UB': np.interp(x=x, xp=z, fp=uni_count.loc[:, 'UB'])})
        range_z = z


    else:
        print("Please choose from 'resampling', 'discrete' or 'continuous'")

    quants = quants.to_numpy()
    iso_df = pd.DataFrame({'X': x, 'Y': y, 'Fit': isofit})
    uncert_df = pd.DataFrame({'Z': range_z, 'LB': quants[:, 0], 'UB': quants[:, 1]})
    return iso_df, uncert_df

def plot_diagram(iso_data, uncert_data):
    import matplotlib.pyplot as plt
    iso_data = iso_data.to_numpy()
    x = iso_data[:, 0]
    y = iso_data[:, 1]
    isofit = iso_data[:, 2]
    uncert_data = uncert_data.to_numpy()
    z = uncert_data[:, 0]
    LB = uncert_data[:, 1]
    UB = uncert_data[:, 2]
    plt.close('all')
    plt.plot(x, isofit, label='fitted')
    plt.scatter(x, isofit)
    plt.scatter(x, y, label='actual')
    plt.axline([0, 0], [1, 1], color='grey')
    plt.plot(z, LB, linestyle='dotted',
             color='lightblue')  # Replace X with other variable because quants are only defined for unique x i.e. z
    plt.plot(z, UB, linestyle='dotted', color='lightblue')
    plt.fill_between(z, LB, UB, alpha=0.2, color='lightblue',
                     label='Uncertainty')
    plt.legend(loc=[0.65, 0.1])
    plt.xlabel('Index')
    plt.ylabel('Fitted')
    plt.title('Isotonic Regression')
    plt.grid()
    plt.show()

def calc_Score_Decomp(iso_data):
    import numpy as np
    import pandas as pd
    iso_data = iso_data.to_numpy()
    x = iso_data[:, 0]
    y = iso_data[:, 1]
    isofit = iso_data[:, 2]
    def calcMSE(y_act, y_pred):
        diff = np.array(y_act) - np.array(y_pred)
        sq = np.square(diff)
        return sq.mean()
    if max(x) > 1:
        print('Probabilities outside logical range [0, 1]')
    else:  # round first to ensure perfect decomposition
        r = np.array(y.mean())
        S_x = calcMSE(x, y)
        S_c = calcMSE(isofit, y)
        S_r = calcMSE(y, r)
        MCB = np.array(S_x - S_c)
        DSC = np.array(S_r - S_c)
        UNC = np.array(S_r)
        Metrics = {'Discrimination': DSC, 'Miscalibration': MCB, 'Uncertainty': UNC, 'Frac Positives': r,
                   'Mean Brier Score': S_x, 'Brier Score': S_c}
        CORP_Decomp = pd.DataFrame([Metrics])
        CORP_Decomp = CORP_Decomp.astype('float64').transpose().round(5)
        CORP_Decomp.columns = ['Value']
        return CORP_Decomp
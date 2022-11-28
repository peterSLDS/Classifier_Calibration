## Python Functions needed for Calibration Analysis

# level = 0.1
# uncertainty = "resampling"
# type = "confidence"
# m=100

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
        z = np.unique(x)
        z_df = pd.DataFrame({'Z': z})
        x_df = pd.DataFrame({'X': x})
        for l in range(1, m + 1):
            y_resample = []
            iso_bs = IsotonicRegression()
            x_resample = x_df.sample(n=len(x_df), replace=True)
            for i in range(1, (len(x) + 1)):
                if type == "consistency":
                    bern = bernoulli(x_resample.iloc[i - 1, 0])
                    y_resample.append(bern.rvs(1)[0])
                if type == "confidence":
                    bern = bernoulli(isoreg.f_(x_resample.iloc[i - 1, 0]))
                    y_resample.append(bern.rvs(1)[0])
            x_resample = np.array(x_resample.loc[:, 'X'])
            y_resample = np.array(y_resample)
            iso_bs.fit(x_resample, y_resample)
            x_uni = np.unique(x_resample)
            isofit_bs = np.array(iso_bs.predict(x_uni))
            bs_row = pd.DataFrame({'Z': x_uni, 'fit_bs' + str(l): isofit_bs})
            bs_row = bs_row.sort_values('Z', axis=0, ascending=True)
            z_df = z_df.merge(bs_row, on='Z', how='left')
        z_df = z_df.T
        z_df = z_df.rename(columns=z_df.iloc[0]).drop(z_df.index[0])
        z_df = z_df.quantile(q=[(level / 2), 1 - (level / 2)], axis=0, numeric_only=True, interpolation='linear')
        z_df = z_df.T
        z_df.columns = ['LB', 'UB']
        quants = z_df

    elif uncertainty == "discrete":
        # Discrete Asymptotics
        q_snd = norm.ppf(q=1 - (level / 2))
        z, c = np.unique(x, return_counts=True)
        uni_count = pd.DataFrame({'Unique': z, 'Count': c})
        uni_count['frac'] = np.sqrt(
            uni_count['Unique'] * (1 - uni_count['Unique']) / uni_count['Count']) * q_snd
        uni_count['LB'] = uni_count['Unique'] - uni_count['frac']
        uni_count['UB'] = uni_count['Unique'] + uni_count['frac']
        uni_count.loc[(uni_count.LB < 0), 'LB'] = 0
        uni_count.loc[(uni_count.UB > 1), 'UB'] = 1
        quants = uni_count.drop(['Unique', 'Count', 'frac'], axis=1)
    elif uncertainty == "contiuous":
        print("Not yet available")
    else:
        print("Please choose from 'resampling', 'discrete' or 'continuous'")

    quants = quants.to_numpy()
    iso_df = pd.DataFrame({'X': x, 'Y': y, 'Fit': isofit})
    uncert_df = pd.DataFrame({'Z':z, 'LB': quants[:, 0], 'UB': quants[:, 1]})
    return iso_df, uncert_df

#iso_out, uncert_out = calc_prep(x=x, y=y, level=0.1, m=100, uncertainty='resampling', type='confidence')

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

#plot_diagram(iso_data=iso_out, uncert_data=uncert_out)

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

#calc_Score_Decomp(iso_data=iso_out)
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:31:16 2022

@author: Liam
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import probplot
from hmmlearn import hmm
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering


# from scipy.stats.mstats import winsorize
# from sklearn.linear_model import SGDClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from sklearn.naive_bayes import GaussianNB
# from scipy.stats import randint as sp_randint
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# import tensorflow as tf

"""
0. Directory Manipulations
"""

os.chdir("/Users/liamsvoboda/Documents/HEC Lausanne/Master/Sem 4 Spring 2022/Thesis Data")

if not os.path.isdir('Plots'):
    os.makedirs('Plots')  
    
if not os.path.isdir('Tables'):
    os.makedirs('Tables') 
    
if not os.path.isdir('Results'):
    os.makedirs('Results')   
   
insample_dir = os.path.join('Results', 'Insample')
if not os.path.exists(insample_dir):
    os.makedirs(insample_dir)

outsample_dir = os.path.join('Results', 'Outsample')
if not os.path.exists(outsample_dir):
    os.makedirs(outsample_dir)     
    
"""
1 Importing Factor Data
"""

def import_treatment(df, dayfirst = True):
    
    """

    Parameters
    ----------
    df : DataFrame
        DataFrame resulting from .csv import

    Returns
    -------
    df : DataFrame
        Modifies the original DataFrame to be usable in further computations
        
    """
    
    df.set_index(df.columns[0], inplace=True)
    df.index =  pd.to_datetime(df.index,dayfirst=dayfirst)
    df = df.sort_index(axis = 1)
    df = df.sort_index(axis = 0)
    df = df[df.index.notnull()]
    df = df.apply(pd.to_numeric)
        
    # global total_data_points
    # total_data_points += df.size #used to count the total number of datapoints used

    return df


def plot_and_save(df, benchmark=None, factor=None, name_modifier='', colour_changer=None):
    fig, ax = plt.subplots()
    ax.plot(df.index, df.values, label=df.name)

    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.values, label=benchmark.name)
    if factor is not None:
        ax.plot(factor.index, factor.values, label=factor.name)

    ax.set_title(df.name + name_modifier)
    ax.legend()
    ax.set_xlim(df.index[0], df.index[-1])

    if colour_changer is not None:
        for i in range(len(df.index) - 1):
            if colour_changer[i] == 1:
                ax.axvspan(df.index[i], df.index[i + 1], facecolor='gray', alpha=0.5)
                ax.axvspan(0, 1, facecolor='gray', alpha=0.5)
                ax.plot([], [], color='gray', alpha=0.5, label='Stress Periods')

    if benchmark is None and factor is None:
        filename = f'Plots/{df.name}{name_modifier}.png'
    elif benchmark is not None and factor is None:
        filename = f'Plots/{df.name}{name_modifier}_vs_{benchmark.name}.png'
    elif benchmark is not None and factor is not None:
        filename = f'Plots/{df.name}{name_modifier}_vs_{benchmark.name}_{factor.name}.png'

    ax.legend()
    plt.savefig(filename)
    plt.show()


def describe_and_save(df, name_modifier = ''):
    """
    
    Parameters
    ----------
    df : DataFrame
        Dataframe for which we want descriptive statistics.
    name_modifier : String, optional
        Text to add to table header and file name. Don't forgat to add a "_" 
        before added text for to remain consistent. The default is ''.

    Returns
    -------
    None. Creates a .tex table which can be easily added to a latex document.

    """
    
    data = df.values.flatten()
    des = pd.DataFrame(data).describe(percentiles = [.25, .3, .5, .7, .75]) 
    #30 and 70 percentiles added since these are the portfolio cutoff values
    exec(f"des.to_latex(buf='Tables/{df.name}{name_modifier}_stats.tex', header=['{df.name}{name_modifier}'])")  #

"""
1.2 Stock Returns 
"""


Prices = pd.read_csv("Total Return Index.csv",sep=';',decimal=",")
Prices = import_treatment(Prices, dayfirst = True)
Prices.name = 'Prices'
describe_and_save(Prices)

Returns = Prices/Prices.shift(1)-1 # set returns for month t as retunrs between periods t+1 and t for ptf/factor returns computaiton
Returns = Returns.drop(Returns.index[0])
Returns.replace([-np.inf, np.inf], np.nan, inplace = True) #if base price was 0, returns become infinite

Returns.name = 'Returns'
describe_and_save(Returns)

Investible = Returns.notna() & ~Returns.eq(0) 

start_date = Returns.index[0]
end_date = Returns.index[-1]


"""
1.1 Data Descriptives
"""

Stocks = pd.read_csv("Stock_info.csv",sep=';')
Stocks.index = Stocks['Ticker']
Unique=Stocks.describe()[1:2][["ISIN", "Region","Country", "Sector", "Industry"]]

def geo_industry_dist(df):
    """
    Parameters:
    df : DataFrame
        DataFrame of portfolio weights.
    Returns:
    None. Plots bar charts for the average region, country, sector, and industry 
    distribution.
    Due to the number of industries and countries, only 50% of values are shown for this 
     

    """
    Stocks_join = Stocks.copy()
    Stocks_join = Stocks_join.merge(df, left_index=True, right_index=True, suffixes=('', '_df'))

    distributions = [('Country', 'Country'), ('Sector', 'Sector'), ('Region', 'Region'), ('Industry', 'Industry')]   
    
    for col_name, plot_title in distributions:
        dist = Stocks_join[[col_name, Stocks_join.columns[-1]]].copy()
        dist = Stocks_join.groupby(col_name)[Stocks_join.columns[-1]].sum().reset_index()
        dist = dist.sort_values(Stocks_join.columns[-1], ascending=False)
        fig, ax = plt.subplots()
        bars = ax.bar(dist[col_name], dist[Stocks_join.columns[-1]])
        ax.set_title(f"{df.name} Distribution by {plot_title}", fontsize=12)
        ax.tick_params(axis='x', labelrotation=90, labelsize=6)
        ax.tick_params(axis='y')
        plt.subplots_adjust(bottom=0.3)        
        plt.tight_layout()
        plt.savefig(f"Plots/{df.name} distribution by {plot_title}.png", dpi=300)
        plt.show()
        
    return None
  
"""
1.1.1 Equally-Weighted Distributions
"""
EW_df = pd.DataFrame(1 / len(Stocks.index), index=Stocks.index, columns=['Value'])
EW_df.name = 'Company'
geo_industry_dist(EW_df)

"""
1.1.1 Cap-Weighted Distributions
"""

Market_caps = pd.read_csv("Market_Caps.csv",sep=',',decimal=".")
Market_caps = import_treatment(Market_caps,dayfirst = True)
Market_caps = Market_caps*Investible 

Market_caps.name = 'Market_caps'
describe_and_save(Market_caps)

Cap_weighting = Market_caps[Market_caps.index >= Returns.index[0]]
Cap_weighting = Cap_weighting.div(Cap_weighting.sum(axis = 1), axis=0)

Avg_cap_weighting = Cap_weighting.sum(axis=0).to_frame()
Avg_cap_weighting = Avg_cap_weighting / Avg_cap_weighting.sum()
Avg_cap_weighting.name = 'Market Cap'
geo_industry_dist(Avg_cap_weighting)

"""
1.2 Stock data 
"""

def df_ranked_z(raw_data):
    
    """
    Parameters
    ----------
    df : DataFrame
        Dataframe of company data or sub-factors z-scores

    Returns
    -------
    df_ranked_z : DataFrame
        Returns a df of cross-sectional z-scores of the ranked inputted df

    """
    
    df = raw_data.copy()
    df[Investible == False] = np.nan #removes univestible stocks from universe
    df_ranked = df.rank(axis=1)
    df_ranked_mean = df_ranked.mean(axis=1)
    df_ranked_std = df_ranked.std(axis=1)
    df_ranked_z = df_ranked.subtract(df_ranked_mean,axis="rows")
    df_ranked_z = df_ranked_z.divide(df_ranked_std,axis="rows")
    # df_ranked_z = df_ranked_z.fillna(df_ranked_z.median(axis=1)) #does this make sense ? Does it matter given Investible mask ?
    
    return df_ranked_z

"""
1.2.1 Profitability Data
"""

Roe = pd.read_csv("Profitability/ROE.csv",sep=';',decimal=",")
Roe = import_treatment(Roe, dayfirst = True)
Roe.name = 'ROE'
describe_and_save(Roe)
Roe_z = df_ranked_z(Roe)
Roe_z.name = 'ROE_z'
describe_and_save(Roe_z)

Roa = pd.read_csv("Profitability/ROA.csv",sep=';',decimal=",")
Roa = import_treatment(Roa)
Roa.name = 'ROA'
describe_and_save(Roa)
Roa_z = df_ranked_z(Roa)
Roa_z.name = 'ROA_z'
describe_and_save(Roa_z)

Cogstosales = pd.read_csv("Profitability/COGS_to_sales.csv",sep=';',decimal=",")
Cogstosales = import_treatment(Cogstosales)
Cogstosales.name = 'COGS_to_Sales'
describe_and_save(Cogstosales)

Cogstosales_z = df_ranked_z(Cogstosales)
Cogstosales_z.name = 'COGS_to_Sales_z'
describe_and_save(Cogstosales_z)

Revenues = pd.read_csv("Profitability/Revenues.csv",sep=';',decimal=",")
Revenues = import_treatment(Revenues)
Revenues.name = 'Revenues'
describe_and_save(Revenues)

Revenues.replace(0, np.nan, inplace=True) #must remove 0s for ratio
Netincome = pd.read_csv("Profitability/Net_Income.csv",sep=';',decimal=",")
Netincome = import_treatment(Netincome)

Netincome.name = 'NetIncome'
describe_and_save(Netincome)

NetProfitMargin = Netincome/Revenues
NetProfitMargin.name = 'PM'
describe_and_save(NetProfitMargin)

NetProfitMargin_z = df_ranked_z(NetProfitMargin)
NetProfitMargin_z.name = 'PM_z'
describe_and_save(NetProfitMargin_z)

Profitability = Roe_z.add(Roa_z, fill_value=0) #filling with 0 assumes normal dist right? so maybe fill with row avg ??
Profitability = Profitability.add(-Cogstosales_z, fill_value=0) #negative since cost ratio and not profit ratio
Profitability = Profitability.add(NetProfitMargin_z, fill_value=0)
Profitability.name = 'Profitability(pre_z)'
describe_and_save(Profitability)

Profitability_z = df_ranked_z(Profitability)
Profitability_z.name = 'Profitability'
describe_and_save(Profitability_z)

"""
1.2.2 Accounting Quality Data
"""

Salesonrec = pd.read_csv("Accounting Quality/Sales_to_receivables.csv",sep=';',decimal=",")
Salesonrec = import_treatment(Salesonrec)
Salesonrec.name = 'SoR'
describe_and_save(Salesonrec)

Salesonrec_z = df_ranked_z(Salesonrec)
Salesonrec_z.name = 'SoR_z'
describe_and_save(Salesonrec_z)

Cfo = pd.read_csv("Accounting Quality/CF_oper.csv",sep=';',decimal=",")
Cfo = import_treatment(Cfo)
Cfo.name = 'CFO'
describe_and_save(Cfo)

Netincome.replace(0, np.nan, inplace=True) #must remove 0s for ratio.

CfoToNetIncome = Cfo/Netincome
CfoToNetIncome.name = 'CFO_to_NI'
describe_and_save(CfoToNetIncome)

CfoToNetIncome_z = df_ranked_z(CfoToNetIncome)
CfoToNetIncome_z.name = 'CFO_to_NI_z'
describe_and_save(CfoToNetIncome_z)

Acc_quality = CfoToNetIncome_z.add(Salesonrec_z, fill_value = 0)
Acc_quality.name = 'Accounting_Quality(pre_z)'
describe_and_save(Acc_quality)

Acc_quality_z = df_ranked_z(Acc_quality)
Acc_quality_z.name = 'Accounting_Quality'
describe_and_save(Acc_quality_z)

"""
1.2.3 Payout Data
"""

Payout_ratio = pd.read_csv("Payout/Payout_ratio.csv",sep=';',decimal=",")
Payout_ratio = import_treatment(Payout_ratio)
Payout_ratio.name = 'Payout_Ratio'
describe_and_save(Payout_ratio)

Payout_ratio_z = df_ranked_z(Payout_ratio)
Payout_ratio_z.name = 'Payout_Ratio_z'
describe_and_save(Payout_ratio_z)

LT_Debt = pd.read_csv("Payout/LT_debt.txt",sep=';',decimal=",")
LT_Debt = import_treatment(LT_Debt)
LT_Debt.name = 'LT_Debt'
describe_and_save(LT_Debt)

Debt_issuance = LT_Debt/LT_Debt.shift(1)-1
Debt_issuance.name = 'DIss'
describe_and_save(Debt_issuance)

Debt_issuance_z = df_ranked_z(Debt_issuance)
Debt_issuance_z.name = 'DIss_z'
describe_and_save(Debt_issuance_z)

Total_assets = pd.read_csv("Payout/Total_Assets.txt",sep=';',decimal=",")
Total_assets = import_treatment(Total_assets)
Total_assets.name = 'Total_Assets'
describe_and_save(Total_assets)

Equity = Total_assets-LT_Debt
Equity.name = 'Equity'
describe_and_save(Equity)

Equity_issuance = Equity/Equity.shift(1)-1
Equity_issuance.name = 'EIss'
describe_and_save(Equity_issuance)

Equity_issuance_z = df_ranked_z(Equity_issuance)
Equity_issuance_z.name = 'EIss_z'
describe_and_save(Equity_issuance_z)

Payout = Payout_ratio_z.add(-Debt_issuance_z, fill_value=0)
Payout = Payout.add(-Equity_issuance_z, fill_value=0)
Payout.name = 'Payout(pre_z)'
describe_and_save(Payout)

Payout_z = df_ranked_z(Payout)
Payout_z.name = 'Payout'
describe_and_save(Payout_z)


"""
1.2.4 Investment Data
"""

CapEx = pd.read_csv("Investment/Capex.csv",sep=';',decimal=",")
CapEx = import_treatment(CapEx)
CapEx.name = 'CapEx'
describe_and_save(CapEx)

DnA = pd.read_csv("Investment/Depr_and_Amort.csv",sep=';',decimal=",")
DnA = import_treatment(DnA)
DnA.name = 'DnA'
describe_and_save(DnA)

DnA.replace(0, np.nan, inplace=True) #must remove 0s for ratio
CapexToDnA = CapEx/DnA
CapexToDnA.name = 'Capex_to_DnA'
describe_and_save(CapexToDnA)

CapexToDnA_z = df_ranked_z(CapexToDnA) # not too high but still above 1??
CapexToDnA_z.name = 'Capex_to_DnA_z'
describe_and_save(CapexToDnA_z)

Asset_growth = Total_assets/Total_assets.shift(1)-1
Asset_growth.name = 'Asset_Growth'
describe_and_save(Asset_growth)

Asset_growth_z = df_ranked_z(Asset_growth)
Asset_growth_z.name = 'Asset_Growth_z'
describe_and_save(Asset_growth_z)

CapEx_growth = CapEx/CapEx.shift(1)-1
CapEx_growth.name = 'CapEx_Growth'
describe_and_save(CapEx_growth)

CapEx_growth_z = df_ranked_z(CapEx_growth)
CapEx_growth_z.name = 'CapEx_Growth_z'
describe_and_save(CapEx_growth_z)

Investment  = CapexToDnA_z.add(Asset_growth_z, fill_value=0)
Investment  = Investment.add(CapEx_growth_z, fill_value=0)
Investment = -Investment #all metrics are negatively associated with quality
Investment.name = 'Investment(pre_z)'
describe_and_save(Investment)

Investment_z = df_ranked_z(Investment)
Investment_z.name = 'Investment'
describe_and_save(Investment_z)

"""
1.2.6 Quality  Data
"""

Quality = Profitability_z.add(Acc_quality_z, fill_value=0)
Quality = Quality.add(Payout_z, fill_value=0)
Quality = Quality.add(Investment_z, fill_value=0)
Quality.name = 'Quality(pre_z)'
describe_and_save(Quality)

Quality_z = df_ranked_z(Quality)
Quality_z.name = 'Quality'
describe_and_save(Quality)

factor_list = [Quality_z, Profitability_z, Acc_quality_z, Payout_z, Investment_z]


neg_metrics = [Cogstosales_z.name, Debt_issuance_z.name, Equity_issuance_z.name, CapexToDnA_z.name, Asset_growth_z.name, CapEx_growth_z.name]

financial_metrics = [Roe_z, Roa_z, Cogstosales_z, NetProfitMargin_z, Salesonrec_z, CfoToNetIncome_z, Payout_ratio_z, Debt_issuance_z, Equity_issuance_z, CapexToDnA_z, Asset_growth_z, CapEx_growth_z]

"""
2 Building the Portfolios 
"""

def df_to_daily(df):
    """

    Parameters
    ----------
    df : DataFrame
        Dataframe we want to transform to a daily time series.

    Raises
    ------
    ValueError
        In order to differenciate between true NaNs which are present in the 
        original df and the ones which arise when the new rows are created between
        observations, the original NaNs are temporarily replaced with a placeholder
        When no suitable placeholder is found, (unprobable), an error is raised

    Returns
    -------
    df : DataFrame
        Transforms the quarterly / monthly DataFrame into a daily one 
        to be used with the returns matrix.       

    """


    
    nan_placeholder = -9999 
    present_values = np.unique(df.to_numpy())
    present_values = present_values[~np.isnan(present_values)]

    if nan_placeholder in present_values:
        for i in np.arange(-len(present_values)/100, -len(present_values)/100 + 0.01, 0.01):
            if i not in present_values:
                nan_placeholder = i
                break
        
    if nan_placeholder in present_values:
        raise ValueError("No suitable nan_placeholder value found in the dataframe.")

    df.fillna(nan_placeholder, inplace=True) #this is pretty slow but it's the only way I found to stop true nans and not only new row nans to be overridden by ffill()
    all_indexes = df.index.union(Returns.index)
    df = df.reindex(all_indexes)
    df = df.ffill()
    df.replace(nan_placeholder, np.nan, inplace = True)
    df = df[df.index.isin(Returns.index)]

    return df

def factor_portfolio(df, cap_weight = True, percentile = 30, long_only = False):
 
    """

    Parameters
    ----------
    df : DataFrame
        Dataframe of factor z-scores.
    cap_weight : Boolean, optional
        False means the portfolio is equal-weight and True means it is cap-weighted. The default is False.
    percentile : Int, optional
        Determines the top percentile to buy and, if applicable, the bottom percentile to sell. The default is 30.
    long_only : long_only, optional
        False means the portfolio is long-short and True means it is long-only. The default is False.
    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    weights : DataFrame
        A Dataframe of weights for each stock at each time period.
    factor_returns : DataFrame
        A Dataframe of returns for the factor portfolio at each time period
    cumulative_returns : DataFrame
        A Dataframe of rthe cumulative returns for the factor portfolio at each time period

    """
    
    z_scores = df.copy()
    
    if Prices.index[0] not in z_scores.index:
        z_scores.loc[start_date] = z_scores.loc[z_scores.index[2]] #not the most wise way to do this for quarterly factors...
    else:
        z_scores.loc[start_date] = z_scores.loc[Prices.index[0]]

    z_scores = z_scores[z_scores.index >= start_date]
    z_scores = z_scores.sort_index(axis = 0)
  
    top_percentile_thresholds = z_scores.apply(lambda row: np.nanpercentile(row, 100-percentile), axis=1)
    if not(long_only):
        bottom_percentile_thresholds = z_scores.apply(lambda row: np.nanpercentile(row, percentile), axis=1)
    
    long_companies = z_scores.gt(top_percentile_thresholds, axis=0)
    if not(long_only):
        short_companies = z_scores.le(bottom_percentile_thresholds, axis=0)
    
    weights = pd.DataFrame(0, index=z_scores.index, columns=z_scores.columns)
        
    for date in tqdm(z_scores.index, desc="Calculating weights for " + str(df.name) + " cap_weight = " + str(cap_weight) + " long_only= " + str(long_only), leave=False):
        long_mask = long_companies.loc[date] 
        
        if not(long_only):
            short_mask = short_companies.loc[date]

        if cap_weight:
                        
            weights.loc[date, long_mask] = Market_caps.loc[date, long_mask] / Market_caps.loc[date, long_mask].sum()
            
            if not(long_only):
                weights.loc[date, short_mask] = - Market_caps.loc[date, short_mask] / Market_caps.loc[date, short_mask].sum()
       
        else:
            
            long_weight = 1 / long_mask.sum() if long_mask.any() else 0
            if not(long_only):
                short_weight = -1 / short_mask.sum() if short_mask.any() else 0

            weights.loc[date, long_mask] = long_weight
            if not(long_only):
                weights.loc[date, short_mask] = short_weight
        
    # Using df_to_daily()) here allows for a the company weights to remain constant between observation dates.
    # An alternative would be to make the dataframes daily as soon as they are downloaded.
    # (i.e. when the import_treatment() funciton is applied).
    # However, this leads to daily weights which vary wildly.
    
    weights = df_to_daily(weights)
    
    return weights #, factor_returns, Cumulative_returns

"""
2.1 Sub-Factors and Quality Factor Weights
"""

list_of_factor_weights_cw = []
list_of_factor_weights_ew = []
list_of_factor_weights_cw_lo = []
list_of_factor_weights_ew_lo = []

for factor in tqdm(factor_list, desc="Calculating factor weights", leave=False):    

    ###### Cap-Weighted Long-Short
    
    weights = factor_portfolio(factor)
    name_modifier = '_Weights_CW'
    exec(f"{factor.name}{name_modifier} = weights")
    exec(f"{factor.name}{name_modifier}.name = '{factor.name}'")  
    exec(f"list_of_factor_weights_cw.append({factor.name}{name_modifier})")
    exec(f"describe_and_save({factor.name}{name_modifier}, name_modifier = name_modifier)")
    
    net_dist = weights.sum(axis = 0)/len(weights.index)
    net_dist.name = factor.name + '_CW_Net'
    geo_industry_dist(net_dist) 
    
    long_dist = weights[weights >= 0].sum(axis = 0)/len(weights.index)
    long_dist.name = factor.name + '_CW_Long'
    geo_industry_dist(long_dist) 
    
    short_dist = abs(weights[weights < 0].sum(axis=0)/len(weights.index))
    short_dist.name = factor.name + '_CW_Short'
    geo_industry_dist(short_dist) 
    
    # ###### Equal-Weighted Long-Short

    # weights = factor_portfolio(factor, cap_weight = False)
    # name_modifier = '_Weights_EW'
    # exec(f"{factor.name}{name_modifier} = weights")
    # exec(f"{factor.name}{name_modifier}.name = '{factor.name}'")  
    # exec(f"list_of_factor_weights_ew.append({factor.name}{name_modifier})")
    # exec(f"describe_and_save({factor.name}{name_modifier}, name_modifier = name_modifier)")
    
    # net_dist = weights.sum(axis = 0)/len(weights.index)
    # net_dist.name = factor.name + '_EW_Net'
    # geo_industry_dist(net_dist) 
    
    # long_dist = weights[weights >= 0].sum(axis = 0)/len(weights.index)
    # long_dist.name = factor.name + '_EW_Long'
    # geo_industry_dist(long_dist) 
    
    # short_dist = abs(weights[weights < 0].sum(axis=0)/len(weights.index))
    # short_dist.name = factor.name + '_EW_Short'
    # geo_industry_dist(short_dist) 
 
    # ###### Cap-Weighted Long-only  
    
    # weights = factor_portfolio(factor,long_only = True)
    # name_modifier = '_Weights_CW_LO'
    # exec(f"{factor.name}{name_modifier} = weights")
    # exec(f"{factor.name}{name_modifier}.name = '{factor.name}'")  
    # exec(f"list_of_factor_weights_cw_lo.append({factor.name}{name_modifier})")
    # exec(f"describe_and_save({factor.name}{name_modifier}, name_modifier = name_modifier)")
      
    # net_dist = weights.sum(axis = 0)/len(weights.index)
    # net_dist.name = factor.name + '_CW_LO'
    # geo_industry_dist(net_dist) 
    
    # ###### Equal-Weighted Long-Only
        
    # weights = factor_portfolio(factor, cap_weight = False,long_only = True)
    # name_modifier = '_Weights_EW_LO'
    # exec(f"{factor.name}{name_modifier} = weights")
    # exec(f"{factor.name}{name_modifier}.name = '{factor.name}'")  
    # exec(f"list_of_factor_weights_ew_lo.append({factor.name}{name_modifier})")
    # exec(f"describe_and_save({factor.name}{name_modifier}, name_modifier = name_modifier)")

    # net_dist = weights.sum(axis = 0)/len(weights.index)
    # net_dist.name = factor.name + '_EW_LO'
    # geo_industry_dist(net_dist) 

"""
2.2 Individual Financial Metrics
"""

list_of_financial_metric_weights_cw = []

for fm in tqdm(financial_metrics, desc="Calculating financial metric weights", leave=False):    

    if fm.name in neg_metrics:
        weights = -factor_portfolio(fm)
    else:
        weights = factor_portfolio(fm)
        
    name_modifier = '_Weights_CW'
    exec(f"{fm.name}{name_modifier} = weights")
    exec(f"{fm.name}{name_modifier}.name = '{fm.name}'")  
    exec(f"list_of_financial_metric_weights_cw.append({fm.name}{name_modifier})")
    exec(f"describe_and_save({fm.name}{name_modifier}, name_modifier = name_modifier)")
                       
"""
2.3 MSCI World
"""

MSCI_world = pd.read_csv("MSCI World.csv",sep=',', decimal='.')
MSCI_world.set_index(MSCI_world.columns[0], inplace=True)
MSCI_world.index =  pd.to_datetime(MSCI_world.index, dayfirst=True)   
MSCI_world = MSCI_world.replace('.', np.NaN)
MSCI_world = MSCI_world.apply(pd.to_numeric)
MSCI_world.name = 'MSCI_World'
plot_and_save(MSCI_world)
describe_and_save(MSCI_world)

"""
3 Determining Insample Performance
"""

def indicator_import_manip(df, dayfirst = False):

    df.set_index(df.columns[0], inplace=True)
    df = df.replace('.', np.NaN)
    df = df.apply(pd.to_numeric)
    df.index =  pd.to_datetime(df.index, dayfirst = dayfirst)   
    df = df.interpolate()
    # df = df[df.index >= start_date]
    df = df[df.index <= end_date]
    
    return df

def cumulative_returns(returns_df):
    
    cum_return = (returns_df + 1).cumprod()
    
    return cum_return
    
def total_return(cum_returns):
    
    total_return = cum_returns[-1]#.iloc[-1][0]
    
    return total_return

def annualized_total_return(cum_returns, total_return):
    
    annualized_return = total_return ** (252/len(cum_returns))-1
    
    return annualized_return


def risk_adjusted_return(returns, benchmark_returns):
    
    if all(returns == benchmark_returns):
        
        risk_adjusted_return = np.nan
        
    else: 
        
        excess_returns = returns - benchmark_returns
        risk_adjusted_return = excess_returns.mean()/excess_returns.std()
        risk_adjusted_return = risk_adjusted_return*np.sqrt(252)
        
    return risk_adjusted_return  
        
def sortino_ratio(returns, rf):
    
    if all(returns == rf):
        
        sortino_ratio = np.nan
        
    else: 
        
        excess_returns = returns - rf
        downside_returns = np.where(returns < 0, returns, 0)

        sortino_ratio = excess_returns.mean()/downside_returns.std()
        sortino_ratio = sortino_ratio*np.sqrt(252)       

    return sortino_ratio

def max_drawdown(cum_returns):
    
    peak = np.maximum.accumulate(cum_returns)
    max_drawdown = (-1)*np.min(cum_returns/peak-1)

    return max_drawdown

def hit_ratio(daily_returns):
    
    hit_ratio = len(daily_returns[daily_returns >= 0]) / len(daily_returns)
    
    return hit_ratio

def var_es_95(returns):
    sorted_returns = np.sort(returns)
    index = int(len(sorted_returns) * (1 - 0.05))
    var_95 = sorted_returns[index]
    es_95= sorted_returns[index:].mean()
    
    return var_95, es_95

def downside_capture_ratio(returns, benchmark):
    strategy_downsidebench_returns = np.where(benchmark < 0, returns, 0)
    benchmark_downside_returns = np.where(benchmark < 0, benchmark, 0)
    
    strategy_average_downbench_return = np.mean(strategy_downsidebench_returns)
    benchmark_average_downside_return = np.mean(benchmark_downside_returns)
    
    downside_capture_ratio = strategy_average_downbench_return / benchmark_average_downside_return
    
    return downside_capture_ratio

def perf_metrics(returns, benchmark, rf):
    
    cum_returns = cumulative_returns(returns)
    tot_return = total_return(cum_returns)
    ann_return = annualized_total_return(cum_returns, tot_return)
    # Annualized_returns = annualized_returns(returns)
    sharpe_ratio = risk_adjusted_return(returns,rf)
    info_ratio = risk_adjusted_return(returns,benchmark) #outsample_returns['MSCI WORLD'])
    max_dd = max_drawdown(cum_returns)
    hr = hit_ratio(returns)
    volatility = np.sqrt(np.var(returns.values.flatten()))*np.sqrt(252)
    downside_returns = np.where(returns < 0, returns, 0)
    downside_volatility = np.sqrt(np.var(downside_returns))*np.sqrt(252)
    sortino = sortino_ratio(returns, rf)
    beta = np.cov(returns.values.flatten(), benchmark.values.flatten())[0, 1]/np.var(benchmark.values.flatten())
    correlation_Pearson, p_value_corr_pearson = stats.pearsonr(returns.values.flatten(), benchmark.values.flatten())
    correlation_Spearman, p_value_corr_spearman = stats.spearmanr(returns.values.flatten(), benchmark.values.flatten())
    var_95, es_95 = var_es_95(returns)
    downside_capture = downside_capture_ratio(returns, benchmark)
    
    data = [ann_return, volatility, downside_volatility, sharpe_ratio, info_ratio, sortino, beta, correlation_Pearson, correlation_Spearman, max_dd, hr, var_95, es_95, downside_capture]
    results = pd.DataFrame(data, columns= [returns.name], index=['Annualized_total_return', 'Volatility', 'Downside_volatility', 'Sharpe_Ratio', 'Information_Ratio', 'Sortino_Ratio', 'Beta', 'Pearson_Correlation', 'Spearman_Correlation', 'Max_Drawdown', 'Hit_Ratio', 'VaR_95%', 'Expected_Shortfall_95%', 'Downside_capture_ratio'])

    return results

def corr_matrix_plot(return_list,name_modifier = ''):
    corr_temp = pd.concat(return_list, axis=1)
    corr = corr_temp.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True) 
    sns.heatmap(corr, cmap=cmap, mask=mask, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    filename = f'Plots/{name_modifier}_correlation_heatmap.png'
    ax.set_title(name_modifier, fontsize=20)
    plt.savefig(filename)

def perf_versus_bench(factor_returns, benchmark, Rf, model_returns = None):
    """

    Parameters
    ----------
    returns : DataFrame
        DataFrame of returns of the portfolio to analyze.
    benchmark : DataFrame
        DataFrame of returns of the benchmark.
    Rf : DataFrame
        DataFrame of returns of the risk-free rate.

    Returns
    -------
    DataFrame of various performance metrics for the portfolio and the benchmark.

    """

    
    comp = pd.DataFrame(index=['Annualized_total_return', 'Volatility', 'Downside_volatility', 'Sharpe_Ratio', 'Information_Ratio', 'Sortino_Ratio', 'Beta', 'Pearson_Correlation', 'Spearman_Correlation', 'Max_Drawdown', 'Hit_Ratio', 'VaR_95%', 'Expected_Shortfall_95%', 'Downside_capture_ratio'])

    # Annualized_total_return, Volatility, SR, IR, Sortino_ratio, Beta, Correlation_Pearson, Correlation_Spearman, Max_drawdown, Hit_ratio = perf_metrics(ret_insample.iloc[:, 0], MSCI_World_insample.iloc[:, 0], Rf_insample)
    comp['Benchmark'] = perf_metrics(benchmark, benchmark, Rf)
    comp[factor_returns.name] = perf_metrics(factor_returns, benchmark, Rf)
    
    if factor_returns is not None:
        comp['Strategy'] = perf_metrics(model_returns, benchmark, Rf)
    return comp





def defensive_score(perf_results):
    """

    Parameters
    ----------
    perf_results : DataFrame
        Dataframe of output of perf_versus_bench().

    Returns
    -------
    defensiveness : Int
        An integer score of all the times the risk metrics of interest are better for the portfolio than for the benchmark.

    """
    
    defensiveness = 0
    defensive_metrics = ['Volatility','Downside_volatility', 'Beta','Max_Drawdown','VaR_95%','Expected_Shortfall_95%', 'Downside_capture_ratio' ]

    for dm in defensive_metrics:
        
        if results.loc[dm, results.columns[1]] < results.loc[dm, results.columns[0]]: #naïve comparison is probably not sufficient, should evaluate if the difference is statistically significant.
        
            defensiveness += 1
    
    return defensiveness
   
def save_results(df, name_modifier = ''):

    exec(f"df.to_latex(buf='Tables/{df.name}{name_modifier}.tex', header=['MSCI World','{df.name}{name_modifier}'])")  #
  
#In-sample period represents 70% of periods, the model validation samples will
#be 15% if any are needed and the rest is used as a test sample

Test_size = 0.3 

"""
3.1 Risk-Free Rate
"""

TenY = pd.read_csv("FRED data/DGS10.csv",sep=',') #Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity
TenY = indicator_import_manip(TenY)
TenY.name = 'TenYear'
plot_and_save(TenY)
describe_and_save(TenY)
Rf = pd.merge(Returns.iloc[:,0], TenY, left_index=True, right_index=True, how='left').iloc[:,1]
Rf = ((1+Rf)**(1/252)-1)/100 #find daily equivalent to use for Sharpe Ratio computation
Rf_insample, Rf_outsample = train_test_split(Rf, test_size=Test_size, shuffle=False)

"""
3.2 Market Portfolio
"""

MSCI_world_returns = MSCI_world/MSCI_world.shift(1)-1
MSCI_world_returns = MSCI_world_returns.drop(MSCI_world_returns.index[0])
MSCI_world_returns.replace([-np.inf, np.inf], np.nan, inplace = True) #if base price was 0, returns become infinite
MSCI_world_returns = MSCI_world_returns[MSCI_world_returns.index >= start_date]
MSCI_world_returns = MSCI_world_returns[MSCI_world_returns.index <= end_date]
MSCI_world_returns.name = 'MSCI_World_Returns'
plot_and_save(MSCI_world_returns)

describe_and_save(MSCI_world_returns)

MSCI_World_insample, MSCI_World_outsample = train_test_split(MSCI_world_returns.squeeze(), test_size=Test_size, shuffle=False)

MSCI_World_Insample_Cum_Returns = cumulative_returns(MSCI_World_insample)

# portfolio = Quality_Insample_Returns_CW
# benchmark = MSCI_World_insample

def returns_v_bench_scatter(portfolio, benchmark, name_modifier='', return_table = False, trend_line = True):
    # Scatter plot of portfolio returns vs. benchmark returns
    plt.figure(figsize=(8, 6))
    plt.scatter(portfolio, benchmark)
    plt.xlabel(portfolio.name + name_modifier)
    plt.ylabel(benchmark.name)
    plt.title('Returns Scatter Plot')
    
    if trend_line:
        model = sm.OLS(benchmark, sm.add_constant(portfolio))
        results = model.fit()
        trendline = results.fittedvalues  
        plt.plot(portfolio, trendline, color='red', label='Trend Line')
        plt.legend()
        
        #make table with coefs for significance test
        coefficients_table = pd.DataFrame({'Coefficient': results.params, 'Std. Error': results.bse, 'T-value': results.tvalues, 'P-value': results.pvalues})
        coefficients_table = coefficients_table.rename(index={'const': 'Intercept', portfolio.name: 'Slope'})
        table_title = f"Coefficients for {portfolio.name}{name_modifier} on {benchmark.name}"
        coefficients_table.to_latex(buf=f"Tables/{portfolio.name}{name_modifier}_{benchmark.name}_Regression_Coefficients.tex", caption=table_title)
        
        if return_table:
            return coefficients_table
        
    filename = f'Plots/{portfolio.name}{name_modifier}_{benchmark.name}_Scatter.png'
    plt.savefig(filename)
    plt.show()

      
"""
3.3 Factor Portfolios Insmaple Results
"""

"""
3.3.1 Cap-Weighted Long-Short
"""

list_of_factor_returns_cw = []

for wgt in list_of_factor_weights_cw:
    returns = (Returns * wgt).sum(axis = 1)
    returns.name = wgt.name
    name_modifier = '_Returns_CW'
    exec(f"{returns.name}{name_modifier} = returns")
    exec(f"{returns.name}{name_modifier}.name = '{returns.name}'")
    exec(f"{returns.name}{name_modifier}.to_csv('{returns.name}{name_modifier}.csv')")
    exec(f"list_of_factor_returns_cw.append({returns.name}{name_modifier})")
    
corr_matrix_plot(list_of_factor_returns_cw,'Cap-Weighted')

list_of_factor_results_cw = []   
for ret in list_of_factor_returns_cw:
    ret_insample = train_test_split(ret, test_size=Test_size, shuffle=False)[0]#.to_frame()
    name_modifier = '_Insample_Returns_CW'
    exec(f"{ret.name}{name_modifier} = ret_insample")
    exec(f"{ret.name}{name_modifier}.name = ret.name")
    exec(f"plot_and_save({ret.name}{name_modifier},name_modifier = name_modifier)")
    exec(f"describe_and_save({ret.name}{name_modifier}, name_modifier = name_modifier)")
    exec(f"returns_v_bench_scatter({ret.name}{name_modifier}, MSCI_World_insample, name_modifier= name_modifier)")
    
    name_modifier = '_Insample_Perf_Metrics_CW'
    results = perf_versus_bench(ret_insample, MSCI_World_insample, Rf_insample).round(2)
    exec(f"{ret.name}{name_modifier} = results")
    exec(f"{ret.name}{name_modifier}.name = ret.name")
    exec(f"list_of_factor_results_cw.append({ret.name}{name_modifier})")   

    Cum_returns = cumulative_returns(ret_insample)
    name_modifier = '_Insample_Cumulative_Returns_CW'
    exec(f"{ret.name}{name_modifier} = Cum_returns")
    exec(f"{ret.name}{name_modifier}.name = ret.name")
    exec(f"plot_and_save({ret.name}{name_modifier}, MSCI_World_Insample_Cum_Returns, name_modifier = name_modifier)")
    
for res in list_of_factor_results_cw:
    save_results(res, name_modifier = '_Results_CW')  
        
# """
# 3.3.2 Equal-Weighted Long-Short
# """   
    
# list_of_factor_returns_ew = []

# for wgt in list_of_factor_weights_ew:
#     returns = (Returns * wgt).sum(axis = 1)
#     returns.name = wgt.name
#     name_modifier = '_Returns_EW'
#     exec(f"{returns.name}{name_modifier} = returns")
#     exec(f"{returns.name}{name_modifier}.name = '{returns.name}'")
#     exec(f"list_of_factor_returns_ew.append({returns.name}{name_modifier})")
 
# corr_matrix_plot(list_of_factor_returns_ew, 'Equal-Weighted')   
# list_of_factor_results_ew = []   
 
# for ret in list_of_factor_returns_ew:
#     ret_insample = train_test_split(ret, test_size=Test_size, shuffle=False)[0]#.to_frame()
#     name_modifier = '_Insample_Returns_EW'
#     exec(f"{ret.name}{name_modifier} = ret_insample")
#     exec(f"{ret.name}{name_modifier}.name = ret.name")
#     exec(f"plot_and_save({ret.name}{name_modifier}, name_modifier = name_modifier)")
#     exec(f"describe_and_save({ret.name}{name_modifier}, name_modifier = name_modifier)")
#     exec(f"returns_v_bench_scatter({ret.name}{name_modifier}, MSCI_World_insample, name_modifier= name_modifier)")

#     name_modifier = '_Insample_Perf_Metrics_EW'
#     results = perf_versus_bench(ret_insample, MSCI_World_insample, Rf_insample).round(2)
#     exec(f"{ret.name}{name_modifier} = results")
#     exec(f"{ret.name}{name_modifier}.name = ret.name")
#     exec(f"list_of_factor_results_ew.append({ret.name}{name_modifier})")   

#     Cum_returns = cumulative_returns(ret_insample)
#     name_modifier = '_Insample_Cumulative_Returns_EW'
#     exec(f"{ret.name}{name_modifier} = Cum_returns")
#     exec(f"{ret.name}{name_modifier}.name = ret.name")
#     exec(f"plot_and_save({ret.name}{name_modifier}, MSCI_World_Insample_Cum_Returns, name_modifier = name_modifier)")

# for res in list_of_factor_results_ew:
#     save_results(res, name_modifier = '_Results_EW')  
    
# """
# 3.3.3 Cap-Weighted Long-Only
# """
    
# list_of_factor_returns_cw_lo = [] 

# for wgt in list_of_factor_weights_cw_lo:
#     returns = (Returns * wgt).sum(axis = 1)
#     returns.name = wgt.name
#     name_modifier = '_Returns_CW_LO'
#     exec(f"{returns.name}{name_modifier} = returns")
#     exec(f"{returns.name}{name_modifier}.name = '{returns.name}'")
#     exec(f"list_of_factor_returns_cw_lo.append({returns.name}{name_modifier})")

# corr_matrix_plot(list_of_factor_returns_cw_lo, 'Cap-Weighted Long-Only')  
# list_of_factor_results_cw_lo = []   
    
# for ret in list_of_factor_returns_cw_lo:
#     ret_insample = train_test_split(ret, test_size=Test_size, shuffle=False)[0]#.to_frame()
#     name_modifier = '_Insample_Returns_CW_LO'
#     exec(f"{ret.name}{name_modifier} = ret_insample")
#     exec(f"{ret.name}{name_modifier}.name = ret.name")
#     exec(f"plot_and_save({ret.name}{name_modifier}, name_modifier = name_modifier)")
#     exec(f"describe_and_save({ret.name}{name_modifier}, name_modifier = name_modifier)")
#     exec(f"returns_v_bench_scatter({ret.name}{name_modifier}, MSCI_World_insample, name_modifier= name_modifier)")

#     name_modifier = '_Insample_Perf_Metrics_CW_LO'
#     results = perf_versus_bench(ret_insample, MSCI_World_insample, Rf_insample).round(2)
#     exec(f"{ret.name}{name_modifier} = results")
#     exec(f"{ret.name}{name_modifier}.name = ret.name")
#     exec(f"list_of_factor_results_cw_lo.append({ret.name}{name_modifier})")   

#     Cum_returns = cumulative_returns(ret_insample)
#     name_modifier = '_Insample_Cumulative_Returns_CW_LO'
#     exec(f"{ret.name}{name_modifier} = Cum_returns")
#     exec(f"{ret.name}{name_modifier}.name = ret.name")
#     exec(f"plot_and_save({ret.name}{name_modifier}, MSCI_World_Insample_Cum_Returns, name_modifier = name_modifier)")
    
# for res in list_of_factor_results_cw_lo:
#     save_results(res, name_modifier = '_Results_CW_LO')  
    
# """
# 3.3.4 Equal-Weighted Long-Only
# """
    
# list_of_factor_returns_ew_lo = []
 
# for wgt in list_of_factor_weights_ew_lo:
#     returns = (Returns * wgt).sum(axis = 1)
#     returns.name = wgt.name
#     name_modifier = '_Returns_EW_LO'
#     exec(f"{returns.name}{name_modifier} = returns")
#     exec(f"{returns.name}{name_modifier}.name = '{returns.name}'")
#     exec(f"list_of_factor_returns_ew_lo.append({returns.name}{name_modifier})")
    
# corr_matrix_plot(list_of_factor_returns_ew_lo, 'Equal-Weighted Long-Only')   
# list_of_factor_results_ew_lo = []    
    
# for ret in list_of_factor_returns_ew_lo:
#     ret_insample = train_test_split(ret, test_size=Test_size, shuffle=False)[0]#.to_frame()
#     name_modifier = '_Insample_Returns_EW_LO'
#     exec(f"{ret.name}{name_modifier} = ret_insample")
#     exec(f"{ret.name}{name_modifier}.name = ret.name")
#     exec(f"plot_and_save({ret.name}{name_modifier}, name_modifier = name_modifier)")
#     exec(f"describe_and_save({ret.name}{name_modifier}, name_modifier = name_modifier)")
#     exec(f"returns_v_bench_scatter({ret.name}{name_modifier}, MSCI_World_insample, name_modifier= name_modifier)")
    
#     name_modifier = '_Insample_Perf_Metrics_EW_LO'
#     results = perf_versus_bench(ret_insample, MSCI_World_insample, Rf_insample).round(2)
#     exec(f"{ret.name}{name_modifier} = results")
#     exec(f"{ret.name}{name_modifier}.name = ret.name")
#     exec(f"list_of_factor_results_ew_lo.append({ret.name}{name_modifier})")   

#     Cum_returns = cumulative_returns(ret_insample)
#     name_modifier = '_Insample_Cumulative_Returns_EW_LO'
#     exec(f"{ret.name}{name_modifier} = Cum_returns")
#     exec(f"{ret.name}{name_modifier}.name = ret.name")
#     exec(f"plot_and_save({ret.name}{name_modifier}, MSCI_World_Insample_Cum_Returns, name_modifier = name_modifier)")

# for res in list_of_factor_results_ew_lo:
#     save_results(res, name_modifier = '_Results_EW_LO')  
     
"""
3.3.5 Financial Metric Portfolios
"""

# list_of_financial_metric_returns_cw = []

# for wgt in list_of_financial_metric_weights_cw:
    
#     net_dist = wgt.sum(axis = 0)/len(wgt.index)
#     net_dist.name = wgt.name + '_CW_Net'
#     geo_industry_dist(net_dist) 
    
#     long_dist = wgt[wgt >= 0].sum(axis = 0)/len(wgt.index)
#     long_dist.name = wgt.name + '_CW_Long'
#     geo_industry_dist(long_dist) 
    
#     short_dist = abs(wgt[wgt < 0].sum(axis=0)/len(wgt.index))
#     short_dist.name = wgt.name + '_CW_Short'
#     geo_industry_dist(short_dist)
    
#     returns = (Returns * wgt).sum(axis = 1)
#     returns.name = wgt.name
#     name_modifier = '_Returns_CW'
#     exec(f"{returns.name}{name_modifier} = returns")
#     exec(f"{returns.name}{name_modifier}.name = '{returns.name}'")
#     exec(f"{returns.name}{name_modifier}.to_csv('{returns.name}{name_modifier}.csv')")
#     exec(f"list_of_financial_metric_returns_cw.append({returns.name}{name_modifier})")   

# corr_matrix_plot(list_of_financial_metric_returns_cw)   
# list_of_financial_metric_results_cw = []

# for ret in list_of_financial_metric_returns_cw:
#     ret_insample = train_test_split(ret, test_size=Test_size, shuffle=False)[0]#.to_frame()
#     name_modifier = '_Insample_Returns_CW'
#     exec(f"{ret.name}{name_modifier} = ret_insample")
#     exec(f"{ret.name}{name_modifier}.name = ret.name")
#     exec(f"plot_and_save({ret.name}{name_modifier}, name_modifier = name_modifier)")
#     exec(f"describe_and_save({ret.name}{name_modifier}, name_modifier = name_modifier)")
#     exec(f"returns_v_bench_scatter({ret.name}{name_modifier}, MSCI_World_insample, name_modifier = name_modifier)")

#     name_modifier = '_Insample_Perf_Metrics_CW'
#     results = perf_versus_bench(ret_insample, MSCI_World_insample, Rf_insample).round(2)
#     exec(f"{ret.name}{name_modifier} = results")
#     exec(f"{ret.name}{name_modifier}.name = ret.name")
#     exec(f"list_of_financial_metric_results_cw.append({ret.name}{name_modifier})")   

#     Cum_returns = cumulative_returns(ret_insample)
#     name_modifier = '_Insample_Cumulative_Returns_CW'
#     exec(f"{ret.name}{name_modifier} = Cum_returns")
#     exec(f"{ret.name}{name_modifier}.name = ret.name")
#     exec(f"plot_and_save({ret.name}{name_modifier}, MSCI_World_Insample_Cum_Returns, name_modifier = name_modifier)")
 

# for res in list_of_financial_metric_results_cw:
#     save_results(res, name_modifier = '_Results_CW')  
     
"""
4. Stress Indicators
"""
def dickey_fuller_test(df):
    """
    Performs a Dickey-Fuller test to the column, that is, H0: the column is 
    not stationary.
    
    Note that we use the type 2 D-F test which makes us select the most 
    conservative critical values.
    
    Parameters:
        column: a time series
    
    Returns:
        relevant information regarding the test
    """
    x = sm.add_constant(df.values[:-1])
    y = df.values[1:]
    results = sm.OLS(y,x).fit()
    
    alpha = results.params[0]
    beta = results.params[1]
    
    sum_squared_eps = 0
    for t in range(1,len(y)):
        sum_squared_eps += (y[t]-alpha-beta*y[t-1])**2
    var_eps = (1/(len(df)-1))*sum_squared_eps
    
    y_mean = np.mean(y)
    
    sum_squared_price_dev = 0
    for t in range(1,len(y)):
        sum_squared_price_dev += (y[t]-y_mean)**2
    
    std_beta = (var_eps/sum_squared_price_dev)**0.5
    
    tstat = (beta - 1) / std_beta
    if tstat <= -3.963: #stat taken from https://real-statistics.com/statistics-tables/augmented-dickey-fuller-table/
                        # is the type 2 DF tesk 
        verdict = 'Stationary' 
    else:
        verdict = 'Non-Stationary'
    
    result = pd.Series(data = [np.round(beta,2), np.round(tstat,2), verdict], index=['beta','t-stat', 'verdict'])   
    exec(f"result.to_latex(buf='Tables/{df.name}_dickey_fuller.tex', header=['{df.name}'])")  #

    return result 
    
"""
4.1 Standalone Financial Market Indicators
"""
      
"""
4.1.1 Standalone Financial Market Indicators Import
"""

Vix = pd.read_csv("FRED data/VIXCLS.csv",sep=',')
Vix = indicator_import_manip(Vix)
Vix.name = 'VIX'

TenYReal = pd.read_csv("FRED data/10YRealInterestRate.csv",sep=',')
TenYReal = indicator_import_manip(TenYReal)
TenYReal.name = 'TenYear_Real'

FedRate = pd.read_csv("FRED data/FederalFundsRate.csv",sep=',')
FedRate = indicator_import_manip(FedRate)
FedRate.name = 'Fed_Rate'

TenYTwoY = pd.read_csv("FRED data/T10Y2Y.csv",sep=',')
TenYTwoY = indicator_import_manip(TenYTwoY)
TenYTwoY.name = 'TenY_TwoY'

TenYThreeM = pd.read_csv("FRED data/T10Y3M.csv",sep=',')
TenYThreeM = indicator_import_manip(TenYThreeM)
TenYThreeM.name = 'TenY_ThreeM'

Market_indicators = [Vix, TenYReal, TenY, FedRate, TenYTwoY, TenYThreeM]  
  
"""
4.1.1 Standalone Financial Market Indicators Analysis
"""

Market_indicators_stationary = []

Market_indicators_non_stationary = []

for ind in Market_indicators:
    
    temp = ind.name
    df_test = ind[ind.index >= start_date]#only want to test over our in sample period but want to have the older data for potential transformaitons.
    df_test = df_test[df_test.index <= MSCI_World_insample.index[-1]]
    df_test.name = temp
    plot_and_save(df_test)
    stationary_test = dickey_fuller_test(df_test)
    
    if stationary_test.loc['verdict'] == 'Non-Stationary':
    
        Market_indicators_non_stationary.append(ind)
        
    else:
        Market_indicators_stationary.append(ind)   
                 
"""
4.1.1 Standalone Financial Market Indicators Transformation
"""

def differencing(df):
    
    df_diff = df - df.shift(1)
    
    return df_diff

indicator_diff = []

for df in Market_indicators_non_stationary:
    
    exec(f"{df.name}_diff = differencing(df)")
    exec(f"{df.name}_diff.name = df.name + '_diff'")
    # exec(f"{df.name}_diff.rename(columns={'{df.name}_diff.columns[0]': '{{df.name}_diff..columns[0]}_diff'})")
    exec(f"plot_and_save({df.name}_diff)")
    exec(f"indicator_diff.append({df.name}_diff)")
    
for df_diff in indicator_diff:
    
    df_temp = df_diff.copy()
    df_temp = df_temp[df_temp.index >= start_date]
    df_temp = df_temp[df_temp.index <= MSCI_World_insample.index[-1]]
    df_temp.name = df_diff.name
    stationary_test = dickey_fuller_test(df_temp)
    
    if stationary_test.loc['verdict'] == 'Stationary':
    
        Market_indicators_stationary.append(df_diff)
        
    # else:
    #     Market_indicators_stationary_2.append(ind) 
    
    
    # len(Market_indicators_stationary) == len(Market_indicators)
    # ^ is True so no need to go further as they are made stationary by first 
    # order differencing. If not, difference until stationary.
        
corr_matrix_plot(Market_indicators_stationary,'Market_indicators_stationary')
    
"""
4.2 Financial Stress Indices
"""

def remove_col_spaces(df):
    """
    
    Parameters
    ----------
    df : DataFrame
        Dataframe with spaces in col names we want to remove. This is to be 
        able to dynamically assign variable names for example.

    Returns
    -------
    None.

    """
    
    new_columns = [col.replace(" ", "_") for col in df.columns]
    df.columns = new_columns

FSIs = []

Ofr_fsi_all_cols = pd.read_csv("FSIs/OFR_FSI.csv",sep=',')
Ofr_fsi_all_cols = indicator_import_manip(Ofr_fsi_all_cols)
remove_col_spaces(Ofr_fsi_all_cols)

for col in Ofr_fsi_all_cols.columns:
    exec(f"{col} = Ofr_fsi['{col}'].to_frame()")
    exec(f"{col}.name = '{col}'")
    exec(f"FSIs.append({col})")
    
Kc_fsi = pd.read_csv("FSIs/KC_FSI.csv",sep=',')
Kc_fsi = indicator_import_manip(Kc_fsi)
Kc_fsi.name = 'KC_FSI'

Stl_fsi = pd.read_csv("FSIs/STL_FSI.csv",sep=',')
Stl_fsi = indicator_import_manip(Stl_fsi)
Stl_fsi.name = 'STL_FSI'

FSIs = FSIs +[Kc_fsi, Stl_fsi]    

FSIs_stationary = []

FSIs_non_stationary = []


for ind in FSIs:
    
    temp = ind.name
    df_test = ind[ind.index >= start_date]#only want to test over our in sample period but want to have the older data for potential transformaitons.
    df_test = df_test[df_test.index <= MSCI_World_insample.index[-1]]
    df_test.name = temp
    plot_and_save(df_test)
    stationary_test = dickey_fuller_test(df_test)
    
    if stationary_test.loc['verdict'] == 'Non-Stationary':
    
        FSIs_non_stationary.append(ind)
        
    else:
        FSIs_stationary.append(ind)   
        
 
FSI_diff = []

for df in FSIs_non_stationary:
    
    exec(f"{df.name}_diff = differencing(df)")
    exec(f"{df.name}_diff.name = df.name + '_diff'")
    # exec(f"{df.name}_diff.rename(columns={'{df.name}_diff.columns[0]': '{{df.name}_diff..columns[0]}_diff'})")
    exec(f"FSI_diff.append({df.name}_diff)")      
        
for df_diff in FSI_diff:
    
    df_temp = df_diff.copy()
    df_temp = df_temp[df_temp.index >= start_date]
    df_temp = df_temp[df_temp.index <= MSCI_World_insample.index[-1]]
    df_temp.name = df_diff.name
    plot_and_save(df_temp)
    stationary_test = dickey_fuller_test(df_temp)
    
    if stationary_test.loc['verdict'] == 'Stationary':
    
        FSIs_stationary.append(df_diff)
        
# corr_matrix_plot(FSIs,'FSIs')


Indicators = Market_indicators_stationary + FSIs_stationary



Qual = list_of_factor_returns_cw[0]
Qual_insample, Qual_outsample = train_test_split(Qual, test_size=Test_size, shuffle=False)


Relevant_indicators = []
# Selected_indicators_corr = []

for ind in Indicators:

    temp = ind.name
    ind_insample = df_to_daily(ind)
    ind_insample = ind_insample[ind_insample.index >= MSCI_World_insample.index[0]]
    ind_insample = ind_insample[ind_insample.index <= MSCI_World_insample.index[-1]]
    ind_insample.name = temp

    msci_reg = returns_v_bench_scatter(ind_insample, MSCI_World_insample, return_table = True)
    qual_reg = returns_v_bench_scatter(ind_insample, Qual_insample, return_table = True)
        
    if np.sign(msci_reg.iloc[1,0]) != np.sign(qual_reg.iloc[1,0]) and qual_reg.iloc[1,3] < 0.01 and msci_reg.iloc[1,3] < 0.01:
        
        ind = df_to_daily(ind)
        ind = ind[ind.index >= start_date]
        ind = ind[ind.index <= Returns.index[-1]]
        ind.name = temp
        Relevant_indicators.append(ind)
        
Relevant_indicators
    ######### This snipept was intended to be used to select the indicators with
    ######### a significant correlation coefficient but it leads to only 2 indicators being removed
    # corr_Qual = pd.concat([ind_insample,Qual_insample], axis = 1).corr().iloc[0,1]
    # corr_Qual_t_stat = corr_Qual*np.sqrt(len(ind_insample)-2)/np.sqrt(1-corr_Qual**2)
    # corr_MSCI = pd.concat([ind_insample,MSCI_World_insample], axis = 1).corr().iloc[0,1]
    # corr_MSIC_t_stat = corr_MSCI*np.sqrt(len(ind_insample)-2)/np.sqrt(1-corr_MSCI**2)
    # critical_value = 0.047006 #c.f. https://real-statistics.com/statistics-tables/pearsons-correlation-table/
    # if  np.sign(corr_MSCI) != np.sign(corr_Qual) and abs(corr_MSIC_t_stat) > critical_value and abs(corr_Qual_t_stat) > critical_value:
    #     Selected_indicators_corr.append(ind)
    


# test = OFR_FSI.copy()
# test = df_to_daily(test)
# test = test[test.index >= start_date]
# test = test[test.index <= MSCI_World_insample.index[-1]]
# test.name = 'OFR_FSI_raw'
# returns_v_bench_scatter(test,MSCI_World_insample)
    
corr_temp = pd.concat([ind_insample[ind_insample.index <= MSCI_World_insample.index[-1]],Qual_insample], axis=1)

corr_matrix_plot(Relevant_indicators,'Relevant Indicators')

corr_matrix_plot([Qual_insample]+ Indicators, 'All_Indicators_v_Quality')
corr_matrix_plot([MSCI_World_insample]+ Indicators, 'All_Indicators_v_MSCI_World')

# Calculate VIF for each feature
x_all = pd.concat(Relevant_indicators, axis = 1)
Vif_all = pd.DataFrame()
Vif_all["Feature"] = x_all.columns
Vif_all["VIF"] = [variance_inflation_factor(x_all.values, i) for i in range(x_all.shape[1])]
Vif_all.set_index('Feature',inplace=True)
Vif_all.to_latex(buf='Tables/VIF_relevant.tex')

# threshold = 20
# high_vif_features = Vif_all[Vif_all["VIF"] > threshold]["Feature"]
# X_reduced = X.drop(high_vif_features, axis=1)
x_OFR_drop = x_all.drop(['OFR_FSI'], axis=1) #,'United_States', 'Other_advanced_economies', 'Emerging_markets'
Vif_OFR_drop = pd.DataFrame()
Vif_OFR_drop["Feature"] = x_OFR_drop.columns
Vif_OFR_drop["VIF"] = [variance_inflation_factor(x_OFR_drop.values, i) for i in range(x_OFR_drop.shape[1])]
Vif_OFR_drop.set_index('Feature',inplace=True)
Vif_OFR_drop.to_latex(buf='Tables/VIF_OFR_drop.tex')

x_cat = x_all.drop(['OFR_FSI','United_States', 'Other_advanced_economies', 'Emerging_markets'], axis=1) #,'United_States', 'Other_advanced_economies', 'Emerging_markets'
Vif_cat = pd.DataFrame()
Vif_cat["Feature"] = x_cat.columns
Vif_cat["VIF"] = [variance_inflation_factor(x_cat.values, i) for i in range(x_cat.shape[1])]
Vif_cat.set_index('Feature',inplace=True)
Vif_cat.to_latex(buf='Tables/VIF_categories.tex')

x_geo = x_all.drop(['OFR_FSI','Credit', 'Funding', 'Volatility'], axis=1)
Vif_geo = pd.DataFrame()
Vif_geo["Feature"] = x_geo.columns
Vif_geo["VIF"] = [variance_inflation_factor(x_geo.values, i) for i in range(x_geo.shape[1])]
Vif_geo.set_index('Feature',inplace=True)
Vif_geo.to_latex(buf='Tables/VIF_regions.tex')

np.logical_and(MSCI_World_insample < 0, Qual_insample > 0 ).sum()
np.logical_and(MSCI_World_insample < 0, Qual_insample > MSCI_World_insample).sum()
(Qual_insample > MSCI_World_insample).sum()
(Qual_insample - MSCI_World_insample > 0.02).sum()

Indicator_means = x_cat.mean(axis = 0)

# Check for high VIF values (above a certain threshold) and remove corresponding features



"""
5 Dynamic Factor Portfolios
"""
np.random.seed(42)

# MSIC_return_lag = MSCI_world_returns.shift(1)
# Quality_return_lag = Qual.shift(1)

# Features_Lag_Only = pd.concat([Quality_return_lag, MSIC_return_lag], axis=1) 
# Features_Lag_Only.drop(Features_Lag_Only.index[0], inplace = True)

Features = x_cat.copy()
Features = Features.shift(1)
Features.drop(Features.index[0], inplace = True)

# Features = pd.concat([x_cat, Quality_return_lag, MSIC_return_lag], axis=1) #we incorporate the previous day's performance in the training set
# Features.drop(Features.index[0], inplace = True)
X_train, X_test = train_test_split(Features, test_size=Test_size, shuffle=False)#.to_frame()
all_returns = pd.concat([MSCI_world_returns, Qual], axis=1)
all_returns.drop(all_returns.index[0], inplace = True)



for feature in X_train.columns:
    
    
    print(feature)

    probplot(X_train[feature], plot=plt)
    plt.title(feature)
    filename = 'Plots/'+ feature + '_Q_Q_Plot.png'
    plt.savefig(filename)
    plt.show()

    

def model_eval(return_df, name_modifier = ''):
    
    cum_ret_strat = cumulative_returns(return_df.iloc[:, -1])
    cum_ret_msci = cumulative_returns(return_df.iloc[:, 0])
    cum_ret_qual = cumulative_returns(return_df.iloc[:, 1])
    plot_and_save(cum_ret_strat, benchmark = cum_ret_msci, factor = cum_ret_qual, colour_changer = return_df.iloc[:, -2], name_modifier = name_modifier)
    
    rf_sample = Rf.loc[Rf.index.isin(return_df.index)]
    Perf_metrics = perf_versus_bench(return_df.iloc[:, 1], return_df.iloc[:, 0], rf_sample, model_returns = return_df.iloc[:, -1]).round(2)

    returns_v_bench_scatter(return_df.iloc[:, 1], return_df.iloc[:, -2], name_modifier = name_modifier, trend_line = False)
    returns_v_bench_scatter(return_df.iloc[:, 0], return_df.iloc[:, -2], name_modifier = name_modifier, trend_line = False)
    
    stress_periods = return_df[return_df.iloc[:, -2] == 1]
    calm_periods = return_df[return_df.iloc[:, -2] == 0]
        
    Rf_stress = Rf.loc[Rf.index.isin(stress_periods.index)]
    Rf_calm = Rf.loc[Rf.index.isin(calm_periods.index)]
    
    stressed_results = pd.DataFrame()
    qual_stressed = perf_metrics(stress_periods.iloc[:, 1], stress_periods.iloc[:, 0], Rf_stress)
    msci_stressed = perf_metrics(stress_periods.iloc[:, 0], stress_periods.iloc[:, 0], Rf_stress)
    stressed_results = pd.concat([qual_stressed, msci_stressed], axis = 1)
    
    calm_results = pd.DataFrame()
    qual_calm = perf_metrics(calm_periods.iloc[:, 1], calm_periods.iloc[:, 0], Rf_calm)
    msci_calm = perf_metrics(calm_periods.iloc[:, 0], calm_periods.iloc[:, 0], Rf_calm)
    calm_results = pd.concat([qual_calm, msci_calm], axis = 1)
      
    split_results = pd.concat([calm_results, stressed_results], axis=1, keys=['Calm', 'Stressed']).rename_axis(return_df.iloc[:, -1].name)
    n_periods = pd.Series([len(calm_periods), len(calm_periods), len(stress_periods), len(stress_periods)], index=split_results.columns, name="Number of periods")
    split_results = split_results.append(n_periods).round(2)
    
    # exec(f"returns_{return_df.iloc[:, -1].name}{name_modifier} = return_df")
    # exec(f"regime_metrics_{return_df.iloc[:, -1].name}{name_modifier} = split_results")
    exec(f"split_results.to_latex(buf='Tables/{return_df.iloc[:, -1].name}{name_modifier}_regime_perf_metrics.tex')") #, header=['{model_name}_split_results']
    # exec(f"perf_metrics_{return_df.iloc[:, -1].name}{name_modifier} = Perf_metrics")
    exec(f"Perf_metrics.to_latex(buf='Tables/{return_df.iloc[:, -1].name}{name_modifier}_perf_metrics.tex')") #, header=['{model_name}_outsample_results']

"""
5.1 Rule-Based Investing
"""

FSI_features = []

for fsi in FSIs:

    temp = fsi.name
    fsi = df_to_daily(fsi)
    fsi.name = temp
    FSI_features.append(fsi)
    
FSI_features = pd.concat(FSI_features, axis = 1)
FSI_features = FSI_features.shift(1) #make sre we're not cheating?
FSI_features.drop(FSI_features.index[0], inplace = True)

states = (FSI_features > 0).any(axis=1).astype(int).to_frame(name = 'State')

all_returns_manip = all_returns.copy()

all_returns_manip = pd.concat([all_returns_manip, states], axis=1)
all_returns_manip['Raw_FSI'] = np.where(all_returns_manip['State'] == 1, all_returns_manip.iloc[:, 1], all_returns_manip.iloc[:, 0])

model_eval(all_returns_manip, name_modifier = '_All_Sample')

Insample_returns = all_returns_manip[all_returns_manip.index < MSCI_World_outsample.index[0]]
model_eval(Insample_returns,  name_modifier = '_In_Sample')

Outsample_returns = all_returns_manip[all_returns_manip.index >= MSCI_World_outsample.index[0]]
model_eval(Outsample_returns,  name_modifier = '_Out_Sample')

"""
5.2 Unsupervised Models
"""

# determine best number of components for GMMHMM:
from sklearn.mixture import GaussianMixture


bic_values = []

for n_components in range(1, 6):
    model = GaussianMixture(n_components=n_components)
    model.fit(X_train)
    log_likelihood = model.score(X_train)
    n_features = X_train.shape[1]
    n_samples = X_train.shape[0]
    bic = -2 * log_likelihood + n_components * np.log(n_samples) * n_features
    bic_values.append(bic)

Bic_test = pd.DataFrame(bic_values, columns=['BIC Values']).round(2)

Bic_test['n Components'] = Bic_test.index +1
Bic_test.to_latex('Tables/GMM_component_BIC.tex', index=False)


best_num_components = np.argmin(bic_values) + 1  

#hence, we keep the default n_mix = 1 which euates to a GaussianHMM
 

models = {'HMM' : hmm.GMMHMM(n_components=2), 'K_Means' : KMeans(n_clusters=2)} #        'Spectral_Clustering' : SpectralClustering(n_clusters=2), 'Agglomerative_Clustering' : AgglomerativeClustering(n_clusters=2)

for model_name, model in models.items():
    
    model.fit(X_train)
    states = model.predict(Features)
    
    #redefine abnormal (i.e. most common) state to 1 and normal state to 0 for
    # ease of cumputation since the models simply define the first state as state 0
    
    if sum(states)>=0.5*len(states):
        states = abs(states-1)
    
    state_string = model_name + ' State'
    all_returns_manip = all_returns.copy()

    all_returns_manip[state_string] = states

    all_returns_manip[model_name] = np.where(all_returns_manip[state_string] == 1, all_returns_manip.iloc[:, 1], all_returns_manip.iloc[:, 0])

    model_eval(all_returns_manip, name_modifier = '_All_Sample')

    Outsample_returns = all_returns_manip[all_returns_manip.index >= MSCI_World_outsample.index[0]]
    model_eval(Outsample_returns,  name_modifier = '_Out_Sample')
    
    Insample_returns = all_returns_manip[all_returns_manip.index < MSCI_World_outsample.index[0]]
    model_eval(Insample_returns,  name_modifier = '_In_Sample')

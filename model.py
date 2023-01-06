import pandas as pd
from prophet import Prophet
import pmdarima as pmd
import pandas as pd 
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
import itertools
from sklearn.metrics import mean_squared_error as MSE
 
############################################## 
def working_day():
    date = "01-01-2019	01-02-2019	01-03-2019	01-04-2019	01-05-2019	01-06-2019	01-07-2019	01-08-2019	01-09-2019	01-10-2019	01-11-2019	01-12-2019	01-01-2020	01-02-2020	01-03-2020	01-04-2020	01-05-2020	01-06-2020	01-07-2020	01-08-2020	01-09-2020	01-10-2020	01-11-2020	01-12-2020	01-01-2021	01-02-2021	01-03-2021	01-04-2021	01-05-2021	01-06-2021	01-07-2021	01-08-2021	01-09-2021	01-10-2021	01-11-2021	01-12-2021	01-01-2022	01-02-2022	01-03-2022	01-04-2022	01-05-2022	01-06-2022	01-07-2022	01-08-2022	01-09-2022	01-10-2022	01-11-2022	01-12-2022	01-01-2023	01-02-2023	01-03-2023	01-04-2023	01-05-2023	01-06-2023	01-07-2023	01-08-2023	01-09-2023	01-10-2023	01-11-2023	01-12-2023"
    date = date.split("\t")

    workday = "29	18	31	28	30	30	31	31	29	31	30	31	21	29	31	26	30	30	31	31	28	31	30	31	30	19	31	26	30	24	18	16	21	31	30	31	27	22	31	27	30	30	31	31	27	31	30	31	20	28	31	28	30	30	31	31	27	31	30	30"
    workday = workday.split("\t")

    data = {'Date': date,
            'WD': workday}  

    # working day
    WD = pd.DataFrame(data)
    return WD
############################################## 

#forecast length
def fc_length(n=12):
    fcperiod = n
    return fcperiod

#clean outlier
def clean_outlier(df: pd.DataFrame):
    if 'Model' in df.columns:
      df.drop(['Model'],axis=1,inplace=True)
    df = df.apply(pd.to_numeric)
    df = df.fillna(df.median())
    df = df[(np.abs(df.apply(zscore))<2.4)]
    df = df.fillna(df.median())
    return df
   
############################################## 
#EXOGENOUS VARABILE
def exog_var(df: pd.DataFrame):
    WD = working_day()
    fcperiod = fc_length()
    exog_fit = df.merge(WD[['WD']],left_index=True,right_index=True,how='inner')
    exog_fit = exog_fit.drop(exog_fit.columns.difference(['WD']),axis=1) #drop other column
    exog_fc = WD.merge(df,left_index=True,right_index=True,how='outer',indicator=True).query('_merge == "left_only"') #anti left join
    exog_fc = exog_fc.drop(exog_fc.columns.difference(['WD']),axis=1).head(fcperiod)
    return exog_fit, exog_fc

#create list of forecast date

df_param = pd.DataFrame(data={'Model': ['SES','Holt-Winter','SARIMAX','UCM']})



############################################## 
def HoltWinter(df: pd.DataFrame,alpha=1,beta=1,gamma=1):
 df = clean_outlier(df)
 fcperiod = fc_length()
 df_HW = pd.DataFrame()
 future_index = []
 future_index.append(df.tail(12).index.shift(12,freq="MS"))
 
 HW_param_gridsearch = {  
    'initialization_method': ['heuristic','estimated','legacy-heuristic'],
    'seasonal': ['add','mul'],
    'trend': ['add','mul'],
    'damped_trend': [True,False],
    'use_boxcox': [True,False],
                    }
 HW_all_params = [dict(zip(HW_param_gridsearch.keys(), v)) for v in itertools.product(*HW_param_gridsearch.values())]
 


 for sku in df.columns:
  if alpha+beta+gamma == 3:
        hw =[]
        try:
         for params in HW_all_params:
           hw.append(sm.tsa.ExponentialSmoothing(np.asarray(df[sku]), seasonal_periods=12,**params).fit(optimized=True).aicc)
           minhw = HW_all_params[np.argmin(hw)]
           fitHW = sm.tsa.ExponentialSmoothing(np.asarray(df[sku]), seasonal_periods=12,**minhw).fit(optimized=True)
        except:
         fitHW = sm.tsa.ExponentialSmoothing(np.asarray(df[sku]),seasonal_periods=12,trend='add', seasonal='add',damped_trend=True).fit(optimized=True)
  else:
        try:
         fitHW = sm.tsa.ExponentialSmoothing(np.asarray(df[sku]), initialization_method="heuristic",seasonal_periods=12,trend='add', seasonal='add',damped_trend=True).fit(smoothing_level=alpha,smoothing_trend=beta,smoothing_seasonal=gamma)
        except:
         fitHW = sm.tsa.ExponentialSmoothing(np.asarray(df[sku]),seasonal_periods=12,trend='add', seasonal='add',damped_trend=True).fit(smoothing_level=alpha,smoothing_trend=beta,smoothing_seasonal=gamma)
 arr_forecast = fitHW.forecast(fcperiod)
 df_HW[sku] = arr_forecast
 df_HW.set_index(future_index,inplace=True)
    
 df_HW['Model'] = 'Holt-Winter'
 return df_HW
 ############################################## 
def SARIMAX(df: pd.DataFrame,p=0,q=0,d=0,pseas=0,qseas=0,dseas=0):
    df = clean_outlier(df)
    fcperiod = fc_length()
    df_SARIMAX = pd.DataFrame()
    future_index = []
    future_index.append(df.tail(12).index.shift(12,freq="MS"))

    
    for sku in df.columns:
        if p+q+d+pseas+qseas+dseas == 0:
              #dtest = pmd.arima.ndiffs(np.asarray(df[sku])) #first diff
              #Dtest = pmd.arima.nsdiffs(np.asarray(df[sku]), 12) #seasonal diff
              ap_autoarimamodel = pmd.arima.auto_arima(np.asarray(df[sku]),
                                         information_criterion = 'aicc',
                                         start_p=0, max_p=5,
                                         d=1, max_d=1,
                                         start_q=0, max_q=5,
                                         start_P=0, max_P=2,
                                         start_Q=0, max_Q=2,
                                         D=1,max_D=1,
                                         m=12,seasonal=True,
                                         error_action='warn',trace=True,supress_warnings=True,stepwise=True,random_state=20,n_fits=50)
        else:
               ap_autoarimamodel = pmd.arima.auto_arima(np.asarray(df[sku]), 
                                         start_p=p, max_p=p,
                                         d=d, max_d=d,
                                         start_q=q, max_q=q,
                                         start_P=pseas, max_P=pseas,
                                         start_Q=qseas, max_Q=qseas,
                                         D=dseas,max_D=dseas,
                                         m=12,seasonal=True,
                                         error_action='warn',trace=True,supress_warnings=True,stepwise=True,random_state=20,n_fits=50)
        
    arr_forecast = ap_autoarimamodel.predict(n_periods=fcperiod,return_conf_int = False)
    df_SARIMAX[sku] = arr_forecast

    df_SARIMAX.set_index(future_index,inplace=True)
    df_SARIMAX['Model'] = 'SARIMAX'
    return df_SARIMAX
############################################## 
def UCM(df: pd.DataFrame,f=0,ar=0,ucmmodel='ntrend'):
      df = clean_outlier(df)
      fcperiod = fc_length()
      df_UCM = pd.DataFrame()
      future_index = []
      future_index.append(df.tail(12).index.shift(12,freq="MS"))
      UCM_param_gridsearch = {  
        'level': ['ntrend','lldtrend','strend','rtrend'],
        'cycle': [True,False],
        'irregular': [True,False],
        #'damped_cycle': [True,False],
        'use_exact_diffuse': [True,False],
        'autoregressive': [0,1]
                        }
      UCM_all_params = [dict(zip(UCM_param_gridsearch.keys(), v)) for v in itertools.product(*UCM_param_gridsearch.values())]
      for sku in df.columns:
        if f+ar == 0 and ucmmodel == 'ntrend':
          ucm =[]
          for params in UCM_all_params: 
            ucm.append(sm.tsa.UnobservedComponents(
                                                    np.asarray(df[sku]),
                                                    #exog = exog_fit,
                                                    **params,
                                                    freq_seasonal=[{'period':12,'harmonics':12}]).fit().aicc)
          minUCM = UCM_all_params[np.argmin(ucm)]

          fitUCM = sm.tsa.UnobservedComponents(
              np.asarray(df[sku]),
              #exog = exog_fit,
              **minUCM,
              freq_seasonal=[{'period':12,'harmonics':12}]).fit()
        else:
           fitUCM = sm.tsa.UnobservedComponents(
              np.asarray(df[sku]),
              #exog = exog_fit,
              level= ucmmodel,
              cycle=True,irregular=True,damped_cycle=True,
              #use_exact_diffuse=False,
              autoregressive= ar,
              freq_seasonal=[{'period':12,'harmonics':f}]).fit()
         
            
      arr_forecast = fitUCM.forecast(fcperiod)#,exog = exog_fc)
      df_UCM[sku] = arr_forecast
      df_UCM.set_index(future_index,inplace=True)
 
      df_UCM['Model'] = 'UCM'
      return df_UCM
    
   
############################################## 
def PPhet(df: pd.DataFrame,growth='linear',seasonality='additive',changepoint=0.1, n=5, fourier=12,select_type='Auto'):
    df = clean_outlier(df)
    fcperiod = fc_length()
    df_P = pd.DataFrame()
    
    param_gridsearch = {  
            'changepoint_prior_scale': [0.1, 0.5],
            'growth': ['logistic','linear'],
            #'seasonality_prior_scale': [0.1, 4],
            'seasonality_mode': ['additive', 'multiplicative'],
            'n_changepoints': [3],
                            }

    all_params = [dict(zip(param_gridsearch.keys(), v)) for v in itertools.product(*param_gridsearch.values())]
    rmses = []
    
    for sku in df.columns:
        df_model = pd.DataFrame(df[sku].copy()).reset_index()
        df_model.rename(columns={'Date': 'ds', sku: 'y'},inplace=True)
        df_model['cap'] = df_model.y.quantile(0.95)+1
        df_model['floor'] = df_model.y.quantile(0.1)
        #df_model['wd'] = np.asarray(exog_fit)
        
    #detect if auto or manual
        if select_type != 'Manual':       
            for params in all_params:
                #cross validation search for best fit
                m = (
                    Prophet(**params,weekly_seasonality=False,daily_seasonality=False,yearly_seasonality=False,uncertainty_samples=0)
                            .add_seasonality(name='monthly', period=12, fourier_order=12,prior_scale=0.1)
                            #.add_regressor('wd')
                            .add_country_holidays(country_name='VN')
                            .fit(df_model))

                rmses.append(np.sqrt(MSE(df_model['y'], m.predict(df_model)['yhat'] )))

            best_params = all_params[np.argmin(rmses)]   
            m = (
                Prophet(**best_params,weekly_seasonality=False,daily_seasonality=False,yearly_seasonality=False,uncertainty_samples=0)
                        .add_seasonality(name='monthly', period=12, fourier_order=12,prior_scale=0.1)
                        #.add_regressor('wd')
                        .add_country_holidays(country_name='VN')
                        .fit(df_model)) 
        else:
            m = (
                Prophet(growth=growth,seasonality_mode=seasonality,changepoint_prior_scale = changepoint, n_changepoints=n, weekly_seasonality=False,daily_seasonality=False,yearly_seasonality=False,uncertainty_samples=0)
                        .add_seasonality(name='monthly', period=12, fourier_order=fourier,prior_scale=0.1)
                        #.add_regressor('wd')
                        .add_country_holidays(country_name='VN')
                        .fit(df_model))
            
        df_f = m.make_future_dataframe(periods=fcperiod,freq='MS')
        df_f['cap'] = df_model.y.quantile(0.95)+1
        df_f['floor'] = df_model.y.quantile(0.1)
        df_f['wd'] = np.asarray(np.concatenate((exog_fit.to_numpy(),exog_fc.to_numpy()),axis=0))
        forecast = m.predict(df_f)
        df_P[sku] = forecast.yhat.tail(fcperiod)
    return df_P
    
    

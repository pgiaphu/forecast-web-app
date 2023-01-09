import pandas as pd
from prophet import Prophet
import pmdarima as pmd
import pandas as pd 
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
import itertools
from sklearn.metrics import mean_squared_error as MSE
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
 
############################################## 
def working_day():
    date = "01-01-2019	01-02-2019	01-03-2019	01-04-2019	01-05-2019	01-06-2019	01-07-2019	01-08-2019	01-09-2019	01-10-2019	01-11-2019	01-12-2019	01-01-2020	01-02-2020	01-03-2020	01-04-2020	01-05-2020	01-06-2020	01-07-2020	01-08-2020	01-09-2020	01-10-2020	01-11-2020	01-12-2020	01-01-2021	01-02-2021	01-03-2021	01-04-2021	01-05-2021	01-06-2021	01-07-2021	01-08-2021	01-09-2021	01-10-2021	01-11-2021	01-12-2021	01-01-2022	01-02-2022	01-03-2022	01-04-2022	01-05-2022	01-06-2022	01-07-2022	01-08-2022	01-09-2022	01-10-2022	01-11-2022	01-12-2022	01-01-2023	01-02-2023	01-03-2023	01-04-2023	01-05-2023	01-06-2023	01-07-2023	01-08-2023	01-09-2023	01-10-2023	01-11-2023	01-12-2023"
    date = date.split("\t")

    workday = "29	18	31	28	30	30	31	31	29	31	30	31	21	29	31	26	30	30	31	31	28	31	30	31	30	19	31	26	30	24	18	16	21	31	30	31	27	22	31	27	30	30	31	31	27	31	30	31	20	28	31	28	30	30	31	31	27	31	30	30"
    workday = workday.split("\t")

    # working day
    wd = pd.DataFrame(workday, columns=['WD'],index=date)
    return wd
############################################## 
def Temperature():
    date = "01-01-2019	01-02-2019	01-03-2019	01-04-2019	01-05-2019	01-06-2019	01-07-2019	01-08-2019	01-09-2019	01-10-2019	01-11-2019	01-12-2019	01-01-2020	01-02-2020	01-03-2020	01-04-2020	01-05-2020	01-06-2020	01-07-2020	01-08-2020	01-09-2020	01-10-2020	01-11-2020	01-12-2020	01-01-2021	01-02-2021	01-03-2021	01-04-2021	01-05-2021	01-06-2021	01-07-2021	01-08-2021	01-09-2021	01-10-2021	01-11-2021	01-12-2021	01-01-2022	01-02-2022	01-03-2022	01-04-2022	01-05-2022	01-06-2022	01-07-2022	01-08-2022	01-09-2022	01-10-2022	01-11-2022	01-12-2022	01-01-2023	01-02-2023	01-03-2023	01-04-2023	01-05-2023	01-06-2023	01-07-2023	01-08-2023	01-09-2023	01-10-2023	01-11-2023	01-12-2023	01-01-2024	01-02-2024	01-03-2024	01-04-2024	01-05-2024	01-06-2024	01-07-2024	01-08-2024	01-09-2024	01-10-2024	01-11-2024	01-12-2024"
    date = date.split("\t")              

    tempe = "32,9	34	34,6	35,2	34,9	33,6	33,2	32	31,8	33,2	32	31,9	33,6	34,1	34,7	35,3	35,5	33,5	33	32,8	32,5	31	32,3	31,8	31,6	31,9	34,8	34,1	34	33,3	32,8	33	32	31,6	32,4	31,8	32,7	34	34,4	33,8	33,2	33,6	32,2	32,5	32	31,90322581	32,19009884	31,72024768	32,5705797	33,35954639	34,48137687	34,44146772	34,22633743	33,36176057	32,65186798	32,45691481	31,92987194	31,78416764	32,11494027	31,64335999	32,47216563	33,27521413	34,39948992	34,35083719	34,13980721	33,27721604	32,56399696	32,37006076	31,84410611	31,69721169	32,11494027	31,64335999"
    tempe = tempe.split("\t")

    #data = {'Date': date,
    #      'max': tempe} 
    temperature = pd.DataFrame(tempe, columns=['max'],index=date)
    return temperature

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
    future_index = []
    future_index.append(df.tail(12).index.shift(12,freq="MS"))
    
    param_gridsearch = {  
            'changepoint_prior_scale': [0.001, 0.1, 0.5],
            'growth': ['logistic','linear','flat'],
            #'seasonality_prior_scale': [0.1, 4],
            'seasonality_mode': ['additive', 'multiplicative'],
            'n_changepoints': [5,10],
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
        #df_f['wd'] = np.asarray(np.concatenate((exog_fit.to_numpy(),exog_fc.to_numpy()),axis=0))
        forecast = m.predict(df_f)
        df_P[sku] = forecast.yhat.tail(fcperiod)
        df_P.set_index(future_index,inplace=True)
        df_P['Model'] = 'Prophet'
    return df_P
    
    
################################################
#make dataframe for walk-forward forecast
def make_future_dataframe(self, periods, freq='MS'):
    #make dataframe for forecast
    last_date = self.index.max()
    dates = pd.date_range(
        start=last_date,
        periods=periods + 1,  # An extra in case we include start
        freq=freq)
    dates = dates[dates > last_date]  # Drop start if equals last_date
    dates = dates[:periods]  # Return correct number of periods
    return pd.DataFrame(index= dates)

#feature enginering    
def time_features(df: pd.DataFrame()):
    temperature = Temperature()
    wd = working_day()
    #Creates time series features from datetime index
    df['date'] = pd.to_numeric(df.index, downcast='integer')
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['lag1'] = df.iloc[:,0].shift(1)
    df['lag2'] = df.iloc[:,0].shift(2)
    df['lag3'] = df.iloc[:,0].shift(3)
    df['lag4'] = df.iloc[:,0].shift(4)
    df['wd'] = wd.WD
    df['wd-lag1'] = wd.WD.shift(1)
    df['wd-lag2'] = wd.WD.shift(2)
    df['tempe'] = temperature['max']
    df['tempe-lag1'] = temperature['max'].shift(1)
    df['tempe-lag2'] = temperature['max'].shift(2)
    #return df
    

#FUNCTION TO FIND OPTIMAL PARAMETER
def optimal_fc(df: pd.DataFrame(),model='XGB'):
    X = df.iloc[:,-len(df.columns)+1:]
    y = df.iloc[:,0]
    X_train = X.head(len(X)-1)
    X_test = X.tail(1)
    y_train = y.head(len(y)-1)
    y_test = y.tail(1)
    
    if model =='XGB':

        xgb_param_gridsearch = {  
            'learning_rate': [0.001, 0.01, 0.1],
            'max_depth': [5,10,25],
            'n_estimators': [100,1000],
            'tree_method': ['hist','exact'],
            'max_leaves': [5,10,20,40]
                            }


        xgb_all_params = [dict(zip(xgb_param_gridsearch.keys(), v)) for v in itertools.product(*xgb_param_gridsearch.values())]
        xgb_list =[]
        for params in xgb_all_params:
            xgboost = XGBRegressor(**params,objective='reg:squarederror')
            xgboost.fit(X_train, y_train)
            xgb_list.append(np.sqrt(MSE(y_train, xgboost.predict(X_train)))  )

        rmse_xgb = np.sqrt(MSE(y_train, xgboost.predict(X_train)))
        minxgb = xgb_all_params[np.argmin(xgb_list)]
        bestparam = minxgb
    
    
    else:
        lgbm_param_gridsearch = {  
            'learning_rate': [0.001, 0.01, 0.1],
            'max_depth': [5,10,25],
            'n_estimators': [10,300,1000],
            #'max_leaves': [5,15,30],
            'num_leaves': [2,10,20],
            'min_gain_to_split': [3,10,20],
            'min_sum_hessian_in_leaf': [1,10]
                            }


        lgbm_all_params = [dict(zip(lgbm_param_gridsearch.keys(), v)) for v in itertools.product(*lgbm_param_gridsearch.values())]
        lgbm_list =[]
        for params in lgbm_all_params:
            lgbm = LGBMRegressor(**params)
            lgbm.fit(X_train, y_train)
            lgbm_list.append(np.sqrt(MSE(y_train, lgbm.predict(X_train)))  )

        rmse_lgbm = np.sqrt(MSE(y_train, lgbm.predict(X_train)))
        minlgbm = lgbm_all_params[np.argmin(lgbm_list)]
        bestparam = minlgbm
    
    return bestparam
####################################################################

#FUNCTION TO MAKE FORECAST
####################################################################
# fit an xgboost model and make a one step prediction
def xgboost_forecast(df: pd.DataFrame(),*args):
    X = df.iloc[:,-len(df.columns)+1:]
    y = df.iloc[:,0]
    X_train = X.head(len(X)-1)
    X_test = X.tail(1)
    y_train = y.head(len(y)-1)
    #y_test = y.tail(1)
    
    minxgb = {}
    for i in args:
        for key, value in i.items():
            minxgb[key] = value

    xgboost = XGBRegressor(**minxgb,objective='reg:squarederror')
    
    
    xgboost.fit(X_train, y_train,verbose=False)
    # make a one-step prediction
    yhat = xgboost.predict(X_test)
    #yhat = xgboost.best_estimator_.predict(X_test)
    
    return yhat[0]

def lightgbm_forecast(df: pd.DataFrame(),*args):
    X = df.iloc[:,-len(df.columns)+1:]
    y = df.iloc[:,0]
    X_train = X.head(len(X)-1)
    X_test = X.tail(1)
    y_train = y.head(len(y)-1)
    #y_test = y.tail(1)
    
    
    minlgbm = {}
    for i in args:
        for key, value in i.items():
            minlgbm[key] = value
    
    lgbm = LGBMRegressor(**minlgbm)
    
   
    
    lgbm.fit(X_train, y_train,verbose=False)
    # make a one-step prediction
    yhat = lgbm.predict(X_test)
    #yhat = lgbm.best_estimator_.predict(X_test)
    return yhat[0]

####################################################################

def ML_FC(data: pd.DataFrame, model='XGB',select_type='Auto',*param):
 df_XGB = pd.DataFrame()
 df_LGBM = pd.DataFrame()
 df_fc = pd.DataFrame()
 fcperiod = fc_length()
 future_index = []
 future_index.append(data.tail(12).index.shift(12,freq="MS"))


 for sku in list(data):
     df = pd.DataFrame(data[sku].copy(deep=True))
     if select_type == 'Auto':
      bestparam = optimal_fc(df,model)
     else:
      bestparam = {}
      for i in param:
        for key, value in i.items():
            bestparam[key] = value 
      

     if model =='LGBM':
         for i in range(1,fcperiod+1):
             if i == 1:
                 df_fc = pd.concat([df,make_future_dataframe(df,1)])
             else:
                 df_fc = pd.concat([df_fc,make_future_dataframe(df_fc,1)])
             time_features(df_fc)
             df_fc.iloc[-1:,0] = lightgbm_forecast(df_fc,bestparam)
         df_LGBM[sku] = df_fc[sku].tail(fcperiod)
         df_LGBM['Model'] = 'LightGBM'

     else:
         for i in range(1,fcperiod+1):
             if i == 1:
                 df_fc = pd.concat([df,make_future_dataframe(df,1)])
             else:
                 df_fc = pd.concat([df_fc,make_future_dataframe(df_fc,1)])
             time_features(df_fc)
             df_fc.iloc[-1:,0] = xgboost_forecast(df_fc,bestparam)
         df_XGB[sku] = df_fc[sku].tail(fcperiod)
         df_XGB['Model'] = 'XGB'
         

 df_XGB.set_index(future_index,inplace=True)
 df_LGBM.set_index(future_index,inplace=True)
 return df_XGB, df_LGBM


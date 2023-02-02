import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import pmdarima as pmd
import statsmodels.api as sm
from datetime import datetime, date
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import month_plot,quarter_plot

###################################
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode
from st_aggrid import GridUpdateMode, DataReturnMode

###################################
import model as md
import footer as footer
###################################

def _max_width_():
    max_width_str = f"max-width: 2400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_icon="📊", page_title="Forecast",layout="wide")


###################################

with st.sidebar:
    st.image(
    "https://www.duytan.com/Data/Sites/1/media/home/logo_duytan_scgp-endorsement_full-color-01.png",
    width=300,
    )    
    st.title("TIME SERIES FORECAST")
    st.title("1. Select Data")
    data_options = st.checkbox("Load sample data", False)
    if data_options == True:
        uploaded_file = "AS.xlsx"
    else:
        uploaded_file = st.file_uploader("Choose a Excel file")
    
    
    #####
    #footer.footer()
    
###################################
    if uploaded_file is not None:
        df_as = pd.read_excel(uploaded_file, sheet_name = "Full History")
        df_as = df_as.drop(['Channel','Customer'], axis=1)
        df_as = df_as.fillna(0)
        df_fc = pd.read_excel(uploaded_file, sheet_name = "Forecast")
        df_fc = df_fc.drop(['Channel','Customer','Customer Name'], axis=1)
        df_fc.loc[:, ~df_fc.columns.isin(['Channel', 'Customer','Customer Name','Group','Material','Description'])] = df_fc.loc[:, ~df_fc.columns.isin(['Channel', 'Customer','Customer Name','Group','Material','Description'])].apply(np.ceil)
        
        
        
        st.info(
            f"""
                👆 Upload your .xlsx file to make forecast. Here's a sample file: [Actual Sales](https://docs.google.com/spreadsheets/d/1Kce3dNFZHRZjz7jd7DNRxCEi5AdjUnaQEpOp-k1xKfw/edit?usp=sharing)
                """)
        st.markdown("If you need support, reach out to PhamGiaPhu@duytan.com or call ☎:240")
        
        
    else:
        st.info(
            f"""
                👆 Upload your .xlsx file to make forecast. Here's a sample file: [Actual Sales](https://docs.google.com/spreadsheets/d/1Kce3dNFZHRZjz7jd7DNRxCEi5AdjUnaQEpOp-k1xKfw/edit?usp=sharing)
                """)
        st.markdown("If you need support, reach out to PhamGiaPhu@duytan.com or call ☎:240")
        st.stop()

###################################





st.subheader('2. Data loading 📋')
st.write("Your baseline forecast will show here.")
#st.write(shows)

#########################################

gb = GridOptionsBuilder.from_dataframe(df_fc)
#gb.configure_default_column(shows2.columns[0])
gb.configure_column(df_fc.columns[0], rowGroup=True)
###

#gb.configure_column(shows2['Material'],headerCheckboxSelection=False)
gb.configure_selection(selection_mode="single",use_checkbox=True)

#gb.configure_side_bar()
gb.configure_columns(df_fc.columns.values.tolist(),headerCheckboxSelection=False, editable=True)
js = JsCode("""
            function(e) {
                let api = e.api;
                let rowIndex = e.rowIndex;
                let col = e.column.colId;

                let rowNode = api.getDisplayedRowAtIndex(rowIndex);
                api.flashCells({
                  rowNodes: [rowNode],
                  columns: [col],
                  flashDelay: 10000000000
                });

            };
            """)

gb.configure_grid_options(onCellValueChanged=js)
gridOptions = gb.build()

st.markdown("""
            ###
            Edited cells are highlighted
            """)

#response = AgGrid(shows2, gridOptions=gridOptions, key=shows2.columns[0], allow_unsafe_jscode=True, reload_data=False)


response = AgGrid(
    df_fc,
    gridOptions=gridOptions,
    allow_unsafe_jscode=True, reload_data=False,
    enable_enterprise_modules=True,
    update_mode=GridUpdateMode.MODEL_CHANGED,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    fit_columns_on_grid_load=False,)
    #columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)




#########################################


st.subheader('3. Forecast')

model =  st.radio(
            "Choose your forecast model 👇",
            ['Holt-Winter','UCM', 'SARIMAX', 'Prophet', 'XGBoosting', 'LightGBM'],
            key="visibility",
            label_visibility='visible',
            horizontal=True)




df_baseline = pd.DataFrame(response["selected_rows"])
df_baseline = df_baseline.drop(['Group','Description'], axis=1)
df_baseline = pd.melt(df_baseline,id_vars=df_baseline.columns[0])
df_baseline.rename({'variable': 'Date'}, axis=1, inplace=True)
df_baseline['Date'] = df_baseline['Date'].apply(lambda x: datetime.strptime("{}".format(x),"%d-%m-%Y").date())
df_baseline = pd.DataFrame(df_baseline.pivot('Date','Material','value'))
df_baseline.index = pd.to_datetime(df_baseline.index)
df_baseline = df_baseline.apply(pd.to_numeric)



#selected sku
sku = response["selected_rows"][0]['Material']
df =  df_as.iloc[:,:][df_as.Material == sku]
df = df.drop(['Group'], axis=1)
df = pd.melt(df,id_vars=df.columns[0])
df.rename({'variable': 'Date'}, axis=1, inplace=True)
df['Date'] = df['Date'].apply(lambda x: datetime.strptime("{}".format(x),"%d-%m-%Y").date())
df = pd.DataFrame(df.pivot('Date','Material','value'))
df.index = pd.to_datetime(df.index)
#df = df.apply(pd.to_numeric)

df_HW = pd.DataFrame()
df_SARIMAX = pd.DataFrame()
df_UCM = pd.DataFrame()
df_P = pd.DataFrame()
df_XGB = pd.DataFrame()
df_LGBM = pd.DataFrame()








df_baseline['Model'] = 'Baseline'



plot_type = 'trend'



col1, col2, col3 = st.columns([1,1,5])
with col1:
    if st.button('Trend chart'):
        plot_type = 'trend'
with col2:
    if st.button('Multiple line chart'):
        plot_type = 'multipleline'
select_type = 'auto'
col1, col2 = st.columns([2,8])
with col1:
    st.write('Select your paramater')
    
    if 'XGBoosting' in model or 'LightGBM' in model: 
        select_type =  st.radio(
                "Choose your forecast model 👇",
                ['Manual'],
                key="selecttype",
                label_visibility='collapsed',
                disabled=False,
                horizontal=True)
    else:
        select_type =  st.radio(
                "Choose your forecast model 👇",
                ['Manual'],
                key="selecttype",
                label_visibility='collapsed',
                disabled=False,
                horizontal=True)
 
    if 'Holt-Winter' in model:
        if select_type == 'Manual':
            alpha = st.slider('alpha', 0.00, 1.00, 0.3)
            beta = st.slider('beta', 0.00, 1.00, 0.25)
            gamma = st.slider('gamma', 0.00, 1.00, 0.5)
            df_HW = md.HoltWinter(df,alpha,beta,gamma)
        else:
            df_HW = md.HoltWinter(df)

    if 'UCM' in model:
        if select_type == 'Manual':
            ucmmodel = st.select_slider(
                'Model',
                options=['ntrend','dconstant','llevel','rwalk','dtrend','lldtrend','rwdrift','lltrend','strend','rtrend'])
            ar = st.number_input('Auto Regressive',value=0,min_value=0,max_value=1,step=1)
            f = st.number_input('Fourier order',value=1,min_value=0,max_value=100,step=2)
            df_UCM = md.UCM(df,f,ar,ucmmodel)         
        else:
            df_UCM = md.UCM(df)
    
    if 'SARIMAX' in model:
        if select_type == 'Manual':
            p = st.number_input('p',value=1,min_value=0,max_value=24,step=1)
            q = st.number_input('q',value=1,min_value=0,max_value=24,step=1)
            d = st.number_input('d',value=1,min_value=0,max_value=3,step=1)
            pseas = st.number_input('P',value=1,min_value=0,max_value=6,step=1)
            qseas = st.number_input('Q',value=1,min_value=0,max_value=6,step=1)
            dseas = st.number_input('D',value=1,min_value=0,max_value=3,step=1)
            df_SARIMAX = md.SARIMAX(df,p,q,d,pseas,qseas,dseas) 
        else:
            df_SARIMAX = md.SARIMAX(df)
            
    if 'Prophet' in model:
        if select_type == 'Manual':
            growth = st.select_slider(
                'Trend',
                options=['logistic','linear','flat'])
            seasonality = st.select_slider(
                'Seasonality',
                options=['additive', 'multiplicative'])
            changepoint = st.slider('changepoint', 0.00, 1.00, 0.001)
            n = st.slider('n_changepoint', 0, 20, 1)
            fourier = st.slider('Fourier', 0, 24, 1)
            df_P = md.PPhet(df,growth,seasonality,changepoint,n,fourier,select_type) 
        else:
            df_P = md.PPhet(df)
            
    if 'XGBoosting' in model:
        if select_type == 'Manual':
            learning = st.slider('learning_rate', 0.000, 1.000, 0.001)
            maxdep = st.slider('max_dept', 0, 50, 3)
            n = st.slider('n_estimator', 1, 150, 10)
            maxlea = st.slider('max_leaves', 2, 10, 2)
            tree = st.select_slider(
                'Tree_method',
                options=['hist','gpu_hist','exact'])

            df_XGB = md.ML_FC(df,model='XGB',select_type='Auto',learning_rate=learning,max_depth=maxdep,n_estimators=n,tree_method=tree,max_leaves=maxlea)         
        else:
            df_XGB = md.ML_FC(df,model='XGB',select_type='Auto')
    if 'LightGBM' in model:
        if select_type == 'Manual':
            learning = st.slider('learning_rate', 0.000, 1.000, 0.010)
            maxdep = st.slider('max_dept', 1, 10, 3)
            n = st.slider('n_estimator', 1, 150, 10)
            maxlea = st.slider('max_leaves', 2, 10, 2)
            mingaintosplit = st.slider('min_gain_ro_split', 1, 10, 2)

            df_LGBM = md.ML_FC(df,model='LGBM',select_type='Auto',learning_rate=learning,max_depth=maxdep,n_estimators=n,min_gain_to_split=mingaintosplit,max_leaves=maxlea)        
        else:
            df_LGBM = md.ML_FC(df,model='LGBM',select_type='Auto')
            
            
    df['Model'] = 'Actual'
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df = pd.concat([df,df_baseline,df_HW,df_SARIMAX,df_UCM,df_P, df_XGB,df_LGBM])


with col2:
    fig, ax = plt.subplots(figsize=(8,4))
    sns.set_theme(style="white")
    if plot_type == 'trend':
        ax = sns.lineplot(data=df,x=df.index,
                          y=df[sku],hue='Model',
                          style='Model',palette='tab10',markers='o')
        ax.grid(False)

    elif plot_type == 'multipleline':
        ax = sns.barplot(data=df.iloc[:,:][df.Model == 'Actual'],
                     x='Month',
                     y=sku,
                     hue='Year',
                     palette='Blues'
                     )

        ax2 = ax.twiny()  
        ax2 = sns.lineplot(data=df.iloc[:,:][df.Model != 'Actual'],
                     x='Month',
                     y=sku,
                     hue='Year',
                     style='Model',palette=sns.dark_palette('red'))
    st.pyplot(fig)



fig1, ax1 = plt.subplots(figsize=(16,8))
month_plot(df.iloc[:,:][df.Model == 'Actual'][sku],ax=ax1)
st.pyplot(fig1)
    
    

st.subheader("Filtered data will appear below 👇 ")
st.text("")
df = df.drop(['Month','Year'], axis=1)

st.table(df)

st.text("")



import xlsxwriter
from io import BytesIO
buffer = BytesIO()

with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    # Write each dataframe to a different worksheet.
    df.to_excel(writer, sheet_name='Sheet1')

    # Close the Pandas Excel writer and output the Excel file to the buffer
    writer.save()

    st.download_button(
        label="Download Excel worksheets",
        data=buffer,
        file_name="forecast.xlsx",
        mime="application/vnd.ms-excel"
    )
    
#st.markdown('If you need support, reach out to PhamGiaPhu@duytan.com or call ☎:240')

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
    uploaded_file = st.file_uploader("Choose a Excel file")
    
    
    #####
    footer.footer()
    
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
                👆 Upload your .xlsx file to make forecast. Here's a sample file: [Actual Sales](https://duytan-my.sharepoint.com/:x:/g/personal/phamgiaphu_duytan_com1/EYe1ArKWaulDhLa1G9mPrnMB7C3G_F_mkvJ-7c93u6c9kw?e=j3HVCj)
                """)
        
        
    else:
        st.info(
            f"""
                👆 Upload your .xlsx file to make forecast. Here's a sample file: [Actual Sales](https://duytan-my.sharepoint.com/:x:/g/personal/phamgiaphu_duytan_com1/EYe1ArKWaulDhLa1G9mPrnMB7C3G_F_mkvJ-7c93u6c9kw?e=j3HVCj)
                """)
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
            ['UCM', 'SARIMAX', 'Prophet', 'Holt-Winter'],
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


if 'SARIMAX' in model:
    df_SARIMAX = md.SARIMAX(df)
    #df = df.merge(df_SARIMAX,left_index=True,right_index=True,how='outer',indicator=True)
if 'UCM' in model:
    df_UCM = md.UCM(df)
    #df = df.merge(df_UCM,left_index=True,right_index=True,how='outer',indicator=True)
df['Model'] = 'Actual'
df_baseline['Model'] = 'Baseline'

#df.drop(['_merge'],axis=1,inplace=True)

plot_type = 'trend'

df['Month'] = df.index.month
df['Year'] = df.index.year

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
    
    select_type =  st.radio(
                "Choose your forecast model 👇",
                ['Auto', 'Manual'],
                key="selecttype",
                label_visibility='collapsed',
                disabled=False,
                horizontal=True)
 
    if 'Holt-Winter' in model:
        if select_type == 'Manual':
            alpha = st.slider('alpha', 0.00, 1.00, 0.25)
            beta = st.slider('beta', 0.00, 1.00, 0.25)
            gamma = st.slider('gamma', 0.00, 1.00, 0.25)
            df_HW = md.HoltWinter(df,alpha,beta,gamma)
        else:
            df_HW = md.HoltWinter(df)
df = pd.concat([df,df_baseline,df_HW,df_SARIMAX,df_UCM])  

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

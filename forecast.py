import streamlit as st
import pandas as pd
from prophet import Prophet
import pmdarima as pmd
import statsmodels.api as sm
from datetime import datetime, date
import matplotlib.pyplot as plt
import seaborn as sns

###################################
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode
from st_aggrid import GridUpdateMode, DataReturnMode

###################################
import model as md
from functionforDownloadButtons import download_button
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
        shows = pd.read_excel(uploaded_file, sheet_name = "Full History")
        shows = shows.fillna(0)
        df_forecast = pd.read_excel(uploaded_file, sheet_name = "Forecast")
        
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

gb = GridOptionsBuilder.from_dataframe(shows)
#gb.configure_default_column(shows2.columns[0])
gb.configure_column(shows.columns[0], rowGroup=True)
###

#gb.configure_column(shows2['Material'],headerCheckboxSelection=False)
gb.configure_selection(selection_mode="single",use_checkbox=True)

#gb.configure_side_bar()
gb.configure_columns(shows.columns.values.tolist(),headerCheckboxSelection=False, editable=True)
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
    shows,
    gridOptions=gridOptions,
    allow_unsafe_jscode=True, reload_data=False,
    enable_enterprise_modules=True,
    update_mode=GridUpdateMode.MODEL_CHANGED,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    fit_columns_on_grid_load=True,
)




#########################################


st.subheader('3. Forecast')

model = st.multiselect(
'Choose your forecast model',
['UCM', 'SARIMAX', 'Prophet', 'Holt-Winter'],
['Holt-Winter'])


df = pd.DataFrame(response["selected_rows"])
df = df.drop(['Group','Description'], axis=1)
df = pd.melt(df,id_vars=df.columns[0])
df.rename({'variable': 'Date'}, axis=1, inplace=True)
df['Date'] = df['Date'].apply(lambda x: datetime.strptime("{}".format(x),"%d-%m-%Y").date())
st.write(df.iloc[0,0])
df = pd.DataFrame(df.pivot('Date','Material','value'))
df.index = pd.to_datetime(df.index)
#df = df.apply(pd.to_numeric)

df_HW = pd.DataFrame()
df_SARIMAX = pd.DataFrame()
df_UCM = pd.DataFrame()
if 'Holt-Winter' in model:
    df_HW = md.HoltWinter(df)
    #df = df.merge(df_HW,left_index=True,right_index=True,how='outer',indicator=True)
if 'SARIMAX' in model:
    df_SARIMAX = md.SARIMAX(df)
    #df = df.merge(df_SARIMAX,left_index=True,right_index=True,how='outer',indicator=True)
if 'UCM' in model:
    df_UCM = md.UCM(df)
    #df = df.merge(df_UCM,left_index=True,right_index=True,how='outer',indicator=True)
df['Model'] = 'Actual'
df = pd.concat([df,df_HW,df_SARIMAX,df_UCM])
st.write(df)
#df.drop(['_merge'],axis=1,inplace=True)


#df.sort_values(by=['Material','Date'],inplace=True)
dfplot = df.copy().reset_index()
st.write(dfplot)
fig = plt.figure(figsize=(16, 8))
sns.lineplot(data=dfplot,x="Date", y=dfplot.columns[1],hue='Model')
st.pyplot(fig)
    
    

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

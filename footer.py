import streamlit as st
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb


def layout(*args):
    style = """
        <style>
          # MainMenu {visibility: hidden;}
          footer {visibility: hidden;}
         .stApp { bottom: 3px; }
        </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(2, 2, 2, 2),
        width=percent(100),
        color="black",
        text_align="left",
        height=3,
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(2, 2, 2, 2),
        border_style="none",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)



def footer():
    myargs = ["If you need support,",
                br(),
                    "reach out to PhamGiaPhu@duytan.com,",  
                br(),
                    "or call ☎:240"]
    layout(*myargs)
   

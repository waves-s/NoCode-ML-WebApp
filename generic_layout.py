import streamlit as st

def generic_main():
    
    st.set_page_config(layout="wide")
    st.write('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
    
    
def generic_footer():
    
    # Add a footer
    st.markdown("""
        <hr style="border:1px solid #f00">
       <p style="font-size: 12px; text-align: center;">
            Disclaimer: This web application is provided as-is, without warranty of any kind. 
            Use of this web application is at your own risk. The developers of this web application 
            take no liability for any direct or indirect damage resulting from its use.<br><br>
            Privacy Statement: No user data is stored or used for any purposes beyond the immediate 
            functionality of this web application. Your privacy and data security are important to us.<br><br>
            This tool is free for individual use. For commercial use, please obtain a license by 
            contacting us. We value your feedback! Share your comments, feedback, and suggestions with us: 
            <a href="https://www.linkedin.com/in/vibha-dhawan/" target="_blank">Connect via LinkedIn</a>.
        </p>
        """, unsafe_allow_html=True)
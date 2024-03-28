import streamlit as st


def generic_main():
    
    st.set_page_config(layout="wide")
    # st.write('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
    
    # Load HTML template
    html_code = load_html()
    # st.header(html_code)
    st.components.v1.html(html_code)


    # Display the Provide Feedback, Comments link   
    st.markdown(
        """
        <div style="position: absolute; top: 10px; right: 10px;">
            <a href="https://www.linkedin.com/in/vibha-dhawan/" target="_blank" style="text-decoration: none; color: #FFDD00; font-weight: bold; font-size: 16px;">
                Provide Feedback, Comments & Improvement Ideas
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
        
    st.markdown(
    """
    <div style="position: fixed; top: 10px; right: 10px;">
        <a href="https://www.linkedin.com/in/vibha-dhawan/" target="_blank">Provide Feedback, Comments & Improvement Ideas</a>
    </div>
    """,
    unsafe_allow_html=True
    )
    
def load_html():
    with open("hotjar_template.html", "r") as file:
        html_code = file.read()
    return html_code
    
def generic_footer():
    
    # Add a footer
    st.markdown("""
        <hr style="border:1px solid #f00">
       <p style="font-size: 12px; text-align: center;">
            Disclaimer: This web application is provided as-is, without warranty of any kind. Use of this web application is at your own risk. The developers of this web application take no liability for any direct or indirect damage resulting from its use.<br><br>
            This tool is free for individual use. For commercial use, please obtain a license by contacting us. We value your feedback! Share your comments, feedback, and suggestions with us: 
            <a href="https://www.linkedin.com/in/vibha-dhawan/" target="_blank">Connect via LinkedIn</a>.<br><br>
            Copyright © 2024 The Creator. All rights reserved.
        </p>
        """, unsafe_allow_html=True)
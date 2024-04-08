import streamlit as st
from streamlit_javascript import st_javascript

def generic_main():
    
    st.set_page_config(layout="wide")
    st.write('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)
    js_code = """
                window._mfq = window._mfq || [];
                (function() {
                    var mf = document.createElement("script");
                    mf.type = "text/javascript"; mf.defer = true;
                    mf.src = "//cdn.mouseflow.com/projects/59584ea9-74b4-4f12-b183-d2ca1af1a9f4.js";
                    document.getElementsByTagName("head")[0].appendChild(mf);
                })(); """
    info = st_javascript(js_code)
    # st.write(info)
    st.markdown(f"""<script> {info} </script>""", unsafe_allow_html=True)
    # Load HTML template
    # html_code = load_html()
    # # st.header(html_code)
    # st.components.v1.html(html_code)
    # st.markdown("""
    #             <script>
    #             window._mfq = window._mfq || [];
    #             (function() {
    #                 var mf = document.createElement("script");
    #                 mf.type = "text/javascript"; mf.defer = true;
    #                 mf.src = "//cdn.mouseflow.com/projects/59584ea9-74b4-4f12-b183-d2ca1af1a9f4.js";
    #                 document.getElementsByTagName("head")[0].appendChild(mf);
    #             })(); </script>
    # """, unsafe_allow_html=True)


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
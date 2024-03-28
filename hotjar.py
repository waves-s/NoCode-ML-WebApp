def hotjar_tracking_code():
    """Insert your Hotjar Tracking Code here."""
    script = """
    <!-- Hotjar Tracking Code -->
    <script>
        (function(h,o,t,j,a,r){
            h.hj=h.hj||function(){(h.hj.q=h.hj.q||[]).push(arguments)};
            h._hjSettings={hjid:3893568,hjsv:6};
            a=o.getElementsByTagName('head')[0];
            r=o.createElement('script');r.async=1;
            r.src=t+h._hjSettings.hjid+j+h._hjSettings.hjsv;
            a.appendChild(r);
        })(window,document,'https://static.hotjar.com/c/hotjar-','.js?sv=');
    </script>
    <!-- End of Hotjar Tracking Code -->
    """

    return script



# <!DOCTYPE html>
# <html>
# <head>
#     <!-- Hotjar Tracking Code -->
#     <script>
#         (function(h,o,t,j,a,r){
#             h.hj=h.hj||function(){(h.hj.q=h.hj.q||[]).push(arguments)};
#             h._hjSettings={hjid:3893568,hjsv:6};
#             a=o.getElementsByTagName('head')[0];
#             r=o.createElement('script');r.async=1;
#             r.src=t+h._hjSettings.hjid+j+h._hjSettings.hjsv;
#             a.appendChild(r);
#         })(window,document,'https://static.hotjar.com/c/hotjar-','.js?sv=');
#     </script>
# </head>
# <body>
#     <!-- Streamlit App Placeholder -->
#     <div id="app_container"></div>

#     <!-- Streamlit App Script -->
#     <script src="https://cdn.jsdelivr.net/npm/@streamlit/streamlit@latest"></script>
#     <script>
#         const container = document.getElementById("app_container");
#         Streamlit.setComponentReady();
#         Streamlit.render(container);
#     </script>
# </body>
# </html>
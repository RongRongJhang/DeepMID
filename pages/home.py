import streamlit as st

deepmid_page = st.Page("deepmid.py", title="DeepMID", icon=":material/medical_services:")
deepvision_page = st.Page("deepvision.py", title="DeepVision", icon=":material/imagesmode:")

pg = st.navigation([deepmid_page, deepvision_page])
pg.run()
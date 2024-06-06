import streamlit as st

# Set the title of the app
st.title("Tornado Prediction and Analysis")

# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Data Visualization", "Model Training"])

if page == "Home":
    # Introduction Page Content
    st.subheader("Predicting tornado events using historical weather data")

    # Project Overview with Image
    st.markdown("## Project Overview")
    st.image("assets/7 Tornadoes at Once.png", caption="Multiple Tornadoes in Action")
    st.markdown("""
    Welcome to the Tornado Prediction App. This tool aims to provide insights and predictions for tornado events using historical weather data from NOAA. Our goal is to leverage historical data to understand and predict tornado occurrences, ultimately helping to mitigate their impact.
    """)

    # Why Tornadoes
    st.markdown("## Why Tornadoes?")
    st.image("assets/TORNADO IMAGE.png.webp", caption="Devastating Power of Tornadoes")
    st.markdown("""
    Tornadoes are among the most destructive natural phenomena, causing significant damage and loss of life. Accurate prediction of tornadoes can help in issuing timely warnings, thereby saving lives and reducing property damage. By analyzing historical weather patterns, we aim to develop models that can predict the likelihood of tornado events.
    """)

    # Why the United States
    st.markdown("## Why the United States?")
    st.image("assets/Tornado Alley.svg.png", caption="Map of Tornado Alley")
    st.markdown("""
    The United States experiences the highest number of tornadoes in the world, with a significant concentration in a region known as Tornado Alley, which includes parts of Texas, Oklahoma, Kansas, and Nebraska. This high frequency makes the U.S. an ideal focus for tornado prediction research.
    """)

    # Background Information with Image
    st.markdown("## Background Information")
    st.image("assets/Map-frequency-tornadoes-range-Texas-Nebraska-Tornado.webp", caption="Tornado Frequency in the U.S.")
    st.markdown("""
    The U.S. witnesses more tornadoes than any other country, largely due to its unique geography. Tornado Alley, in particular, sees a high frequency of tornadoes because of the convergence of warm, moist air from the Gulf of Mexico and cool, dry air from Canada. Understanding and predicting these events can significantly reduce their devastating impacts.
    """)

    # Methodology with Image
    st.markdown("## Methodology")
    st.image("assets/Doppler-radar-NWS-Tampa.jpeg", caption="Doppler Radar System")
    st.markdown("""
    We collect various weather data types, including temperature, wind, and precipitation, from NOAA's historical records. This data, spanning from 1950 to 2022, is then processed and used to train predictive models. Our dataset has been pre-cleaned and pre-treated to ensure accuracy and reliability. Users will not have the option to upload data, as we use our own pre-cleaned data.
    """)

    # Features of the App
    st.markdown("## Features of the App")
    st.image("assets/Blue Thunder.jpg", caption="App Interface Screenshot")
    st.markdown("""
    - **Interactive visualizations** of historical weather data.
    - **Predictive insights** on potential tornado events.
    - **Easy-to-use interface** for exploring weather patterns.
    """)

    # Instructions for Use
    st.markdown("## Instructions for Use")
    st.markdown("""
    Navigate through the app using the sidebar. Input your data or select options to explore visualizations and predictions. Follow the guidance provided to interpret results.
    """)

    # Acknowledgments
    st.markdown("## Acknowledgments")
    st.image("assets/Image of a house with a scary storm on top.jpg", caption="Storm Image")
    st.markdown("""
    This app was developed with data provided by NOAA. We obtained historical tornado data from 1950 to 2022 and fetched additional data using the NOAA API. Special thanks to all contributors and collaborators.
    """)

    # Contact Information
    st.markdown("## Contact Information")
    st.image("assets/Tornado in the middle of the ocean.jpg", caption="Tornado Over the Ocean")
    st.markdown("""
    For any questions or feedback, please contact AYOUB DJELLAOUI at [ayoubdjellaoui20@gmail.com](mailto:ayoubdjellaoui20@gmail.com).

    Connect with me on [LinkedIn](https://www.linkedin.com/in/ayoub-djellaoui-649a37289/) and check out my projects on [GitHub](https://github.com/AyoubDev8).

    You can view my [Resume](https://drive.google.com/file/d/1F8wiA4PthH2X7VQCZyGgkJ_cgxgFaqNO/view?usp=sharing).
    """)


elif page == "Data Visualization":
    import pages.data_visualization
elif page == "Model Training":
    import pages.model_training

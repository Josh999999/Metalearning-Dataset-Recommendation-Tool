#!------------------IMPORTS------------------!
import streamlit as st
from streamlit_utils import IMAGES
from database_view import st_database_view
from manual_view import st_manual_view
from view_utils import get_database_datasets








#!------------------FUNCTIONALITY------------------!




#!------------------TOP LEVEL EXECUTION------------------!

# Top level execution - retrieve Database information (Datasets with data and algorithms) - used in the Database page (st_database_view)
datasets_dict, algorithms_dict = get_database_datasets()




#!------------------MAIN APPLICATION------------------!


# Run the main visual / functional component
if __name__ == "__main__":

    # Top level execution NOTE: Streamlit *REQUIRES* this to be the first executed st. command to make the page tab/favicon work.
    st.set_page_config(page_title="Meta-learning Search Engine", page_icon=str(IMAGES / "BU.png"))


    # Load file uploader keys for the uploaders that take Datasets to be inserted into the Database into the session for later manipulation
    if "insert_comparison_file_uploader_key" not in st.session_state:
        st.session_state["insert_comparison_file_uploader_key"] = 1
    

    if "insert_performance_metrics_file_uploader_key" not in st.session_state:
        st.session_state["insert_performance_metrics_file_uploader_key"] = 0


    st.title(":material/search_insights: Meta-learning Search Engine")

    st.write("Search for which Dataset is the most statistically similar to your chosen Dataset - allowing you to decide which group of machine learning algorithms are most likely to be effective for your task.")
    
    # Divider block
    st.divider()
    st.html("<br><br>")

    st.subheader(":material/upload: Upload Data & Select Task Type")

    st.sidebar.title("Select page")

    # Selectbox for pages
    page = st.sidebar.selectbox("", ["Manual selection", "Database selection"])
    

    if page == "Manual selection":
        st_manual_view()

    elif page == "Database selection":
        st_database_view(datasets_dict, algorithms_dict)
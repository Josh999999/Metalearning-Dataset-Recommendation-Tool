#!------------------IMPORTS------------------!
import streamlit as st
from pca import *
from streamlit_utils import get_stored_datasets
from pathlib import Path
from database import INSERT_dataset_with_algorithms_safe
from view_utils import get_database_datasets
from main_comparison import st_main_comparison
from streamlit_modal import Modal







#!------------------FUNCTIONALITY------------------!


""" Function that creates and runs the page for the Database Comparison functionality"""
def st_database_view(
        datasets_dict: dict,
        algorithms_dict: dict
    ):
    """
    @param datasets_dict - Dictionary containing Datasets retrieved from the Database in key-value pairs
    @param algorithms_dict - Dictionary containing Sets of Algorithms retrieved from the Database in key-value pairs (where the key is {dataset_name}_performance_metrics)
    """

    tab_upload_reference, tab_upload_comparison = st.tabs(["Compare Reference Dataset", "Upload Comparison Dataset + Performance Information into Database"])




    #!------------------COMPARISON SECTION------------------!
    
    # Set global variables (used in multiple sections of the web app)
    reference_dataframes_list, comparison_dataframes_list, performance_metric_dataframes_list, database_task_type_selection = None, datasets_dict, algorithms_dict, None
    
    reference_dataframe, reference_dataset_name = None, None

    upload_reference_error_flag, upload_reference_error_message = False, None


    # TAB - Here contains all content to do with the uploading of the Reference Dataset
    with tab_upload_reference:   
        st.write("Upload a `Reference Dataset`, then selecting your `Task Type` to begin searching.") 

        tab_upload_reference_dataset, tab_task_type_selection = st.tabs([":material/counter_1: Upload Independent Dataset", ":material/counter_2: Select Task Type"])

        
        # TAB - Here we upload the Reference Dataset (singular)
        with tab_upload_reference_dataset:
            column_upload_reference_dataset, column_display_reference_dataset = st.columns(2)


            # COLUMN - File upload for the Reference Dataset (singular)
            with column_upload_reference_dataset:
                reference_dataset = st.file_uploader("Choose a file", type=["csv", "json"], accept_multiple_files=False, key="reference_dataset")


                if reference_dataset:
                        

                    try:
                        
                        # Load Reference Dataset into a DataFrame
                        reference_dataframes_list = get_stored_datasets(datasets_path=[reference_dataset], is_file_obj=False)
                    
                    except Exception as Error:

                        # Flag any errors for output (such as incorrect JSON format, etc)     
                        upload_reference_error_flag = True
                        upload_reference_error_message = Error
                    
                    else:
                        upload_reference_error_flag = False


            # COLUMN - Display for the uploaded for the Reference Dataset (singular)
            with column_display_reference_dataset:
                    

                if reference_dataframes_list:
                    st.caption(":material/info: Dataset previews show the first 5 entries of the Dataset")                

                    # Displaying the Reference Dataset (shows the first 5 rows)
                    reference_dataset_name, reference_dataframe = next(iter(reference_dataframes_list.items()))


                    with st.expander(f"{reference_dataset_name} Preview"):
                        st.dataframe(reference_dataframe.head(5), hide_index=True)


        # TAB - Here we set the task type for which an algorithm is required to run on the Reference Dataset (singular)
        with tab_task_type_selection:
            database_task_type_selection = st.selectbox("Select which type of machine learning task you want to search for.", ["None", "Classification", "Regression"], key="manual_task_type_selection")



        
        #!------------------COMPARISON FILE UPLOAD ERRORS------------------!
        if upload_reference_error_flag:
            st.warning(f"There was an issue uploading the uploaded Reference Dataset: {upload_reference_error_message}")




        #!------------------COMPARISON SECTION ERRORS------------------!
        # Check that Database has Datasets to compare against
        elif comparison_dataframes_list == None or len(comparison_dataframes_list) < 1:
            st.info("The Database doesn't contain any Datasets to compare against")


        # Check that the Reference Dataframes have been uploaded and task type selected
        elif not reference_dataframes_list or database_task_type_selection == "None":       
            conditions = [
                ["&nbsp; :material/counter_1: Upload Reference Dataset", reference_dataframes_list != None],
                ["&nbsp; :material/counter_2: Select Task Type", database_task_type_selection != "None"],
            ] 

            st.info("Submit data for each tab to proceed with the Dataset search.\n\n" + 
                    "\n\n".join([f":material/check: {entry[0]}" if entry[1] == True else f":material/close: {entry[0]}" for entry in conditions]), icon=":material/search:")


        # Check that the Reference Dataset has data inside it
        elif reference_dataframe.empty:            
            st.warning(f"The Reference Dataset: {reference_dataset_name} doesn't contain any data to be compared against")


        

        #!------------------COMPARISON SECTION------------------!
        else:

            #!------------------MAIN COMPARISON SECTION------------------!
            st_main_comparison(performance_metric_dataframes_list, reference_dataframes_list, comparison_dataframes_list, database_task_type_selection)








    #!------------------INSERT SECTION------------------!

    # Set global variables (used in multiple sections of the web app)
    insert_comparison_dataframes_list, insert_performance_metric_dataframes_list, performance_metrics_datasets_uploaded = None, None, None
        
    error_file_name, insert_comparison_dataframes_names_list, insert_performance_metrics_dataframes_names_list = None, [], []

    upload_comparison_error_flag, upload_comparison_error_message = False, None

    upload_performance_metrics_error_flag, upload_performance_metrics_error_message = False, None

    comparison_has_data, performance_has_data = True, True

    is_inserted = False


    # TAB - Here contains all content to do with the uploading Comparison Datasets and Performance Metrics
    with tab_upload_comparison:      
        st.write("Upload A series of `Comparison Datasets` and each corresponding Comparison Datasets `Performance Metrics` file - then click insert to upload to the Database")

        column_upload_comparison, column_insert_button = st.columns([5, 0.6])


        # COLUMN - Here contains all content to do with the uploading Comparison Datasets and Performance Metrics (without the insert buttons section)
        with column_upload_comparison:
            tab_upload_comparison_datasets, tab_upload_performance_metrics = st.tabs([":material/counter_1: Upload System Datasets", ":material/counter_2: Upload Performance Metrics"])


            # TAB - Here we upload the Comparison Datasets (multiple)
            with tab_upload_comparison_datasets:
                column_upload_comparison_datasets, column_display_comparison_datasets = st.columns(2)


                # COLUMN - File upload for the Comparison Datasets (multiple)
                with column_upload_comparison_datasets:                    
                    insert_comparison_datasets_list = st.file_uploader("Choose a file", type=["csv", "json"], accept_multiple_files=True, key=st.session_state["insert_comparison_file_uploader_key"])


                    if insert_comparison_datasets_list:


                        try:   
                                   
                            # Load Comparison Datasets into a DataFrame                      
                            insert_comparison_dataframes_list = get_stored_datasets(datasets_path=insert_comparison_datasets_list, is_file_obj=False)
                        
                        except Exception as Error: 

                            # Flag any errors for output (such as incorrect JSON format, etc)      
                            upload_comparison_error_flag = True
                            upload_comparison_error_message = Error
                        
                        else:
                            upload_comparison_error_flag = False


                # COLUMN - Display for the uploaded Comparison Datasets (multiple)
                with column_display_comparison_datasets:


                    if insert_comparison_dataframes_list:
                        st.caption(":material/info: Dataset previews show the first 5 entries of each Dataset")
                        

                        # Displaying the Reference Dataset (shows the first 5 rows)
                        for comparison_name, comparison_dataframe in insert_comparison_dataframes_list.items():


                            with st.expander(f"{comparison_name} Preview"):
                                st.dataframe(comparison_dataframe.head(5))


            # TAB - Here we upload the Performance Metrics Datasets (multiple)
            with tab_upload_performance_metrics:
                column_upload_performance_metrics_datasets, column_display_performance_metrics_datasets = st.columns(2)


                # COLUMN - File upload for the Performance Metrics Datasets (multiple)
                with column_upload_performance_metrics_datasets:                    
                    insert_performance_metric_datasets_list = st.file_uploader("Choose a file", type=["csv", "json"], accept_multiple_files=True, key=st.session_state["insert_performance_metrics_file_uploader_key"])


                    if insert_performance_metric_datasets_list:
                        

                        try:

                            # Load Performance Metrics Datasets into a DataFrame      
                            insert_performance_metric_dataframes_list = get_stored_datasets(datasets_path=insert_performance_metric_datasets_list, is_file_obj=False)
                        
                        except Exception as Error: 
                    
                            # Flag any errors for output (such as incorrect JSON format, etc)         
                            upload_performance_metrics_error_flag = True
                            upload_performance_metrics_error_message = Error
                        
                        else:
                            upload_performance_metrics_error_flag = False


                # COLUMN - Display for the uploaded Performance Metrics Datasets (multiple)
                with column_display_performance_metrics_datasets:


                    if insert_performance_metric_dataframes_list:
                        st.caption(":material/info: Dataset previews show the first 5 entries of each Dataset")


                        # Displaying the Reference Dataset (shows the first 5 rows)
                        for performance_name, performance_dataframe in insert_performance_metric_dataframes_list.items():


                            with st.expander(f"{performance_name} Preview"):
                                st.dataframe(performance_dataframe.head(5), hide_index=True)


        # COLUMN - Here contains all content to do with the inserting of the given Comparison Datasets and Performance Metrics Datasets
        with column_insert_button:


            if st.button('Insert'):
                is_inserted = True
                

                if insert_performance_metric_dataframes_list and insert_comparison_dataframes_list:               

                    # Check each Comparison Dataset has data inside it
                    comparison_has_data = performance_has_data = True      
                    

                    for comparison_name, comparison_dataframe in insert_comparison_dataframes_list.items():


                        if comparison_dataframe.empty:
                            comparison_has_data = False
                            error_file_name = comparison_name


                    # Check each Performance Metrics file has data inside it
                    for performance_name, performance_dataframe in insert_performance_metric_dataframes_list.items():


                        if performance_dataframe.empty:
                            performance_has_data = False
                            error_file_name = performance_name

                    
                    # Check each Comparison Dataset has a Performance Metrics file and vice-versa
                    insert_comparison_dataframes_names_list = list(insert_comparison_dataframes_list.keys())
                    insert_performance_metrics_dataframes_names_list = list(insert_performance_metric_dataframes_list.keys())


                    # Check DataFrame file inputs for warning sanitisation
                    for dataframe_name in insert_comparison_dataframes_names_list.copy():


                        if f"{dataframe_name}_perform_metrics" in insert_performance_metrics_dataframes_names_list:
                            insert_comparison_dataframes_names_list.remove(dataframe_name)
                            insert_performance_metrics_dataframes_names_list.remove(f"{dataframe_name}_perform_metrics") 


                    # If all Datasets are accoumted for insert the Datasets into the Database
                    if (
                        not insert_comparison_dataframes_names_list or 
                        not insert_performance_metrics_dataframes_names_list or
                        comparison_has_data or
                        performance_has_data
                    ): 


                        for comparison_name, comparison_dataframe in insert_comparison_dataframes_list.items():
                            _, performance_dataframe = next(iter(insert_performance_metric_dataframes_list.items()))


                            try:
                                # Run INSERT query
                                INSERT_dataset_with_algorithms_safe(comparison_name, comparison_dataframe, performance_dataframe)
                                        
                                comparison_dataframes_list, performance_metric_dataframes_list = get_database_datasets()

                                insert_comparison_dataframes_list, insert_performance_metric_dataframes_list, performance_metrics_datasets_uploaded = None, None, None

                                error_file_name, insert_comparison_dataframes_names_list, insert_performance_metrics_dataframes_names_list = None, [], []

                                st.rerun()
                            
                            except Exception as Error:
                                is_inserted = False                                
                                insert_error_popup = Modal(key="Demo Key",title=f"Dataset: {comparison_name} couldn't be inserted into the Database")


                                with insert_error_popup.container():
                                    st.write(str(Error))


                                break

                
                        st.session_state["insert_comparison_file_uploader_key"] += 1

                        st.session_state["insert_performance_metrics_file_uploader_key"] -= 1

                        


        #!------------------COMPARISON FILE UPLOAD ERRORS------------------!
        if upload_comparison_error_flag:
            st.warning(f"There was an issue loading the uploaded Comparison Datasets: {upload_comparison_error_message}")


        elif upload_performance_metrics_error_flag:
            st.warning(f"There was an issue loading the uploaded Performance Metrics Dataset: {upload_performance_metrics_error_message}")



        #!------------------INSERT SECTION ERRORS------------------!
        # Check all Performance Metrics Datasets are accounted for
        elif insert_comparison_dataframes_names_list:
            comparison_dataframes_names_break = '\n' if insert_comparison_dataframes_names_list else ''

            st.info("The following Datasets do not have a clear paired Comparison/Performance Metrics file:\n\n" + '**Comparison Datasets**:\n' + '\n'.join([f'- {name}' for name in insert_comparison_dataframes_names_list]) + comparison_dataframes_names_break + '\nThe following Datasets must have a clear pair before moving forward. \n\n **Performance Metrics**:\n' + '\n'.join([f'- {name}_perform_metrics' for name in insert_comparison_dataframes_names_list]))


        # Check all Comparison Datasets are accounted for
        elif insert_performance_metrics_dataframes_names_list:
            
            # Check all Performance Metrics Dataframes are accounted for
            performance_metrics_datasets_uploaded = "\n".join([f"\n:material/check: &nbsp; {name}_perform_metrics" if f"{name}_perform_metrics" in [Path(file.name).stem for file in insert_performance_metric_dataframes_list.keys()] else f"\n:material/close: &nbsp; {name}_perform_metrics" for name in insert_comparison_dataframes_list.keys()])

            st.warning(f"Model Performance Metrics Datasets are required for each uploaded Comparison Dataset. Each Dataset's file name must be appended with '`_perform_metrics`' to be correctly identified. Expected file name(s) are shown below:\n\n{performance_metrics_datasets_uploaded}")


        # Check all Comparison Datasets contain data
        elif not comparison_has_data:
            st.warning(f"The Comparison Dataset: {error_file_name} doesn't contain any data to be compared against")


        # Check all Performance Metrics Datasets contain data
        elif not performance_has_data:
            st.warning(f"The Performance Metrics file: {error_file_name} doesn't contain any data to be compared against")


        # Check if the Comparison Datasets have been uploaded for the insertion
        elif not insert_comparison_dataframes_list:
            st.warning(f"There are no Comparison Datasets that have been uploaded for the insertion") 


        # Check if the Performance Metrics Datasets have been uploaded for the insertion
        elif not insert_performance_metric_dataframes_list:
            st.warning(f"There are no Performance Metrics Datasets that have been uploaded for the insertion") 
        

        # Otherwise the Datasets have been inserted successfully into the Database
        elif is_inserted:
            st.info("The Datasets have been inserted into the Database")
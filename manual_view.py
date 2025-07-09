#!------------------IMPORTS------------------!
import pandas as pd
import streamlit as st
import json
from streamlit_utils import get_stored_datasets, EXAMPLES
from pathlib import Path
from database import check_algorithm
from main_comparison import st_main_comparison








#!------------------FUNCTIONALITY------------------!


"""Function that creates and runs the page for the manual Comparison functionality"""
def st_manual_view():

    # Set global variables (used in multiple sections of the web app)
    reference_dataframes_list, comparison_dataframes_list, performance_metric_dataframes_list, manual_task_type_selection,  =  None, None, None, None
    
    upload_reference_error_flag, upload_reference_error_message = False, None

    upload_comparison_error_flag, upload_comparison_error_message = False, None

    upload_performance_metrics_error_flag, upload_performance_metrics_error_message = False, None

    performance_metrics_datasets_uploaded_message = None

    st.write("Upload a `Reference Dataset`, a series of `Comparison Datasets` and each corresponding Comparison Datasets `Performance Metrics` file - then selecting your `Task Type` to begin searching.")

    tab_upload_reference_dataset, tab_upload_comparison_datasets, tab_upload_performance_metrics, tab_task_type = st.tabs([":material/counter_1: Upload Reference Dataset", ":material/counter_2: Upload Comparison Datasets", ":material/counter_3: Upload Performance Metrics", ":material/counter_4: Select Task Type"])


    # TAB - Here we upload the Reference Dataset (singular)
    with tab_upload_reference_dataset:
        column_upload_reference_dataset, column_display_reference_dataset = st.columns(2)


        # COLUMN - File upload for the Reference Dataset (singular)
        with column_upload_reference_dataset:
            reference_dataset = st.file_uploader("Upload a Reference Dataset file", type=["csv", "json"], accept_multiple_files=False, key="comparison_dataset")


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
                st.caption(":material/info: Dataset previews show the first 5 entries of the data file")                

                # Displaying the Reference Dataset (shows the first 5 rows)
                reference_dataset_name, reference_dataframe = next(iter(reference_dataframes_list.items()))


                with st.expander(f"{reference_dataset_name} Preview"):
                    st.dataframe(reference_dataframe.head(5), hide_index=True)


    # TAB - Here we upload the Comparison Datasets (multiple)
    with tab_upload_comparison_datasets:
        column_upload_comparison_datasets, column_display_comparison_datasets = st.columns(2)


        # COLUMN - File upload for the Comparison Datasets (multiple)
        with column_upload_comparison_datasets:
            comparison_datasets_list = st.file_uploader("Upload 1 or more Comparison Dataset file(s)", type=["csv", "json"], accept_multiple_files=True, key="comparison_datasets")

            
            if comparison_datasets_list:    


                try:            
                    
                    # Load Comparison Datasets into a DataFrame                
                    comparison_dataframes_list = get_stored_datasets(datasets_path=comparison_datasets_list, is_file_obj=False)
                
                except Exception as Error:     

                    # Flag any errors for output (such as incorrect JSON format, etc)  
                    upload_comparison_error_flag = True
                    upload_comparison_error_message = Error
                
                else:
                    upload_comparison_error_flag = False


        # COLUMN - Display for the uploaded for the Comparison Datasets (multiple)
        with column_display_comparison_datasets:


            if comparison_dataframes_list:
                st.caption(":material/info: Dataset previews show the first 5 entries of each data file")
                

                # Displaying the Reference Dataset (shows the first 5 rows)
                for comparison_dataset_name, comparison_df in comparison_dataframes_list.items():

                    
                    with st.expander(f"{comparison_dataset_name} Preview"):
                        st.dataframe(comparison_df.head(5), hide_index=True)


    # TAB - Here we upload the Performance Metrics Datasets (multiple)
    with tab_upload_performance_metrics:
        column_upload_performance_metrics_datasets, column_display_performance_metrics_datasets = st.columns(2)


        # COLUMN - File upload for the Performance Metrics Datasets (multiple)
        with column_upload_performance_metrics_datasets:
            performance_metric_datasets_list = st.file_uploader("Upload corresponding Performance Metric file(s) for each Comparison Dataset", type=["csv", "json"], accept_multiple_files=True, key="performance_metric_datasets")

            
            if performance_metric_datasets_list:    
                

                try:

                    # Load Performance Metrics Datasets into a DataFrame      
                    performance_metric_dataframes_list = get_stored_datasets(datasets_path=performance_metric_datasets_list, is_file_obj=False)
                
                except Exception as Error:
                    
                    # Flag any errors for output (such as incorrect JSON format, etc)       
                    upload_performance_metrics_error_flag = True
                    upload_performance_metrics_error_message = Error
                
                else:
                    upload_performance_metrics_error_flag = False   


        # ROW - Display showing the format the .json files and/or the .csv Datasets containing the Performance Metrics should be submitted
        with st.expander("How should I structure the files?", icon=":material/help:"):
            st.write("Click the respective button(s) below to see the expected format for each file type.")

            column_csv_example_file, column_json_example_file, _  = st.columns([1, 1, 5])
            

            # COLUMN - Display showing the format the .csv Dataset files containing the Performance Metrics should be submitted
            with column_csv_example_file:


                with st.popover(".csv"):
                    st.write("Typically it is best to open a spreadsheet software and save to the .csv file format.\n\nRequired fields: `Algorithm` & your chosen Performance Metrics fields.")
                    
                    st.dataframe(pd.read_csv(EXAMPLES / "example_csv.csv"), hide_index=True)


            # COLUMN - Display showing the format the .json Dataset files containing the Performance Metrics should be submitted
            with column_json_example_file:


                with st.popover(".json"):
                    st.write("Required fields: `Algorithm` & your chosen Performance Metricss fields.")


                    with open(EXAMPLES / "example_json.json", "r") as file:
                        st.json(json.load(file))


        # COLUMN - Display for the uploaded for the Performance Metrics Datasets (multiple)
        with column_display_performance_metrics_datasets:


            if performance_metric_dataframes_list:
                st.caption(":material/info: Dataset previews show the first 5 entries of each data file")


                # Displaying the Reference Dataset (shows the first 5 rows)
                for performance_metric_name, performance_metric_dataframe in performance_metric_dataframes_list.items():


                    with st.expander(f"{performance_metric_name} Preview"):
                        st.dataframe(performance_metric_dataframe.head(5), hide_index=True)


    # TAB - Here we set the task type for which an algorithm is required to run on the Reference Dataset (singular)
    with tab_task_type:
        manual_task_type_selection = st.selectbox("Select which type of machine learning task you want to search for.", ["None", "Classification", "Regression"], key="database_task_type_selection")



    
    #!------------------FILE UPLOAD ERRORS------------------!
    # Errors reported when the user is uploading Datasets for the Comparison operation

    # Upload Error for the Reference Dataset
    if upload_reference_error_flag:
        st.warning(f"There was an issue loading the uploaded Reference Dataset: {upload_reference_error_message}")


    # Upload Error for the Comparison Datasets
    elif upload_comparison_error_flag:
        st.warning(f"There was an issue loading the uploaded Comparison Datasets: {upload_comparison_error_message}")


    # Upload Error for the Performance Metrics Dataset
    elif upload_performance_metrics_error_flag:
        st.warning(f"There was an issue loading the uploaded Performance Metrics Dataset: {upload_performance_metrics_error_message}")




    #!------------------COMPARISON SECTION ERRORS------------------!
    # Displays the current uploaded Datasets and ready for Comparison
    elif not (reference_dataframes_list and comparison_dataframes_list and performance_metric_dataframes_list) or manual_task_type_selection == "None":
        conditions = [
            ["&nbsp; :material/counter_1: Upload Reference Dataset", reference_dataframes_list != None],
            ["&nbsp; :material/counter_2: Upload Comparison Datasets", comparison_dataframes_list != None],
            ["&nbsp; :material/counter_3: Upload Performance Metrics", performance_metric_dataframes_list != None],
            ["&nbsp; :material/counter_4: Select Task Type", manual_task_type_selection != "None"],
        ]

        st.info("Submit data for each tab to proceed with the Dataset search.\n\n" + 
                
                "\n\n".join([f":material/check: {entry[0]}" if entry[1] == True else f":material/close: {entry[0]}" for entry in conditions]), icon=":material/search:")




    #!------------------DATA TESTING SECTION------------------!
    # Performs tests and accesses the given Datasets / DataFrames 
    else:
        error_file_name = None
        reference_has_data = comparison_has_data = performance_metrics_has_data = True      

        # Check Reference Dataset has data inside it
        reference_dataset_name, reference_dataframe = next(iter(reference_dataframes_list.items()))


        if reference_dataframe.empty:
            reference_has_data = False
            error_file_name = reference_dataset_name
        

        # Check each Comparison Dataset has data inside it
        for comparison_dataset_name, comparison_dataframe in comparison_dataframes_list.items():


            if comparison_dataframe.empty:
                comparison_has_data = False
                error_file_name = comparison_dataset_name


                break


        # Check each Performance Metrics file has data inside it
        for performance_metrics_dataset_name, performance_metrics_dataframe in performance_metric_dataframes_list.items():


            if performance_metrics_dataframe.empty:
                performance_metrics_has_data = False
                error_file_name = performance_metrics_dataset_name


                break
        
        
        # Here we perform tests that can be run once we possess "all" the data
        algorithm_error_flag = False   


        # Check Performance file contains neccessary and valid information
        # Check each Performance Metrics file has data inside it
        for performance_metrics_dataset_name, performance_metrics_dataframe in performance_metric_dataframes_list.items():


            if performance_metrics_dataframe.empty:
                performance_metrics_has_data = False
                error_file_name = performance_metrics_dataset_name


                break

            else:


                try:
                    algorithms_list = json.loads(performance_metrics_dataframe.to_json(orient='records'))
                    

                    for algorithm_dict in algorithms_list:     
                        check_algorithm(algorithm_dict)

                except Exception as Error:
                    algorithm_error_message = f"{performance_metrics_dataset_name} file caused error: {Error}"
                    algorithm_error_flag = True

        
        # Check each Comparison Dataset has a Performance Metrics file and vice-versa
        comparison_dataframes_names_list = list(comparison_dataframes_list.keys())
        performance_dataframes_names_list = list(performance_metric_dataframes_list.keys())

        for comparison_dataset_name in comparison_dataframes_names_list.copy():


            if f"{comparison_dataset_name}_perform_metrics" in performance_dataframes_names_list:
                comparison_dataframes_names_list.remove(comparison_dataset_name)
                performance_dataframes_names_list.remove(f"{comparison_dataset_name}_perform_metrics")     
        



        #!------------------FURTHER COMPARISON SECTION ERRORS------------------!
        # Check the Reference file contains data
        if not reference_has_data:
            st.warning(f"The Reference Dataset: {error_file_name} doesn't contain any data")


        # Check all Comparison Datasets contain data
        elif not comparison_has_data:
            st.warning(f"The Comparison Dataset: {error_file_name} doesn't contain any data")


        # Check all Performance Datasets contain data
        elif not performance_metrics_has_data:
            st.warning(f"The Performance Metrics Dataset: {error_file_name} doesn't contain any data")


        # Check all Comparison Datasets are accounted for
        elif comparison_dataframes_names_list:
            comparison_dataframes_list_names_break = '\n' if comparison_dataframes_names_list else ''

            # Create visual of the missing Comparison Datasets
            st.info("The following Datasets do not have a clear Comparison Datasets - Performance Metrics Datasets pairing:\n\n" + 
                    
                    '**Comparison Datasets**:\n' + 

                    '\n'.join([f'- {name}' for name in comparison_dataframes_names_list]) + 

                    comparison_dataframes_list_names_break + 

                    '\nThe following Datasets must have a clear Comparison Dataset pair before moving forward. \n\n **Performance Metrics**:\n' + ''

                    '\n'.join([f'- {name}_perform_metrics' for name in comparison_dataframes_names_list]))


        # Check all Performance Datasets are accounted for
        elif performance_dataframes_names_list:
            
            # Create visual of the missing Performance Metric Datasets
            performance_metrics_datasets_uploaded_message = "\n".join(
                [f"\n:material/check: &nbsp; {name}_perform_metrics" 
                 
                 if f"{name}_perform_metrics" in [Path(file.name).stem for file in performance_metric_datasets_list] else f"\n:material/close: &nbsp; {name}_perform_metrics" 
                 
                 for name in comparison_dataframes_list.keys()
                ]
            )

            st.info(f"Performance Metrics Datasets are required for each uploaded Comparison Dataset. \nSome Comparison Datasets don't have a paired Performance Metrics Dataset \nEach Compariosn Dataset's Performance Metrics Dataset file name must be appended with '`_perform_metrics`' to be correctly identified. Expected Dataset name(s) are shown below:\n\n{performance_metrics_datasets_uploaded_message}")

        
        # Check errors that can only be caused in manual mode
        elif algorithm_error_flag:
            st.info(f"{algorithm_error_message}")


        else:




            #!------------------MAIN COMPARISON SECTION------------------!
            st_main_comparison(performance_metric_dataframes_list, reference_dataframes_list, comparison_dataframes_list, manual_task_type_selection)
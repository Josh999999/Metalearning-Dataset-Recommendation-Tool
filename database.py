#!------------------IMPORTS------------------!
from database_connection import setup_connection_db
import os
import json
import pandas as pd
import mysql.connector








#!------------------FUNCTIONALITY------------------!

# Database file - File used to query the Database by inserting and removing Database and algorithm data


"""Get all the text based queries from the sql files and insert into the dictionary for future use"""
def setup_queries_dictionary(path: str="./SQL/queries") -> dict[str]:
    """
    @param path - path to the folder where the SQL Queries are stored as .txt files

    @return query_dict - lookup containing each SQL query (as string) from the given folder location
    """
    query_dict = {}

    query_files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]


    for query_file in query_files:


        if query_file.endswith("txt"):
            query_file_path = f"{path}/{query_file}"


            with open(query_file_path) as read_query_file:
                san_read_query_file = query_file.split(".")[0]   

                query_dict[san_read_query_file] = read_query_file.read()


    return query_dict




"""Check functions"""


"""Check algorithm data is valid and return back correctly formatted data"""
def check_algorithm(algorithm_JSON: dict) -> str:
    """
    @param algorithm_JSON - objects containing all algorithm information

    @return algorithm_str - JSON object as string with sanatised algorithm attributes
    """    


    # raise exception if there is no algorithm column  
    if not "Algorithm" in algorithm_JSON:


        raise KeyError("'Algorithm' column must be present")
    

    # raise exception if there is no category column
    if not "Category" in algorithm_JSON:


        raise KeyError("'Category' column must be present")
    

    # raise exception if category value is not in valid list
    if algorithm_JSON["Category"].strip().lower() not in ["classification", "regression"]:


        raise ValueError("'Category' must be one of the following values: 'Classification', 'Regression'")

    
    # raise exception if there is no algorithm name (Required in the SQL table)
    if algorithm_JSON["Algorithm"].strip() == '':

        
        raise ValueError("'Algorithm' columns value is required in the SQL table")

    
    """Defunct code (keep incase neccessary)
    # raise exception if there is no Accuracy column  
    if not "Accuracy" in algorithm_JSON.keys():


        raise KeyError("'Accuracy' column must be present in the Database")
    
    
    # raise exception if there is no Accuracy score
    if isinstance(algorithm_JSON["Accuracy"], str) and algorithm_JSON["Accuracy"].strip() == '':

        
        raise ValueError("Each algorithm requires a numerical accuracy score for the related Database")
    

    # raise excpetion if Accuracy score can't be interpreted as float
    try:            
        _ = float(algorithm_JSON["Accuracy"])
    
    except ValueError as Error:

        raise ValueError(f"Accuracy must be interpretable as float, error caused: {Error}")
    """
    

    algorithm_str = json.dumps(algorithm_JSON) # convert to JSON for insertion


    return algorithm_str


"""Check Dataset name is valid"""
def check_dataset(dataset_name: str):
    """
    @param dataset_name - name of the given Dataset
    """
    
    # raise exception if there is no Dataset name (Required in the SQL table)
    if not dataset_name or dataset_name.strip() == '':

        
        raise ValueError("Dataset name is required in the SQL table")




"""INSERT statements"""


"""INSERT a single row (name, data, algorithm) into the Dataset table"""
def INSERT_dataset(db_cursor, dataset_name: str, dataset_data_JSON: list[dict], algorithms_JSON: list[dict]) -> int:
    """
    @param db_cursor - cursor pointing to the Database
    @param dataset_name - name of the Dataset to be inserted
    @param dataset_data_JSON - data contained in the Dataset
    @param algorithms_JSON - set of algorithms run on the Dataset

    @return db_cursor.lastrowid - the Datasets id if the insertion is successful
    """

    dataset_data_str = json.dumps(dataset_data_JSON)


    # Check the algorithms
    for algorithm_JSON in algorithms_JSON:
        check_algorithm(algorithm_JSON)
    

    algorithms_str = json.dumps(algorithms_JSON)


    # Check the Dataset
    check_dataset(dataset_name) 
    
    # Get the query through lookup dictionary
    query_dict = setup_queries_dictionary()
    INSERT_dataset_query = query_dict["INSERT_dataset"]

    db_cursor.execute(INSERT_dataset_query, (dataset_name, dataset_data_str, algorithms_str))


    return db_cursor.lastrowid




"""GET / SELECT functionality"""


"""GET all Datasets data with the Dataset name and algorithms"""
def GET_datasets_algorithms(db_cursor) -> tuple[str, list[dict], list[dict], list[dict]]:
    """
    @param db_cursor - cursor pointing to the Database

    @return list of tuples containing all Datasets and their algorithm respectively
    """

    # Get the query through lookup dictionary
    query_dict = setup_queries_dictionary()
    SELECT_datasets = query_dict["SELECT_datasets"]

    dataset_name_list = []
    dataset_data_list = []
    algorithms_list = []

    # Execute query        
    db_cursor.execute(SELECT_datasets)
    SELECTED_datasets = db_cursor.fetchall()


    for SELECTED_dataset in SELECTED_datasets:
        SELECTED_dataset_name = SELECTED_dataset[0]
        SELECTED_dataset_data = json.loads(SELECTED_dataset[1])


        # Check Dataset data has at least one meaniningfull key-value pair
        SELECTED_dataset_data_test = SELECTED_dataset_data[0] if isinstance(SELECTED_dataset_data, list) else SELECTED_dataset_data


        if not SELECTED_dataset_data_test or not any(v not in [None, '', [], {}, (), [{}]] for v in SELECTED_dataset_data_test.values()):


            continue
          

        SELECTED_dataset_algorithms = json.loads(SELECTED_dataset[2])  

        dataset_name_list.append(SELECTED_dataset_name)
        dataset_data_list.append(SELECTED_dataset_data)
        algorithms_list.append(SELECTED_dataset_algorithms)


    return (dataset_name_list, dataset_data_list, algorithms_list)


"""Get all Datasets from the Database, handling and errors that occur"""
def get_datasets_safe() -> tuple[list[str], list[dict], list[dict]]:
    """
    @return datasets_algorithms_tuple - list of tuples containing the Dataset name, data and algorithms 
    """

    db_connection = None
    datasets_algorithms_tuple = None


    try:



        with setup_connection_db() as db_connection:
            
            
            with db_connection.cursor() as db_cursor:
                datasets_algorithms_tuple = GET_datasets_algorithms(db_cursor)
            

            db_connection.commit()
    
    except Exception as Error:
        

        raise mysql.connector.InterfaceError(f"MySQL Interface Error: {Error}")
        

    return datasets_algorithms_tuple




"""INSERT given Dataset (dataset_name and dataset_data_DF) and algorithms, handling errors that occur"""
def INSERT_dataset_with_algorithms_safe(dataset_name: str, dataset_data_DF: pd.DataFrame, dataset_algorithms_DF: pd.DataFrame) -> list[str, bool]:
    """
    @param dataset_name - name of the Dataset to be inserted
    @param dataset_data_DF - a list of Dataset data in a pandas DataFrame object
    @param dataset_algorithms_DF - a list of algorithms in a pandas DataFrame object
    """

    # Convert the Dataframes into JSON objects (dicts)
    dataset_data_JSON = json.loads(dataset_data_DF.to_json(orient='records', lines=False))
    dataset_algorithms_JSON = json.loads(dataset_algorithms_DF.to_json(orient='records', lines=False))
    
    db_connection = None


    # Get the cursor for query execution - if there is any error raise general connection error message
    try:


        with setup_connection_db() as db_connection:

            # Ensure autocommit is disabled for manual transaction control
            db_connection.autocommit = False  
                
            
            with db_connection.cursor() as db_cursor:
                _ = INSERT_dataset(db_cursor, dataset_name, dataset_data_JSON, dataset_algorithms_JSON)            


            #otherwise we commit the last SQL function
            db_connection.commit()
        
    except Exception as Error:
        
        
        raise mysql.connector.DataError(f"MySQL Insertion Error: {Error}")
    
    
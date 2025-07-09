#!------------------IMPORTS------------------!
import pandas as pd
from pathlib import Path
from typing import Callable
import json








#!------------------FUNCTIONALITY------------------!


# Top level file declaratives
cwd = Path.cwd()
FILES = cwd / "files"
DATASETS = FILES / "datasets"
EXAMPLES = FILES / "examples"
IMAGES = FILES / "images"


def is_flat_json(obj):


    """Dictionaries will need to be assessed multiple times"""
    def is_dictionary_unnested(dictionary):


        return not (any([isinstance(value, list) for value in dictionary.values()]) or 
                    any([isinstance(value, dict) for value in dictionary.values()]) or
                    any([isinstance(value, set) for value in dictionary.values()]) or
                    any([isinstance(value, tuple) for value in dictionary.values()])
                    )


    # The main content (Dictionary containing columns key-value pairs) will be wrapped in a list or it won't
    if isinstance(obj, dict):


        return is_dictionary_unnested(obj)
    
    elif isinstance(obj, list):


        for item in obj:


            # Make sure the list only contains dictionaries
            if not isinstance(item, dict):


                return False

            elif not is_dictionary_unnested(item):


                return False
        

        return True
    

    # If not dictionary or list then not a valid JSON structure for tabular data
    return False


"""Import each Dataset, load as DataFrames & return as dict of DataFrame dicts"""
def get_stored_datasets(datasets_path_list, is_file_obj=True) -> dict[pd.DataFrame]:
    """
    @param datasets_path_list - List of path locations of each uploaded Dataset 
    @param is_file_obj - Boolean value that specifies wether the Datasets are held in a file object or not (likely file location)

    @return dataframes - Dictionary of Datasets loaded as Dataframes with the Dataset name as the key
    """
    
    dataframes = {}


    for file in datasets_path_list:


        if not is_file_obj:
            file_obj = Path(file.name)
    
    
        # Best not to do a series of embedded if statements, probably change to a switch case down line
        if file_obj.suffix == ".csv":
            dataframes[file_obj.stem] = pd.read_csv(file, skipinitialspace=True, converters={col: str.strip for col in range(0, 10)})


            if not is_flat_json(json.loads(dataframes[file_obj.stem].to_json(orient="records"))):

                
                raise ValueError(f"Value Error: {file.name} is not DataFrame consumable (likely nested)")
        
        elif file_obj.suffix == ".json":
            dataset = None


            try:
                file.seek(0)
                dataset = json.loads(file.read().decode("utf-8"))

            except Exception as e:


                raise json.JSONDecodeError("JSON Decoding Error: File was likely not in correct JSON format and could not be decoded")
        

            file.seek(0)


            if not dataset:
            

                raise ValueError(f"Value Error: {file.name} doesn't contain any data")

            elif isinstance(dataset, list):


                try:
                    dataframes[file_obj.stem] = pd.read_json(file).applymap(lambda x: x.strip() if isinstance(x, str) else x)

                except Exception as Error:


                    raise json.JSONDecodeError("JSON Decoding Error: File was not in correct JSON format and could not be decoded into DataFrame")

            elif isinstance(dataset, dict):


                try:
                    dataframes[file_obj.stem] = pd.read_json(json.dumps([dataset])).applymap(lambda x: x.strip() if isinstance(x, str) else x)

                except Exception as Error:


                    raise json.JSONDecodeError("JSON Decoding Error: File was not in correct JSON format and could not be decoded into DataFrame")


            if isinstance(dataframes[file_obj.stem], pd.DataFrame):


                if dataframes[file_obj.stem].empty:
            

                    raise ValueError(f"Value Error: {file.name} doesn't contain any data")
            
            else:
                
                raise ValueError(f"Value Error: {file.name} couldn't be converted to a DataFrame")


            if not is_flat_json(json.loads(dataframes[file_obj.stem].to_json(orient="records"))):

                
                raise ValueError(f"Value Error: {file.name} is not DataFrame consumable (likely nested)")
        
        else:


            # Throw up a warning here eventually
            raise FileNotFoundError(f"File Not Found Error: Dataset: {file.name} is not of a supported type")


    return dataframes


"""Style a specific column in a Streamlit display of a Pandas DF based on a given column name"""
def highlight_column(highlighted_column: str) -> Callable[[str], list[str]]:
    """
    @param highlighted_column - Name of the column to be highlighted

    @return style_function - Function that highlights the applied to column if it's the given column name
    """


    # Return style function that will highlight the column
    def style_function(column: str) -> list[str]:


        if column.name == highlighted_column:


            return ["background-color: lightblue" for _ in column]
        
        else:


            return ["" for _ in column]
        

    return style_function
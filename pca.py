#!------------------IMPORTS------------------!
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype








#!------------------FUNCTIONALITY------------------!


"""Applies the basic PCA operations to a given Dataset to reduce it to n number of features (columns)"""
def applyPCA(df_1: pd.DataFrame, components: int) -> list[pd.DataFrame, PCA]:
    """
    @param df_1 - DataFrame to apply the PCA operation to
    @param components - Number of components (features / column) to reduce the Dataset down to
    
    @returns df1_pca - DataFrame after the PCA operation has been applied
    @returns pca - Object used to transform DataFrame; contains information about the PCA transformation
    @returns pca - Object used to transform DataFrame; contains information about the PCA transformation
    """
    

    # Handle any column which ins't numerical
    columns_list = df_1.columns.tolist()
    encoded_columns_list = list()

    
    # Encode each column that isn't numerical (treats them as categorical)
    for index in range(0, len(columns_list)):
        column = columns_list[index]

        
        if not is_numeric_dtype(df_1[column]):
            label_encoder = LabelEncoder()
            df_1[column] = label_encoder.fit_transform(df_1[column])
            encoded_columns_list.append((index, label_encoder))


    # Apply PCA
    scaler = StandardScaler()
    df1_scaled = scaler.fit_transform(df_1)
    pca = PCA(n_components=components)
    df1_pca = pd.DataFrame(pca.fit_transform(df1_scaled))


    return df1_pca, pca


"""Applies PCA to two DataFrames so that both Datasets have above 95% explained variance and the same number of components (features / columns)"""
def sklearn_two_datasets_pca(ds_1: pd.DataFrame, ds_2: pd.DataFrame) -> list[pd.DataFrame, pd.DataFrame, PCA, PCA, list[int, int]]:
    """
    @param df_1 - First DataFrame to apply the PCA operation to 
    @param ds_2 - Second DataFrame to apply the PCA operation to
    
    @returns df1_pca - First DataFrame after the PCA operation has been applied
    @returns df2_pca - Second DataFrame after the PCA operation has been applied
    @returns pca_1 - Object used to transform the first DataFrame; contains information about the PCA transformation
    @returns pca_1 - Object used to transform the second DataFrame; contains information about the PCA transformation
    @returns num_components - Contains the number of components the first and second data have been reduced to (not neccessarily the same number)
    """

    # Find 95% explained variance for both Datasets:
    dspca_1, pca_1 = applyPCA(ds_1, .95)
    dspca_2, pca_2 = applyPCA(ds_2, .95)

    # Resolves the Datasets to have the same number of components above 95% explained variance (or the next best thing)
   
    # Initialise variables
    num_components = [0, 0]
    ds_1_columns = len(ds_1.columns)
    ds_2_columns = len(ds_2.columns)

    # Get the number of components from pca
    num_components[0] = pca_1.n_components_
    num_components[1] = pca_2.n_components_


    # Bring dspca_2 num of components up to same amount as dspca_1
    if num_components[0] > num_components[1]:


        try:


            # Check ds_2 has as many columns as dspca_1 has number of components
            if ds_2_columns <  num_components[0]:

                # Not enough components in Dataset 2 to match Dataset 1 PCA
                dspca_1, pca_1 = applyPCA(ds_1, ds_2_columns)
                dspca_2, pca_2 = applyPCA(ds_2, ds_2_columns)

                num_components[0] = num_components[1] = ds_2_columns
            
            else:
                dspca_2, pca_2 = applyPCA(ds_2, num_components[0])
                num_components[1] = num_components[0]

        except Exception as Error:
                
                # Likely cause by not having enough samples
                n_min_component = min(ds_2.shape)
                
                # Not enough components in Dataset 2 to match Dataset 1 PCA
                dspca_1, pca_1 = applyPCA(ds_1, n_min_component)
                dspca_2, pca_2 = applyPCA(ds_2, n_min_component)

                num_components[0] = num_components[1] = n_min_component
    
    # Bring dspca_1 num of components up to same amount as dspca_2
    elif num_components[0] < num_components[1]:


        try:


            # Check ds_1 has as many columns as dspca_2 has number of components
            if ds_1_columns < num_components[1]:                        

                # Not enough components in Dataset 1 to match Dataset 2 PCA
                dspca_1, pca_1 = applyPCA(ds_1, ds_1_columns)
                dspca_2, pca_2 = applyPCA(ds_2, ds_1_columns)

                num_components[0] = num_components[1] = ds_1_columns

            else:
                dspca_1, pca_1 = applyPCA(ds_1, num_components[1])
                num_components[0] = num_components[1]

        
        except Exception as Error:
                
                # Likely cause by not having enough samples
                n_min_component = min(ds_1.shape)
                
                # Not enough components in Dataset 2 to match Dataset 1 PCA
                dspca_1, pca_1 = applyPCA(ds_1, n_min_component)
                dspca_2, pca_2 = applyPCA(ds_2, n_min_component)

                num_components[0] = num_components[1] = n_min_component


    return dspca_1, dspca_2, pca_1, pca_2, num_components
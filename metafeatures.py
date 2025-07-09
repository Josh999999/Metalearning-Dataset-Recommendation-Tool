#!------------------IMPORTS------------------!
import numpy as np
import pandas as pd
from scipy import stats








#!------------------FUNCTIONALITY------------------!


"""Extracts PCA based metafeatures from a DataFrame / Dataset that has had PCA applied"""
def get_mf_pca(df: pd.DataFrame, df_pca: pd.DataFrame) -> dict[str, int]:
    """
    @param df - DataFrame where PCA has been applied for the metafeatures to be extracted from
    @param df_pca - DataFrame after PCA has been applied for the metafeatures to be extracted from

    @returns mf_pca - Dictionary of PCA metafeatures extracted from a Dataset and the PCA applied Dataset
    """

    mf_pca = {}
    
    # Number of dimensions (features / columns) after pca
    mf_pca["n_d#"] = len(df_pca.columns)
    
    # Number of dimensions (features / columns)
    mf_pca["n_d"] = len(df.columns)

    # Number of samples
    mf_pca["n_n"] = len(df)

    # Log number of features
    mf_pca["log_d"] = np.log(len(df.columns))

    # Log number of features over samples
    mf_pca["log_n_over_d"] = np.log(len(df.columns)/len(df))

    # Create p Metrics for pca applied Dataset
    mf_pca["n_p"] = len(df_pca.columns) / len(df.columns)


    return mf_pca


"""Creates DataFrame to be displayed as a table containing information about PCA metafeatures"""
def create_pca_metrics_table(pca_metafeature_list: list[dict]) -> pd.DataFrame:
    """
    @param pca_metafeature_list - List of dictionaries containing PCA metafeatures extracted from a PCA applied Dataset

    @returns mf_pca - DataFrame containing information extracted from the PCA metafeatures list
    """

    # Create pca Metrics table as seen in the documentation
    pca_table_df = pd.DataFrame()

    # Get list of features to get statistics from
    d_pca_features = [pca_metafeature['n_d#'] for pca_metafeature in pca_metafeature_list]
    d_features = [pca_metafeature['n_d'] for pca_metafeature in pca_metafeature_list]
    n_features = [pca_metafeature['n_n'] for pca_metafeature in pca_metafeature_list]
    log_d_features = [pca_metafeature['log_d'] for pca_metafeature in pca_metafeature_list]
    log_n_over_d_features = [pca_metafeature['log_n_over_d'] for pca_metafeature in pca_metafeature_list]
    p_features = [pca_metafeature['n_p'] for pca_metafeature in pca_metafeature_list]

    # Add to min collumn
    pca_table_df['min'] = [np.min(d_pca_features),
                           np.min(d_features),
                           np.min(n_features),
                           np.min(log_d_features),
                           np.min(log_n_over_d_features),
                           np.min(p_features)                           
    ]

    # Add 1st percentage quartile column
    pca_table_df['1stQrtl'] = [stats.scoreatpercentile(d_pca_features, 25),
                           stats.scoreatpercentile(d_features, 25),
                           stats.scoreatpercentile(n_features, 25),
                           stats.scoreatpercentile(log_d_features, 25),
                           stats.scoreatpercentile(log_n_over_d_features, 25),
                           stats.scoreatpercentile(p_features, 25)                           
    ]

    # Add 2nd percentage quartile (median) column
    pca_table_df['median'] = [stats.scoreatpercentile(d_pca_features, 50),
                           stats.scoreatpercentile(d_features, 50),
                           stats.scoreatpercentile(n_features, 50),
                           stats.scoreatpercentile(log_d_features, 50),
                           stats.scoreatpercentile(log_n_over_d_features, 50),
                           stats.scoreatpercentile(p_features, 50)                           
    ]

    # Add 3rd percentage quartile column
    pca_table_df['3rdQrtl'] = [stats.scoreatpercentile(d_pca_features, 75),
                           stats.scoreatpercentile(d_features, 75),
                           stats.scoreatpercentile(n_features, 75),
                           stats.scoreatpercentile(log_d_features, 75),
                           stats.scoreatpercentile(log_n_over_d_features, 75),
                           stats.scoreatpercentile(p_features, 75)                           
    ]

    # Add to max collumn
    pca_table_df['max'] = [np.max(d_pca_features),
                           np.max(d_features),
                           np.max(n_features),
                           np.max(log_d_features),
                           np.max(log_n_over_d_features),
                           np.max(p_features)                           
    ]

    # Rename rows for readability
    pca_table_df.index = ["d`", "d", "n", "log(d)", "log(n/d)", "p"]


    return pca_table_df


def get_mf_simple(df: pd.DataFrame) -> dict[str, int]:
    """
    @param df - DataFrame for the metafeatures to be extracted from

    @returns mf_pca - Dictionary containing metafeatures extracted from the DataFrame
    """

    mf_simple = {}


    # Metafeatures not in the table

    # Number of patterns (rows)
    mf_simple["n_patterns"] = df.shape[0] # Check this is best way of getting rows

    # Number of continuous features
    mf_simple["n_cols_continuous"]  = df.select_dtypes(include=["int64", "float64"]).shape[1]

    # Number of categorical features
    mf_simple["n_cols_categorical"] = df.select_dtypes(include=["object", "category"]).shape[1]

    # Log number of features over patterns
    mf_simple["log_n_features_over_n_patterns"] = np.log(df.shape[0])/(df.shape[1])



    # Metafeature in the table

    # Log number of patterns
    mf_simple["log_n_patterns"] = np.log(df.shape[0])

    # Number of classes

    # Number of Features (cols)
    mf_simple["n_features"] = df.shape[1]

    # Log number of features
    mf_simple["log_n_features"] = np.log(df.shape[1])

    # Number of patterns with missing values    
    mf_simple["n_patterns_with_missing"] = df.isna().any(axis=1).sum()

    # Percentage of patterns with missing values    
    mf_simple["p_patterns_with_missing"] = (df.isna().any(axis=1).sum() / df.shape[0]) * 100

    # Number of features with missing values
    mf_simple["n_features_with_missing"] = df.isna().any(axis=0).sum()

    # Percentage of features with missing values
    mf_simple["p_features_with_missing"] = (df.isna().any(axis=0).sum() / df.shape[1]) * 100

    # Number of missing values
    mf_simple["n_missing_values"] = df.isna().sum().sum()

    # Percentage of missing values
    mf_simple["p_missing_values"] = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100

    # Number of numeric features 
    mf_simple["n_numeric_features"] = df.select_dtypes(include='number').shape[1]

    # Number of categorical features
    mf_simple["n_categorical_features"] = df.select_dtypes(include=['object', 'category']).shape[1]


    # Ratio of numeric features to categorical features
    if df.select_dtypes(include=['object', 'category']).shape[1] > 0:
        mf_simple["n_categorical_features"] = df.select_dtypes(include='number').shape[1] / df.select_dtypes(include=['object', 'category']).shape[1]
    
    else:
        mf_simple["n_categorical_features"] = 0


    # Ratio of categorical features to numeric features
    if df.select_dtypes(include='number').shape[1] > 0:
        mf_simple["n_categorical_features"] = df.select_dtypes(include=['object', 'category']).shape[1] / df.select_dtypes(include='number').shape[1]
    
    else:
        mf_simple["n_categorical_features"] = 0


    # Ratio of categorical features to numeric features
    if df.select_dtypes(include='number').shape[1] > 0:
        mf_simple["n_categorical_features"] = df.select_dtypes(include=['object', 'category']).shape[1] / df.select_dtypes(include='number').shape[1]

    else:
        mf_simple["n_categorical_features"] = 0


    # Dimensionality of the Dataset
    mf_simple["d_dataset"] = sum(_ for _ in df.shape)

    # Log dimensionality of the Dataset    
    mf_simple["log_d_dataset"] = np.log(sum(_ for _ in df.shape))

    # Inverse dimensionality of the Dataset    
    mf_simple["inv_d_dataset"] = 1 / sum(_ for _ in df.shape)

    # Log inverse dimensionality of the Dataset    
    mf_simple["log_inv_d_dataset"] = np.log(1 / sum(_ for _ in df.shape))

    # class probability min

    # class probability max

    # class probability mean

    # class probability standard deviation


    return mf_simple


"""Create a dictionary of all extractable metafeatures"""
def get_mf(df: pd.DataFrame, df_pca: pd.DataFrame) -> dict[str, int]:
    """
    @param df - DataFrame where PCA has been applied for the metafeatures to be extracted from
    @param df_pca - DataFrame after PCA has been applied for the metafeatures to be extracted from

    @returns mf_pca - Dictionary of all extractable metafeatures from the DataFrame
    """

    mf_pca = get_mf_pca(df, df_pca)
    mf_simple = get_mf_simple(df)

    mf = mf_pca | mf_simple


    return mf


"""Calculates similarity of explained variance patterns"""
def variance_ratio_similarity(pca1_explained_variance: np.ndarray[float], pca2_explained_variance: np.ndarray[float]) -> np.float64:
    """
    @param pca1_explained_variance - List of real numbers describing how much explained variance is contained in each column (in order)
    @param pca2_explained_variance - List of real numbers describing how much explained variance is contained in each column (in order)

    @returns similarity - Variance ration similarity of the two sets of Dataset columns explained variance
    """

    # Make same length by padding shorter with zeros
    max_len = max(len(pca1_explained_variance), len(pca2_explained_variance))
    pca1_explained_variance_padded = np.pad(pca1_explained_variance, (0, max_len - len(pca1_explained_variance)))
    pca2_explained_variance_padded = np.pad(pca2_explained_variance, (0, max_len - len(pca2_explained_variance)))
    
    # Calculate cosine similarity or other distance Metrics
    similarity = abs(1 - np.sqrt(np.mean((pca1_explained_variance_padded - pca2_explained_variance_padded)**2)))


    return similarity


"""Calculate the similarity between PCAs based on minimum num of components for each PCA to reach minimum threshold for Explained Variance Ratio (95%)"""
def component_number_similarity(pca1_explained_variance: np.ndarray[float], pca2_explained_variance: np.ndarray[float], threshold: float = 0.95) -> np.float64:
    """
    @param pca1_explained_variance - List of real numbers describing how much explained variance is contained in each column (in order)
    @param pca2_explained_variance - List of real numbers describing how much explained variance is contained in each column (in order)
    @param threshold - Percentage of explained variance needed to calculate component number similarity

    @returns similarity - Component number similarity of the two sets of Dataset columns explained variance
    """

    # Find number of components needed for threshold variance
    def n_components_for_variance(explained_variance_ratio, threshold):
        cumsum = np.cumsum(explained_variance_ratio)


        return np.argmax(cumsum >= threshold) + 1
    

    n1 = n_components_for_variance(pca1_explained_variance, threshold)
    n2 = n_components_for_variance(pca2_explained_variance, threshold)
    
    # Calculate similarity based on difference in component numbers
    similarity = abs(1 - abs(n1 - n2) / max(n1, n2))


    return similarity


"""Calculates eigenvalue similarity of explained variance patterns"""
def eigenvalue_similarity(pca1_explained_variance: np.ndarray[float], pca2_explained_variance: np.ndarray[float]) -> np.float64:
    """
    @param pca1_explained_variance - List of real numbers describing how much explained variance is contained in each column (in order)
    @param pca2_explained_variance - List of real numbers describing how much explained variance is contained in each column (in order)

    @returns similarity - Eigenvalue similarity of the two sets of Dataset columns explained variance
    """

    # Normalise eigenvalues
    eig1_norm = pca1_explained_variance / pca1_explained_variance.sum()
    eig2_norm = pca2_explained_variance / pca2_explained_variance.sum()
    
    # Make same length
    max_len = max(len(eig1_norm), len(eig2_norm))
    eig1_padded = np.pad(eig1_norm, (0, max_len - len(eig1_norm)))
    eig2_padded = np.pad(eig2_norm, (0, max_len - len(eig2_norm)))
    
    # Calculate decay pattern similarity
    similarity = abs(1 - np.sqrt(np.mean((eig1_padded - eig2_padded)**2)))


    return similarity


"""Comparison method 1 - calculate the pnorm value for a difference between Datasets"""
def get_pNorm(metafeatures1: list[np.float64], metafeatures2: list[np.float64], p: int) -> np.float64:
    """
    @param metafeatures1 - List of extracted metafeatures
    @param metafeatures1 - List of extracted metafeatures

    @returns metafeatures_p_norm_calc2 - P-Norm similarity from the calculation including the two sets of metafeatures
    """

    metafeatures_p_norm_calc2 = 0.0


    if len(metafeatures1) != len(metafeatures2):


        return metafeatures_p_norm_calc2

    
    metafeatures_difference_vector = np.array([metafeatures1[index] - metafeatures2[index] for index in range(0, len(metafeatures1))]) 
    metafeatures_p_norm_calc1 = abs(np.linalg.norm(metafeatures_difference_vector, ord=p))
    metafeatures_p_norm_calc2 = 1 if metafeatures_p_norm_calc1 == 0 else 1 / metafeatures_p_norm_calc1

    
    return metafeatures_p_norm_calc2


"""Comparison method 2 - Calculate a similarity value by comparing the components-feature ration between two Datasets"""
def component_feature_similarity(metafeatures1: dict, metafeatures2: dict) -> np.float64:
    """
    @param metafeatures1 - List of extracted metafeatures
    @param metafeatures1 - List of extracted metafeatures

    @returns component_feature_similarity_value - Component feature similarity from the calculation including the two sets of metafeatures
    """

    component_feature_similarity_value = 0.0    

    ds1_component_feature_ratio = metafeatures1["n_d#"] / metafeatures1["n_n"]
    ds2_component_feature_ratio = metafeatures2["n_d#"] / metafeatures2["n_n"]


    if ds1_component_feature_ratio == ds2_component_feature_ratio:


        return component_feature_similarity_value


    component_feature_similarity_value = abs(1 - abs(ds1_component_feature_ratio - ds2_component_feature_ratio))


    return component_feature_similarity_value
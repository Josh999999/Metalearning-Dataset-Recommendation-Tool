#!------------------IMPORTS------------------!
import pandas as pd
import streamlit as st
import numpy as np
from metafeatures import *
from pca import *
from view_utils import create_ranked_bar_chart, rank_algorithms








#!------------------FUNCTIONALITY------------------!


"""Function that creates and runs the Comparison functionality and streamlit UI elements for it"""
def st_main_comparison(
    performance_metric_dataframes: list[pd.DataFrame],
    reference_dataframes: list[pd.DataFrame],
    comparison_dataframes: list[pd.DataFrame],
    task_type_selection: str
) -> None:
    """
    @param performance_metric_dataframes - Dictionary of DataFrames containing the algorithms (with their Performance Metrics) applied to each Comparison Dataset
    @param reference_dataframes - Dictionary of DataFrames containing the data of the Reference Dataset
    @param comparison_dataframes - Dictionary of DataFrames containing the data of the Comparison Datasets
    @param task_type_selection - The type of task that the algorithms should be able to perform
    """


    # Filter the Performance Metrics Dataframes based on the selected task type - program should error if algorithms don't have categoriesprint("\n\n\n")                
    for name_performance_metric_dataframe, performance_metric_dataframe in performance_metric_dataframes.items():


        if "category" not in performance_metric_dataframe.columns.str.lower():
            performance_metric_dataframe['Category'] = ["Classification"] * performance_metric_dataframe.shape[0]
            

        performance_metric_dataframe['Category'] = [category.strip().lower() for category in performance_metric_dataframe['Category']]
        performance_metric_dataframe = performance_metric_dataframe.loc[performance_metric_dataframe['Category'] == task_type_selection.strip().lower()]
        performance_metric_dataframe['Category'] = performance_metric_dataframe['Category'].str.capitalize()

        # Re-assign the key (file names) DataFrame value to the filtered version
        performance_metric_dataframes[name_performance_metric_dataframe] = performance_metric_dataframe


    st.divider()
    st.html("<br>")

    # SUBHEADER - Dataset Similarity Ranking
    st.subheader(":material/text_compare: Dataset Similarity Ranking")

    st.write("Investigate Dataset similarity by a selection of different statistical methods.")

    pca_similarity_tab, component_feature_similarity_tab, = st.tabs(["PCA P-norm Similarity", "Component-feature Similarity"])


    with pca_similarity_tab:
        st.html("<h3 style='margin: 0px; padding: 0px;'><b>PCA P-norm Similarity</b></h3>")

        st.write("""
        In this method, similarity is calculated based on 'Database distance'\nThis takes the summed difference between each Datasets respective metafeatures and applies the p-norm operation to this sum.
                    
        \n\nHere we experiment with a p value in the range of 1 to 3.
                        
        \n\n- A p-norm where `p = 1` is known as the '**Manhattan distance**' Metrics - this p-norm is especially sensitive to small differences.       
        \n- A p-norm where `p = 2` is known as the '**Euclidean vector norm**' - this p-norm penalizes large differences moreso than smaller ones, thus it is likely the Euclidean vector norm is preferred.
        """)
        
        st.html("</br> <h3 style='margin: 0px; padding: 0px;'><b>Dataset Comparison Statistics</b></h3>")
        st.write("Inspect the data points behind each Dataset-to-Dataset Comparison.")
            
        name_reference, df_reference = next(iter(reference_dataframes.items()))

        df_similarity_rankings = pd.DataFrame()


        for name_comparison, df_comparison in comparison_dataframes.items():
            reference_df_pca, comparison_df_pca, reference_pca_obj, comparison_pca_obj, num_components = sklearn_two_datasets_pca(df_reference, df_comparison)
            
            df_reference_metafeatures = get_mf(df_reference, reference_df_pca)
            df_comparison_metafeatures = get_mf(df_comparison, comparison_df_pca)


            with st.expander(f"{name_reference} <-> {name_comparison}"):
                col_reference_evr, col_num_components = st.columns(2)


                # COLUMN - Contains the explained variance ratio of the Comparison Dataset
                with col_reference_evr:
                    st.metric(f"{name_reference} Explained Variance Ratio", str(np.round(np.sum(reference_pca_obj.explained_variance_ratio_), 3)))


                # COLUMN - Contains the number of PCA components for both Datasets
                with col_num_components:
                    st.metric("Num Components", str(num_components[0]))


                col_variance_ratio_similarity, col_component_number_similarity, col_eigenvalue_similarity = st.columns(3)
                

                # COLUMN - Contains the variance ration similarity between the two Datasets
                with col_variance_ratio_similarity:
                    st.metric("Variance Ratio Similarity", np.round(variance_ratio_similarity(reference_pca_obj.explained_variance_, comparison_pca_obj.explained_variance_), 3))


                # COLUMN - Contains component number similarity between the two Datasets
                with col_component_number_similarity:
                    st.metric("Component Number Similarity", np.round(component_number_similarity(reference_pca_obj.explained_variance_, comparison_pca_obj.explained_variance_), 3))


                # COLUMN - Contains Eigenvalue similarity between the two Datasets
                with col_eigenvalue_similarity:
                    st.metric("Eigenvalue Similarity", np.round(eigenvalue_similarity(reference_pca_obj.explained_variance_, comparison_pca_obj.explained_variance_), 3))


                p_norm_1, p_norm_2, p_norm_3 = st.columns(3)


                # COLUMN - Contains the P-Norm similarity between the two Datasets (set to p = 1)
                with p_norm_1:
                    st.metric("P-norm 1", np.round(get_pNorm(list(df_reference_metafeatures.values()), list(df_comparison_metafeatures.values()), 1), 3))


                # COLUMN - Contains the P-Norm similarity between the two Datasets (set to p = 2)
                with p_norm_2:
                    st.metric("P-norm 2", np.round(get_pNorm(list(df_reference_metafeatures.values()), list(df_comparison_metafeatures.values()), 2), 3))


                # COLUMN - Contains the P-Norm similarity between the two Datasets (set to p = 3)
                with p_norm_3:
                    st.metric("P-norm 3", np.round(get_pNorm(list(df_reference_metafeatures.values()), list(df_comparison_metafeatures.values()), 3), 3))


                # Create the Metrics table
                st.write("**PCA Metrics Table**")

                pca_metrics = create_pca_metrics_table([df_reference_metafeatures, df_comparison_metafeatures])
                pca_metrics = pca_metrics.apply(pd.to_numeric, errors='coerce')  # Converts all columns to numeric where possible
                pca_metrics_rounded = pca_metrics.round(3)

                st.dataframe(pca_metrics_rounded.style.format(precision=3), hide_index=True)


            # Add similarity Metrics to the DataFrame
            df_similarity = pd.DataFrame({
                "Dataset": name_comparison,
                "P-norm-1 Similarity": np.round(get_pNorm(list(df_reference_metafeatures.values()), list(df_comparison_metafeatures.values()), 1), 3),
                "P-norm-2 Similarity": np.round(get_pNorm(list(df_reference_metafeatures.values()), list(df_comparison_metafeatures.values()), 2), 3),
                "P-norm-3 Similarity": np.round(get_pNorm(list(df_reference_metafeatures.values()), list(df_comparison_metafeatures.values()), 3), 3)
            }, index=[0])
            df_similarity_rankings = pd.concat([df_similarity_rankings, df_similarity])


        # Create table and visualisations of similarity matrix
        df_similarity_rankings_sorted = df_similarity_rankings.sort_values(by="P-norm-1 Similarity", ascending=False)
        df_similarity_rankings_sorted = df_similarity_rankings_sorted.reset_index(drop=True)            

        st.html("</br> <h3 style='margin: 0px; padding: 0px;'><b>P-norm Similarity Table</b></h3>")

        st.dataframe(df_similarity_rankings_sorted)

        create_ranked_bar_chart(df_similarity_rankings_sorted, "P-norm-1 Similarity")
        
        st.divider()
        st.html("<br>")

        rank_algorithms(df_similarity_rankings_sorted, performance_metric_dataframes, similarity_type="PCA P-norm Similarity (P-norm-1)", prefix="pnrm")


    with component_feature_similarity_tab:            
        st.html("<h3 style='margin: 0px; padding: 0px;'><b>Component-feature Similarity</b></h3>")

        st.write("""
        This method calculates the component-feature ratio of each Dataset, then calculating the difference between each of the two results to get the similarity.
                    
        The result obtained is inversed, as the objective is maximisation.
        """)

        st.html("</br> <h3 style='margin: 0px; padding: 0px;'><b>Dataset Comparison Statistics</b></h3>")
        st.write("Inspect the data points behind each Dataset-to-Dataset Comparison.")
            
        name_reference, df_reference = next(iter(reference_dataframes.items()))

        df_similarity_rankings = pd.DataFrame()


        for name_comparison, df_comparison in comparison_dataframes.items():
            reference_df_pca, reference_pca_obj = applyPCA(df_reference, .95)
            comparison_df_pca, comparison_pca_obj = applyPCA(df_comparison, .95)
            
            df_reference_metafeatures = get_mf(df_reference, reference_df_pca)
            df_comparison_metafeatures = get_mf(df_comparison, comparison_df_pca)


            with st.expander(f"{name_reference} <-> {name_comparison}"):
                col_reference_evr, col_num_components = st.columns(2)


                # COLUMN - Contains the explained variance ratio of the Comparison Dataset
                with col_reference_evr:
                    st.metric(f"{name_reference} Explained Variance Ratio", str(np.round(np.sum(reference_pca_obj.explained_variance_ratio_), 3)))


                # COLUMN - Contains the number of PCA components for both Datasets
                with col_num_components:
                    st.metric("Num Components", str(num_components[0]))


                col_variance_ratio_similarity, col_component_number_similarity, col_eigenvalue_similarity = st.columns(3)
                

                # COLUMN - Contains the variance ration similarity between the two Datasets
                with col_variance_ratio_similarity:
                    st.metric("Variance Ratio Similarity", np.round(variance_ratio_similarity(reference_pca_obj.explained_variance_, comparison_pca_obj.explained_variance_), 3))


                # COLUMN - Contains component number similarity between the two Datasets
                with col_component_number_similarity:
                    st.metric("Component Number Similarity", np.round(component_number_similarity(reference_pca_obj.explained_variance_, comparison_pca_obj.explained_variance_), 3))


                # COLUMN - Contains Eigenvalue similarity between the two Datasets
                with col_eigenvalue_similarity:
                    st.metric("Eigenvalue Similarity", np.round(eigenvalue_similarity(reference_pca_obj.explained_variance_, comparison_pca_obj.explained_variance_), 3))


                st.metric("Component-feature Similarity", np.round(component_feature_similarity(df_reference_metafeatures, df_comparison_metafeatures), 3))


                # Create Metrics table
                st.write("**PCA Metrics Table**")

                pca_metrics = create_pca_metrics_table([df_reference_metafeatures, df_comparison_metafeatures])
                pca_metrics = pca_metrics.apply(pd.to_numeric, errors='coerce')  # Converts all columns to numeric where possible
                pca_metrics_rounded = pca_metrics.round(3)

                st.dataframe(pca_metrics_rounded.style.format(precision=3), hide_index=True)


            # Add similarity Metrics to the DataFrame
            df_similarity = pd.DataFrame({
                "Dataset": name_comparison,
                "Component-feature Similarity": np.round(component_feature_similarity(df_reference_metafeatures, df_comparison_metafeatures), 3),
                "Dataset Component-feature rationale": f"{np.round(df_reference_metafeatures['n_d'] / df_reference_metafeatures['n_n'], 3)}  =  {df_reference_metafeatures['n_d']} / {df_reference_metafeatures['n_n']}",
                "pca Dataset Component-feature rationale": f"{np.round(df_reference_metafeatures['n_d#'] / df_reference_metafeatures['n_n'], 3)}  =  {df_reference_metafeatures['n_d#']} / {df_reference_metafeatures['n_n']}",                
                name_reference + " Component-feature rationale": f"{np.round(df_comparison_metafeatures['n_d'] / df_comparison_metafeatures['n_n'], 3)}  =  {df_comparison_metafeatures['n_d']} / {df_comparison_metafeatures['n_n']}",
                "pca " + name_reference + " Component-feature rationale": f"{np.round(df_comparison_metafeatures['n_d#'] / df_comparison_metafeatures['n_n'], 3)}  =  {df_comparison_metafeatures['n_d#']} / {df_comparison_metafeatures['n_n']}"
            }, index=[0])
            
            df_similarity_rankings = pd.concat([df_similarity_rankings, df_similarity])


        # Create table and visualisations of similarity matrix
        df_similarity_rankings_sorted = df_similarity_rankings.sort_values(by="Component-feature Similarity", ascending=False)
        df_similarity_rankings_sorted = df_similarity_rankings_sorted.reset_index(drop=True)
        
        st.html("</br> <h3 style='margin: 0px; padding: 0px;'><b>Component-feature Similarity table</b></h3>")

        st.dataframe(df_similarity_rankings_sorted)

        create_ranked_bar_chart(df_similarity_rankings_sorted, "Component-feature Similarity")
        
        st.divider()
        st.html("<br>")

        rank_algorithms(df_similarity_rankings_sorted, performance_metric_dataframes, similarity_type="Component-feature Similarity", prefix="cfs")

        st.divider()
        st.html("<br>")
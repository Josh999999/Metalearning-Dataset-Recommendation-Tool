#!------------------IMPORTS------------------!
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go 
from streamlit_utils import highlight_column
from database import get_datasets_safe
import json








#!------------------FUNCTIONALITY------------------!

"""Function to replace string duplicates in a list with another value (value + "_n" - where n is the numbered duplicate)"""
def avoid_duplicates(strings: list[str]) -> list[str]:
    """
    @param strings - List of strings for duplicates to be removed from

    @return unique_strings - List of given strings made distinct (unique) from one another
    """

    seen_strings = {}
    unique_strings = []
    

    for string in strings:


        if string in seen_strings:
            seen_strings[string] += 1
            new_string = f"{string}_{seen_strings[string]}"

        else:
            seen_strings[string] = 0
            new_string = string


        unique_strings.append(new_string)
    

    return unique_strings


"""
Behaviour for the "Similarity per Dataset" chart(s)

Called as a re-usable function as we've adapted to show the bar chart per method section

Function that uses DataFrame of algorithms and their scores to create a bar chart
"""
def create_ranked_bar_chart(df_similarity_rankings: dict[pd.DataFrame], y_var: str) -> st.plotly_chart:
    """
    @param df_similarity_rankings_sorted - Dictionary of DataFrames containing the similarity rankings of the Comparison Datasets
    @param y_var - Variable (column / feature) which the Datasets are sorted by

    @return ranked_bar_chart - Plotly bar chart visualising the similarity rankings
    """

    # Re-sort to be ascending
    df_similarity_rankings_sorted = df_similarity_rankings.sort_values(by=y_var, ascending=True)

    fig = go.Figure(
        go.Bar(
            y=df_similarity_rankings_sorted["Dataset"],
            x=df_similarity_rankings_sorted[y_var],
            orientation="h",
            marker=dict(color="#32ab60"),
            hovertemplate="<b>%{y}</b><br>Similarity: %{x}<extra></extra>",
        )
    )
    
    fig.update_layout(
        title=dict(
            text=f"Similarity per Dataset<br><sup style='color: #787878; font-weight:normal'>Best: {df_similarity_rankings_sorted['Dataset'][0]} ({float(df_similarity_rankings_sorted[y_var][0])} / 1.0)</sup>",
            font=dict(size=24),
        ),
        xaxis=dict(
            title="Similarity",
            range=[0, 1],
        ),
        template="plotly_white",
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                showactive=True,
                x=1,
                y=1.15,
                buttons=[
                    dict(
                        label="Full Axis",
                        method="relayout",
                        args=[{"xaxis.range": [0, 1]}]
                    ),
                    dict(
                        label="Limit Axis",
                        method="relayout",
                        args=[{"xaxis.range": [0, df_similarity_rankings_sorted[y_var].max()]}]
                    )
                ]
            )
        ]
    )

    ranked_bar_chart = st.plotly_chart(fig, key=y_var)

    
    return ranked_bar_chart


"""
Behaviour for the "Algorithm Rankings" section(s)

Called as a re-usable function as we've adapted to show algorithm ranking per method (P-norm, Component-feature, total similarity)

Function that uses DataFrame of algorithms to generate an algorithm ranking
"""
def rank_algorithms(df_similarity_rankings_sorted: dict[pd.DataFrame], performance_metric_dataframes: dict[pd.DataFrame], similarity_type: str, prefix: str) -> None:
    """
    @param df_similarity_rankings_sorted - Dictionary of DataFrames containing the sorted similarity rankings of the Comparison Datasets
    @param performance_metric_dataframes - Dictionary of DataFrames containing the algorithms (with their Performance Metrics) applied to each ranked dataset
    @param similarity_type - Type of similarity method used to create the ranking (PCA or Component-feature)
    @param prefix - Prefix to be applied to the ranking_var selection box to give it a unique identifier
    """
    
    st.subheader(":material/leaderboard: Algorithm Rankings", help=f"Algorithm rankings are displayed according to which similarity method is shown. Currently displaying **{similarity_type}**")

    st.write("Analyse which algorithms performed the best on other Datasets by Metrics, ordered by most similar, to least similar dataset.")


    for index, df_ranking in df_similarity_rankings_sorted["Dataset"].items():
        df_performance = performance_metric_dataframes[f"{df_ranking}_perform_metrics"]
        
        df_ranking_label = ""


        if index == 0:
            df_ranking_label = "*(most similar dataset)*"


        with st.expander(f"**{index+1}**\t-\t{df_ranking} {df_ranking_label}"):

            df_performance_ranked_styled = df_performance


            if not df_performance.empty:
                
                # Remove Category column from ordering selection
                performance_metrics = df_performance.columns.to_numpy()
                ranking_metrics = performance_metrics[~np.isin(performance_metrics, ["Category", "category"])]
            
                ranking_metric = st.selectbox("Rank by:", ranking_metrics, key=f"rank_box_{prefix}_{index}")

                st.warning("If the table has been order by user interation with a column to re-order the tables rows then the rank by functionality won't work (Check for arrows on columns)")


                if ranking_metric:                 
                    df_performance_ranked = df_performance.sort_values(by=ranking_metric, ascending=False)
                    df_performance_ranked = df_performance_ranked.reset_index(drop=True)

                    df_performance_ranked_styled = (
                        df_performance_ranked
                        .style
                        .apply(highlight_column(ranking_metric), axis=0)  # Apply column highlighting
                        .format(precision=3)  # Format to 2 decimal places
                    )

                else:
                    df_performance_ranked_styled = (
                        df_performance
                        .format(precision=3)  # Format to 2 decimal places
                    )


            st.dataframe(df_performance_ranked_styled)
        

"""Function to gather Database information (Datasets and algorithms)"""
def get_database_datasets():   
    """
    @return datasets_dict - Dictionary containing Key-Value pairs of Datasets names and their data
    @return algorithms_dict - Dictionary containing Key-Value pairs of algorithm names and their Performance Metrics
    """

    # Get all the Datasets and top algorithms from the Database at the top level to avoid repeating
    dataset_names_list, dataset_data_list, algorithms_list = get_datasets_safe()

    # Avoid duplicate names
    unique_dataset_names_list = avoid_duplicates(dataset_names_list)

    # JSON format requires data to be wrapped in []
    dataset_data_list = [dataset_dict if isinstance(dataset_dict, list) else [dataset_dict] for dataset_dict in dataset_data_list]

    # Create dict of Datasets using names as keys
    datasets_dict = {unique_dataset_names_list[index]: pd.read_json(json.dumps(dataset_data_list[index])) for index in range(0, len(unique_dataset_names_list))}

    # JSON format requires data to be wrapped in []
    algorithms_list = [algorithms if isinstance(algorithms, list) else [algorithms] for algorithms in algorithms_list]

    # Create dict of algorithms using names as keys
    algorithms_dict = {f"{unique_dataset_names_list[index]}_perform_metrics": pd.read_json(json.dumps(algorithms_list[index])) for index in range(0, len(unique_dataset_names_list))}


    return datasets_dict, algorithms_dict
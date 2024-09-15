''' This is the code for the bachelor thesis of Ritu Suhas Shetkar from University of Duisburg-Essen on the topic
    Visualizing information retrieval study results from Result Assessment Tool. The timeperiod for the completion of this thesis was 
    29.06.2024 to 20.09.2024.
    For this code to run there is a need to install the dependecies from requirements.txt
    There is a command that the user needs to run in the terminal that is 
     streamlit run script2.py
    A new browser window will be opened.
 '''

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

#The dataset used for the study is based on the data collected for the paper published in 2014
#All the functions needed for the add-on are here.
#The image path is local but can be edited according to the need. This is inserted keeping in mind the design aspect.


def read_and_fix(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        #Columns that are not needed and have redundant information are deleted
        df = df.drop(columns=['scale_name', 'idscale'])

        #All the blanks are filled with -1
        df['scale_value'] = df['scale_value'].fillna(-1)
    
    
    else:
        st.write('File not uploaded properly')
    
    return df

def create_binary_relevance(df):
    BinaryRelevance=df[df['scale_description']=='Is this website relevant?']
    BinaryRelevance.rename(columns={'scale_value':'Is this website relevant?'},inplace=True)
    BinaryRelevance= BinaryRelevance[~BinaryRelevance['Is this website relevant?'].isin(['not_reached', 'skipped',-1])]
    BinaryRelevance = BinaryRelevance.drop(columns=['scale_description'])
    BinaryRelevance = BinaryRelevance.reset_index(drop=True)
    BinaryRelevance['Is this website relevant?'] = BinaryRelevance['Is this website relevant?'].astype(int)
    BinaryRelevance['ID'] = BinaryRelevance.groupby(['search_task_idsearch_task', 'search_engine_name',
                                                      'idsearch_results', 'result_position']
    ).cumcount() + 1
    BinaryRelevance = BinaryRelevance[BinaryRelevance['ID'] == 1].drop(columns=['ID'])
    BinaryRelevance = BinaryRelevance.groupby(
        ['search_task_idsearch_task', 'search_engine_name', 'result_position']
    ).first().reset_index()

    return  BinaryRelevance

def create_graded_relevance(df):
    GradedRelevance=df[df['scale_description']=='Please rate how relevant this website is (where 4 is the best possible value).'
                       ]
    GradedRelevance.rename(
        columns={'scale_value':'Please rate how relevant this website is (where 4 is the best possible value).'},
        inplace=True)
    GradedRelevance= GradedRelevance[~GradedRelevance[
        'Please rate how relevant this website is (where 4 is the best possible value).'].isin(
        ['not_reached', 'skipped',-1])]
    GradedRelevance = GradedRelevance.drop(columns=['scale_description'])
    GradedRelevance = GradedRelevance.reset_index(drop=True)
    GradedRelevance['Please rate how relevant this website is (where 4 is the best possible value).'] = GradedRelevance[
        'Please rate how relevant this website is (where 4 is the best possible value).'].astype(int)
    GradedRelevance['ID'] = GradedRelevance.groupby(['search_task_idsearch_task', 'search_engine_name', 
                                                     'idsearch_results', 'result_position']
    ).cumcount() + 1
    GradedRelevance = GradedRelevance[GradedRelevance['ID'] == 1].drop(columns=['ID'])
    GradedRelevance = GradedRelevance.groupby(
        ['search_task_idsearch_task', 'search_engine_name', 'result_position']
    ).first().reset_index()

    return GradedRelevance

def calculate_totals_precision_fallout(BinaryRelevance):
    precision=BinaryRelevance.groupby(['search_engine_name'])['Is this website relevant?'].mean()
    total_Retrieved_Items=BinaryRelevance.groupby(['search_engine_name'])['Is this website relevant?'].count()
    total_queries=BinaryRelevance.groupby('search_engine_name')['search_task_idsearch_task'].nunique()
    Graph = pd.DataFrame({'search_engine_name': precision.index, 'Total Retrieved Results':total_Retrieved_Items,
                          'Total Queries':total_queries.values,'Precision': precision.values})
    Graph['Fall Out']=1-Graph['Precision']
    return Graph


def caclulate_Query_Cal(BinaryRelevance):
    unique_queries=BinaryRelevance['search_task_idsearch_task'].unique()

    QueryCal=BinaryRelevance.groupby(['search_task_idsearch_task','search_engine_name'])['Is this website relevant?'].count().reset_index()
    QueryCal=QueryCal.rename(columns={'Is this website relevant?': 'Results in Each Query'})

    relevance=BinaryRelevance.groupby(['search_task_idsearch_task','search_engine_name'])['Is this website relevant?'].apply(lambda x: (x == 1).sum()).reset_index()
    relevance.rename(columns={'Is this website relevant?':'Sum of Relevant Items per Query'},inplace=True)
    QueryCal = QueryCal.merge(relevance, on=['search_task_idsearch_task', 'search_engine_name'], how='left')
    return QueryCal


def calculate_recall(BinaryRelevance, Graph, QueryCal):
    unique_queries=BinaryRelevance['search_task_idsearch_task'].unique()
    for query in unique_queries:
        group=BinaryRelevance[BinaryRelevance['search_task_idsearch_task']==query]
        total_relevant_in_pool=len(group[group['Is this website relevant?']==1]['result_url'].unique())
        unique_search_engines=group['search_engine_name'].unique()
        for engine in unique_search_engines:
            relevant_results=len(group[(group['Is this website relevant?']==1)&(group['search_engine_name']==engine)])
            if relevant_results==0:
                recall_per_query=0
            else:
                recall_per_query=relevant_results/total_relevant_in_pool
            index = QueryCal[(QueryCal['search_engine_name'] == engine) & (QueryCal['search_task_idsearch_task'] == query)].index
            if not index.empty:
                QueryCal.loc[index, 'Recall per query'] = recall_per_query

    Graph['Recall']=QueryCal.groupby('search_engine_name')['Recall per query'].mean()
    return Graph, QueryCal


def calculate_F1_measure(Graph):
    Graph['F1-Measure']=(2*Graph['Recall']*Graph['Precision'])/(Graph['Precision']+Graph['Recall'])
    return Graph

def precision_at_k(group):
    relevant_count = group['Is this website relevant?'].cumsum()  
    total_count = range(1, len(group) + 1)  
    return relevant_count / total_count


def calculate_average_precision(group):
    relevant_group = group[group['Is this website relevant?'] == 1]
    if len(relevant_group) == 0:
        return 0 
    avg_precision = relevant_group['Precision @k'].sum() / len(relevant_group)
    return avg_precision

def calculate_mean_average_precision(QueryCal, BinaryRelevance, Graph):
    AveragePrecison = BinaryRelevance.groupby([
        'search_task_idsearch_task', 'search_engine_name']).apply(lambda x:calculate_average_precision(x)).reset_index(name='Average Precision')

    QueryCal = QueryCal.merge(AveragePrecison, on=['search_task_idsearch_task', 'search_engine_name'], how='left')
    Graph['Mean Average Precision']=QueryCal.groupby(['search_engine_name'])['Average Precision'].mean()

    return QueryCal, Graph

def dcg_at_k(group):
    relevant_count = group['DCG Part from each row'].cumsum()  
    return relevant_count 
def idcg_at_k(group):
    relevant_count = group['IDCG Part from each row'].cumsum() 
    return relevant_count 
def ndcg_at_k(group):
    relevant_count = group['DCG @k']/group['IDCG @k']  
    return relevant_count 
def calculate_ndcg(GradedRelevance, Graph,QueryCal):
    GradedRelevance['DCG Part from each row']=((
        2**GradedRelevance['Please rate how relevant this website is (where 4 is the best possible value).'])-1)/(np.log10(
        GradedRelevance['result_position']+1))
    GradedRelevance['Normalized Relevance'] = GradedRelevance.groupby(
        ['search_task_idsearch_task', 'search_engine_name']
    )['Please rate how relevant this website is (where 4 is the best possible value).'].transform(
        lambda x: sorted(x, reverse=True)
    )
    GradedRelevance['IDCG Part from each row']=((2**GradedRelevance['Normalized Relevance'])-1)/(np.log10(GradedRelevance['result_position']+1))
    GradedRelevance['DCG @k'] = GradedRelevance.groupby(['search_task_idsearch_task','search_engine_name']).apply(lambda x:dcg_at_k(x)).reset_index(
        level=[0, 1], drop=True)
    GradedRelevance['IDCG @k'] = GradedRelevance.groupby(['search_task_idsearch_task','search_engine_name']).apply(lambda x:idcg_at_k(x)).reset_index(
        level=[0, 1], drop=True)
    GradedRelevance['NDCG @k']=GradedRelevance.groupby(['search_task_idsearch_task','search_engine_name']).apply(lambda x: ndcg_at_k(x)).reset_index(
        level=[0, 1], drop=True)


    QueryCal['DCG Per Query']=GradedRelevance.groupby(['search_task_idsearch_task','search_engine_name'])['DCG Part from each row'].sum().reset_index(
        level=[0,1],drop=True)
    QueryCal['IDCG Per Query']=GradedRelevance.groupby(['search_task_idsearch_task','search_engine_name'])['IDCG Part from each row'].sum().reset_index(
        level=[0,1],drop=True)
    QueryCal['NDCG Per Query']=QueryCal['DCG Per Query']/QueryCal['IDCG Per Query']
    Graph['NDCG']=QueryCal.groupby('search_engine_name')['NDCG Per Query'].mean()
    return GradedRelevance, Graph, QueryCal


def first_relevant_position(group):
    relevant = group[group['Is this website relevant?'] == 1]
    if not relevant.empty:
        return relevant.iloc[0]['result_position']
    return None

def calculcate_mean_reciprocal_rank(QueryCal, BinaryRelevance, Graph):
    first_relevant = BinaryRelevance.groupby(['search_task_idsearch_task', 'search_engine_name']).apply(first_relevant_position).reset_index()

    
    first_relevant.columns = ['search_task_idsearch_task', 'search_engine_name', 'first_relevant_position']


    QueryCal = QueryCal.merge(first_relevant, left_on=['search_task_idsearch_task', 'search_engine_name'],
                            right_on=['search_task_idsearch_task', 'search_engine_name'], how='left')

    QueryCal['first_relevant_position'].fillna(0, inplace=True)
    QueryCal['Reciprocal Rank'] = QueryCal['first_relevant_position'].apply(lambda x: 1/x if x != 0 else 0)
    Graph['Mean Reciprocal Rank']=QueryCal.groupby(['search_engine_name'])['Reciprocal Rank'].mean()

    return QueryCal, Graph

def recall_at_k(BinaryRelevance):
    total_relevant_unique = (
    BinaryRelevance[BinaryRelevance['Is this website relevant?'] == 1]
    .groupby('search_task_idsearch_task')['result_url']
    .nunique()
    .reset_index(name='total_relevant_unique')
    )
    BinaryRelevance = BinaryRelevance.merge(total_relevant_unique, on='search_task_idsearch_task')
    BinaryRelevance = BinaryRelevance.sort_values(by=['search_task_idsearch_task', 'result_position'])
    BinaryRelevance['cumulative_relevant'] = (
        BinaryRelevance
        .groupby(['search_task_idsearch_task','search_engine_name'])
        .apply(lambda x: x['Is this website relevant?'].cumsum())
        .reset_index(level=[0,1], drop=True)
    )
    BinaryRelevance['Recall @k'] = BinaryRelevance['cumulative_relevant'] / BinaryRelevance['total_relevant_unique']
    BinaryRelevance.drop(columns=['total_relevant_unique', 'cumulative_relevant'], inplace=True)

    
    BinaryRelevance = BinaryRelevance.sort_values(by=['search_task_idsearch_task', 'search_engine_name'])
    return BinaryRelevance


def calculate_position_cal(BinaryRelevance, GradedRelevance):
    
    unique_search_engines = BinaryRelevance['search_engine_name'].unique()
    unique_positions = sorted(BinaryRelevance['result_position'].unique())
    combinations = [(se, rp) for se in unique_search_engines for rp in unique_positions]
    PositionsCal = pd.DataFrame(combinations, columns=['search_engine_name', 'result_position'])
    PositionsCal[['Precision @k', 'Recall @k']]=BinaryRelevance.groupby(['search_engine_name', 'result_position'])[[
        'Precision @k', 'Recall @k']].mean().reset_index(level=[0,1],drop=True)
    PositionsCal['NDCG @k']=GradedRelevance.groupby(['search_engine_name', 'result_position'])['NDCG @k'].mean().reset_index(level=[0,1],drop=True)
    return PositionsCal


def calculate_interpolated_precision(group, standard_recall):
    recall_precision_pairs = list(zip(group['Recall @k'], group['Precision @k']))
    recall_precision_pairs.sort()
    results = []
    for i in range(len(standard_recall) - 1):
        lower_bound = standard_recall[i]
        upper_bound = standard_recall[i + 1]
        precisions_in_range = [precision for recall, precision in recall_precision_pairs if lower_bound <= recall < upper_bound]
        max_precision = max(precisions_in_range) if precisions_in_range else 0
        results.append((group['search_task_idsearch_task'].iloc[0], group['search_engine_name'].iloc[0], lower_bound, max_precision))
    return pd.DataFrame(results, columns=['search_task_idsearch_task', 'search_engine_name', 'standard_recall', 'interpolated_precision'])

def interpolated_precision(BinaryRelevance):
    standard_recall = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    standard_recall_levels = pd.DataFrame(columns=['search_task_idsearch_task', 'search_engine_name', 'standard_recall', 'interpolated_precision'])
    for (query, engine), group in BinaryRelevance.groupby(['search_task_idsearch_task', 'search_engine_name']):
        standard_recall_levels = pd.concat([standard_recall_levels, calculate_interpolated_precision(group, standard_recall)], ignore_index=True)
    return standard_recall_levels


#Functions to Plot

def plot_precision(Graph):
    st.header('Precision per Search Engine')
    st.write('This graph describes the ratio of relevant results to all the results for obtained from each of the selected search engines.'
              'Ranking is not taken into consideration here.')
    x_labels = Graph.apply(lambda row: f"{row['search_engine_name']}<br> {row['Total Retrieved Results']}", axis=1)
    text_values = Graph['Precision'].apply(lambda x: f'{x:.4f}')
    fig = go.Figure(data=[
        go.Bar(name='Precision', x=x_labels, y=Graph['Precision'], marker_color='lightgrey', text=text_values, textposition='outside')
    ])
    fig.update_layout(title='Precision for Different Search Engines',
                    xaxis_title='Search Engine',
                        yaxis_title='Precision', 
                        width=500, 
                        height=600,
                        yaxis=dict(
                        range=[0, 1],  
                        dtick=0.1 )

                    )
    st.plotly_chart(fig)

def plot_recall(Graph):
    st.header('Recall (with Pooling) per Search Engine')
    st.write('This graph describes the ratio of the relevant results in a search engine to the total pooled relevant results. Ranking is not taken into consideration here.')
    x_labels = Graph.apply(lambda row: f"{row['search_engine_name']}<br> {row['Total Retrieved Results']}", axis=1)
    text_values = Graph['Recall'].apply(lambda x: f'{x:.4f}')
    fig = go.Figure(data=[
        go.Bar(name='Recall', x=x_labels, y=Graph['Recall'], marker_color='lightgrey', text=text_values, textposition='outside')
    ])
    fig.update_layout(title='Recall (with Pooling) for Different Search Engines',
                    xaxis_title='Search Engine',
                        yaxis_title='Recall',
                        width=500,
                        height=600,
                        yaxis=dict(
                        range=[0, 1],  
                        dtick=0.1
                        )
                    )
    st.plotly_chart(fig)

def plot_ndcg(Graph):
    st.header('Normalized Discounted Cumulative Gain (NDCG) per Search Engine')
    st.write('This graph describes the normalized discounted cumulatice gain for each search engine. Ranking is taken into consideration here.')
    x_labels = Graph.apply(lambda row: f"{row['search_engine_name']}<br> {row['Total Retrieved Results']}", axis=1)
    text_values = Graph['NDCG'].apply(lambda x: f'{x:.4f}')
    fig = go.Figure(data=[
        go.Bar(name='NDCG', x=x_labels, y=Graph['NDCG'], marker_color='lightgrey', text=text_values, textposition='outside')
    ])
    fig.update_layout(title='NDCG for Different Search Engines',
                    xaxis_title='Search Engine',
                        yaxis_title='NDCG',
                        width=500, 
                        height=600,
                        yaxis=dict(
                        range=[0, 1],  
                        dtick=0.1 
                        )
                    
                        )
    st.plotly_chart(fig)

def plot_f_measure(Graph):
    st.header('F1 Measure per Search Engine')
    st.write('This graph describes the F measure for the search engines, where precision and recall are weighted equally, therefore a balanced F Measure or F1 Measure is calculated.')
    x_labels = Graph.apply(lambda row: f"{row['search_engine_name']}<br> {row['Total Retrieved Results']}", axis=1)
    text_values = Graph['F1-Measure'].apply(lambda x: f'{x:.4f}')
    fig = go.Figure(data=[
        go.Bar(name='F1-Measure', x=x_labels, y=Graph['F1-Measure'], marker_color='lightgrey', text=text_values, textposition='outside')
    ])
    fig.update_layout(title='F1-Measure for Different Search Engines',
                        xaxis_title='Search Engine',
                        yaxis_title='F1-Measure',
                        width=500,
                        height=600,
                        yaxis=dict(
                        range=[0, 1], 
                        dtick=0.1 
                        )
                    ) 
    st.plotly_chart(fig)

def plot_fall_out(Graph):
    st.header('Fallout per Search Engine')
    st.write('This graph describes the ratio of non-relevant results retrieved for the search engines. Ranking is not taking into consideration here.')
    x_labels = Graph.apply(lambda row: f"{row['search_engine_name']}<br> {row['Total Retrieved Results']}", axis=1)
    text_values = Graph['Fall Out'].apply(lambda x: f'{x:.4f}')
    fig = go.Figure(data=[
        go.Bar(name='Fall Out', x=x_labels, y=Graph['Fall Out'], marker_color='lightgrey', text=text_values, textposition='outside')
    ])
    fig.update_layout(title='Fallout for Different Search Engines',
                    xaxis_title='Search Engine',
                        yaxis_title='Fallout', 
                        width=500, 
                        height=600,
                        yaxis=dict(
                        range=[0, 1],  
                        dtick=0.1
                        )
                        )
    st.plotly_chart(fig)

def plot_map(Graph):
    st.header('Mean Average Precision per Search Engine')
    x_labels = Graph.apply(lambda row: f"{row['search_engine_name']}<br> {row['Total Retrieved Results']}", axis=1)
    st.write('This graph describes the average precision over all the queries relaying information about how effective is the ranking of relevant items over all the queries.')
    text_values = Graph['Mean Average Precision'].apply(lambda x: f'{x:.4f}')
    fig = go.Figure(data=[
        go.Bar(name='Mean Average Precision', x=x_labels, y=Graph['Mean Average Precision'], marker_color='lightgrey', text=text_values, textposition='outside')
    ])
    fig.update_layout(title='Mean Average Precision for Different Search Engines', 
                        xaxis_title='Search Engine', 
                        yaxis_title='Mean Average Precision',
                        width=500,
                        height=600,
                        yaxis=dict(
                        range=[0, 1],  
                        dtick=0.1
                        )
                    

                    )
    st.plotly_chart(fig)

def plot_mrr(Graph):
    st.header('Mean Reciprocal Rank per Search Engine')
    st.write('This graph describes the reciprocal rank averaged over all the queries relaying the information about how quickly the first relevant result is retrieved per query.')
    x_labels = Graph.apply(lambda row: f"{row['search_engine_name']}<br> {row['Total Retrieved Results']}", axis=1)
    text_values = Graph['Mean Reciprocal Rank'].apply(lambda x: f'{x:.4f}')
    fig = go.Figure(data=[
        go.Bar(name='Mean Reciprocal Rank', x=x_labels, y=Graph['Mean Reciprocal Rank'], marker_color='lightgrey', text=text_values, textposition='outside')
    ])
    fig.update_layout(title='Mean Reciprocal Rank for Different Search Engines', 
                        xaxis_title='Search Engine', 
                        yaxis_title='Mean Reciprocal Rank',
                        width=500,
                        height=600,
                        yaxis=dict(
                        range=[0, 1],  
                        dtick=0.1
                        )
                    

                    )
    st.plotly_chart(fig)

def plot_combined_metrics(Graph):
    st.header('Combined Metrics per Search Engine ')
    st.write('This graph shows the different measures like Precision, Recall, NDCG, F1 Measure, Mean Average Precision, Fallout,'
             'Mean Reciprocal Rank in a single graph for quick insights.')
    fig = go.Figure()
    x_labels = Graph.apply(lambda row: f"{row['search_engine_name']}<br> {row['Total Retrieved Results']}", axis=1)
    metrics = ['Precision', 'Recall', 'NDCG', 'F1-Measure', 'Mean Average Precision', 'Fall Out','Mean Reciprocal Rank']
    patterns = ['/', '|', 'x', '-', '\\', '+', '.']  
    for metric, pattern in zip(metrics, patterns):
        fig.add_trace(go.Bar(
            name=metric,
            x=x_labels,
            y=Graph[metric],
            marker=dict(
                color='lightgrey',
                pattern_shape=pattern
            ),
            text=Graph[metric].round(3),  
            textposition='outside'
        ))
    fig.update_layout(
        title='Combined Metrics for Different Search Engines',
        xaxis_title='Search Engine',
        yaxis_title='Value',
        barmode='group',
        width=1000,
        height=800,
        template='plotly_white',  
        plot_bgcolor='rgba(0,0,0,0)',  
        paper_bgcolor='white',
        yaxis=dict(
                        range=[0, 1],  
                        dtick=0.1
                        ) 
    )
    st.plotly_chart(fig)

def plot_graded_scale_result_position(PositionsCal, GradedRelevance):
    st.header('Average relevance judgements on the Likert scale')
    st.write('The line graph shows the average of the relevance judgement from Likert scale for every rank averaged over all the queries for the search engines.')
    PositionsCal['Graded Relevance Average']=GradedRelevance.groupby(['search_engine_name','result_position'])[
        'Please rate how relevant this website is (where 4 is the best possible value).'].mean().reset_index(level=[0,1],drop=True)
    fig = go.Figure()
    unique_engines = PositionsCal['search_engine_name'].unique()
    line_styles = [
        dict(color='black', width=3),  # thick solid
        dict(color='black', width=1),  # thin solid
        dict(color='black', dash='dash', width=3),  # thick dashed
        dict(color='black', dash='dash', width=1),  # thin dashed
        dict(color='black', dash='dot', width=3),  # thick dotted
        dict(color='black', dash='dot', width=1),  # thin dotted
        dict(color='black', dash='dashdot', width=3),  # thick dashdot
        dict(color='black', dash='dashdot', width=1)   # thin dashdot
    ]
    for i, engine_name in enumerate(unique_engines):
        subset = PositionsCal[PositionsCal['search_engine_name'] == engine_name]
    

        fig.add_trace(go.Scatter(
            x=subset['result_position'],
            y=subset['Graded Relevance Average'],
            mode='lines+markers',
            name=engine_name,
            line=line_styles[i % len(line_styles)],
            marker=dict(color='black')
        ))

    fig.update_layout(
        title='Average relevance judgements on the Likert scale',
        xaxis_title='Postion (Cumulative)',
        yaxis_title='Average of the judgements from 5 point scale',
        legend_title='Search Engine',
        width=1000,
        height=600,
        yaxis=dict(
                        range=[0, 4], 
                        dtick=0.5
                        ),
        xaxis=dict(
                    
                        dtick=1
                        ),  
        template='plotly_white'
    )
    #t.plotly_chart(fig)
    st.plotly_chart(fig)

def plot_precision_result_position(PositionsCal):
    st.header('Recall-Precision (Precision @k to Rank)')
    st.write('This line graph shows the progression precision at rank k averaged over the queries for all the search engines.')
    fig = go.Figure()
    unique_engines = PositionsCal['search_engine_name'].unique()
    line_styles = [
        dict(color='black', width=3),  # thick solid
        dict(color='black', width=1),  # thin solid
        dict(color='black', dash='dash', width=3),  # thick dashed
        dict(color='black', dash='dash', width=1),  # thin dashed
        dict(color='black', dash='dot', width=3),  # thick dotted
        dict(color='black', dash='dot', width=1),  # thin dotted
        dict(color='black', dash='dashdot', width=3),  # thick dashdot
        dict(color='black', dash='dashdot', width=1)   # thin dashdot
    ]
    for i, engine_name in enumerate(unique_engines):
        subset = PositionsCal[PositionsCal['search_engine_name'] == engine_name]
    

        fig.add_trace(go.Scatter(
            x=subset['result_position'],
            y=subset['Precision @k'],
            mode='lines+markers',
            name=engine_name,
            line=line_styles[i % len(line_styles)],
            marker=dict(color='black')
        ))

    fig.update_layout(
        title='Recall-Precision (Precision @k to Rank)',
        xaxis_title='Postion (Cumulative)',
        yaxis_title='Precision @k',
        legend_title='Search Engine',
        width=1000,
        height=600,
        yaxis=dict(
                        range=[0, 1], 
                        dtick=0.1
                        ),
        xaxis=dict(
                    
                        dtick=1
                        ),
        template='plotly_white'
    )
    
    st.plotly_chart(fig)

def plot_graded_scale_percentage_distribution(GradedRelevance):
    st.header('Distribution of the Likert-scale relevance judgments for the search engines')
    st.write('This graph shows the proportion of each relevance judgement on the Likert-scale for the search engines.')
  
    grouped = GradedRelevance.groupby(['search_engine_name', 'Please rate how relevant this website is (where 4 is the best possible value).']).size().reset_index(name='count')
    patterns = ['/', '|', 'x', '-', '\\', '+', '.']  # Patterns for each metric

    total_counts = GradedRelevance.groupby('search_engine_name').size().reset_index(name='total_count')

    merged = pd.merge(grouped, total_counts, on='search_engine_name')

    merged['percentage'] = (merged['count'] / merged['total_count']) * 100

    
    merged = merged.sort_values(by=['search_engine_name', 'Please rate how relevant this website is (where 4 is the best possible value).'])

    merged['search_engine_name'].unique()

    fig = go.Figure()

    
    unique_search_engines = merged['search_engine_name'].unique()


    for i, engine_name in enumerate(unique_search_engines):
        data_to_plot = merged[merged['search_engine_name'] == engine_name]
        text_values = data_to_plot['percentage'].apply(lambda x: f'{x:.2f}')
        fig.add_trace(go.Bar(
            x=data_to_plot['Please rate how relevant this website is (where 4 is the best possible value).'],
            y=data_to_plot['percentage'],
            name=engine_name,
            text=text_values,
            textposition='outside',
            marker=dict(
                pattern_shape=patterns[i % len(patterns)],  
                color='lightgray' 
                
            )
        ))


    fig.update_layout(
        title='Distribution of the Likert-scale relevance judgements for the search engines',
        xaxis_title='Relevance Judgements',
        yaxis_title='Percentage (%)',
        barmode='group',
        xaxis=dict(tickmode='linear'), 
        showlegend=True,
        yaxis=dict(
                        range=[0, 100],  
                        dtick=10
                        ),
        template='plotly_white'
    )

    # Show the plot
    st.plotly_chart(fig)

def plot_binary_scale_percentage_distribution(BinaryRelevance):
    st.header('Distribution of the Binary Relevance judgments for the search engines')
    st.write('This graph shows the proportion of each relevance judgement on the Binary Scale for the search engines.')
    patterns = ['/', '|', 'x', '-', '\\', '+', '.']  # Patterns for each metric
    
    grouped = BinaryRelevance.groupby(['search_engine_name', 'Is this website relevant?']).size().reset_index(name='count')
    total_counts = BinaryRelevance.groupby('search_engine_name').size().reset_index(name='total_count')
    merged = pd.merge(grouped, total_counts, on='search_engine_name')
    merged['percentage'] = (merged['count'] / merged['total_count']) * 100
    merged = merged.sort_values(by=['search_engine_name', 'Is this website relevant?'])
    merged['search_engine_name'].unique()

    fig = go.Figure()


    unique_search_engines = merged['search_engine_name'].unique()
    for i, engine_name in enumerate(unique_search_engines):
        data_to_plot = merged[merged['search_engine_name'] == engine_name]
        text_values = data_to_plot['percentage'].apply(lambda x: f'{x:.2f}')
        fig.add_trace(go.Bar(
            x=data_to_plot['Is this website relevant?'],
            y=data_to_plot['percentage'],
            name=engine_name,
            text=text_values,
            textposition='outside',
            marker=dict(
                pattern_shape=patterns[i % len(patterns)], 
                color='lightgray'
            )
        ))


    fig.update_layout(
        title='Distribution of the Binary relevance judgements for search engines',
        xaxis_title='Relevance Judgements',
        yaxis_title='Percentage (%)',
        barmode='group',
        xaxis=dict(tickmode='linear'), 
        showlegend=True,
        yaxis=dict(
                        range=[0, 100], 
                        dtick=10
                        ),
        template='plotly_white'
    )


    st.plotly_chart(fig)


def main():
        image_path = "/Users/ritushetkar/Desktop/Thesis/rat_logo_grau.png"
        st.image(image_path, use_column_width=True)
        st.title("Visualizing the IR Results from RAT")
        st.header("An add-on for RAT")

        #Getting the csv file 
        st.title('Upload your csv file from RAT results here.')
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        #uploaded_file = "/Users/ritushetkar/Thesis New/RAT-Results.csv"

        if uploaded_file is not None:
            try:

                df=read_and_fix(uploaded_file)
                st.write('File is being read. Please wait.')
                BinaryRelevance=create_binary_relevance(df)
                GradedRelevance=create_graded_relevance(df)

                Graph=calculate_totals_precision_fallout(BinaryRelevance)
                QueryCal=caclulate_Query_Cal(BinaryRelevance)

                Graph, QueryCal=calculate_recall(BinaryRelevance, Graph, QueryCal)
                Graph=calculate_F1_measure(Graph)

                BinaryRelevance['Precision @k'] = BinaryRelevance.groupby(['search_task_idsearch_task', 
                                                                           'search_engine_name']).apply(lambda x: precision_at_k(x)).reset_index(level=[0, 1], drop=True)

                QueryCal, Graph=calculate_mean_average_precision(QueryCal, BinaryRelevance, Graph)
                GradedRelevance, Graph, QueryCal=calculate_ndcg(GradedRelevance, Graph,QueryCal)
                QueryCal, Graph=calculcate_mean_reciprocal_rank(QueryCal, BinaryRelevance, Graph)

                BinaryRelevance=recall_at_k(BinaryRelevance)
                PositionsCal=calculate_position_cal(BinaryRelevance, GradedRelevance)

                options = st.multiselect(
                    'Select the graphs you want to see:',
                    ['All Graphs',
                    'Precision', 
                    'Recall', 
                    'NDCG',
                    'F1-Measure',
                    'Fall Out',
                    'Mean Average Precision',
                    'Mean Reciprocal Rank',
                    'All Metrics combined',
                    'Average relevance judgements in the Likert Scale',
                    'Distribution of the Likert-scale relevance judgements search engines',
                    'Distribution of the Binary relevance judgements for search engines',
                    'Recall-Precision (Precision @k to Rank)'
                    
                    ]
                )




                # Display selected graphs
                if 'Precision' in options:
                    plot_precision(Graph)

                if 'Recall' in options:
                    plot_recall(Graph)

                if 'NDCG' in options:
                    plot_ndcg(Graph)

                if 'Fall Out' in options:
                    plot_fall_out(Graph)

                if 'Mean Average Precision' in options:
                    plot_map(Graph)
                if 'F1-Measure' in options:
                    plot_f_measure(Graph)
                    
                if 'All Metrics combined' in options:
                    plot_combined_metrics(Graph)

                if 'Mean Reciprocal Rank' in options:
                    plot_mrr(Graph)

                if 'Average relevance judgements in the Likert Scale' in options:
                    plot_graded_scale_result_position(PositionsCal, GradedRelevance)

                if 'Distribution of the Likert-scale relevance judgements search engines' in options:
                    plot_graded_scale_percentage_distribution(GradedRelevance)

                if 'Recall-Precision (Precision @k to Rank)' in options:
                    plot_precision_result_position(PositionsCal)

                if 'Distribution of the Binary relevance judgements for search engines' in options:
                    plot_binary_scale_percentage_distribution(BinaryRelevance)
                  

                if 'All Graphs' in options:
                    plot_precision(Graph)
                    plot_recall(Graph)
                    plot_fall_out(Graph)
                    plot_ndcg(Graph)
                    plot_map(Graph)
                    plot_f_measure(Graph)
                    plot_mrr(Graph)
                    plot_combined_metrics(Graph)
                    plot_graded_scale_result_position(PositionsCal, GradedRelevance)
                    plot_graded_scale_percentage_distribution(GradedRelevance)
                    plot_binary_scale_percentage_distribution(BinaryRelevance)
                    plot_precision_result_position(PositionsCal)
                
            except Exception as e:
                st.error(f"Error processing file: {e}")
        
        else:
            st.info("Please upload a CSV file.")
        

if __name__=="__main__":
    main()

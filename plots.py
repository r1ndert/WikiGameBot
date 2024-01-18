import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def line_plot(game_csv):
    fig = px.line(game_csv, x='turn', y='similarity_to_target', markers=True,
                  title='Line Plot of Similarity Over Time',
                  labels={'turn': 'Wiki Page Number', 'similarity_to_target': 'Similarity to Target'})

    fig.update_traces(hovertemplate='Wiki Page Number: %{x}<br>Similarity: %{y}<br>Page Title: %{customdata[0]}<br>Turn Time (seconds): %{customdata[1]}<extra></extra>',
                      customdata=game_csv[['current_topic', 'turn_time']])

    # Update the layout for the title and the background colors
    fig.update_layout({
        # 'plot_bgcolor': 'rgba(0, 0, 0, 0)',  # Transparent background
        # 'paper_bgcolor': 'rgba(200, 200, 200, 1)',  # Light grey background
        'title': {
            'text': 'Similarity Over Time',
            'x': 0.5,  # Centers the title
            'xanchor': 'center'
        }
    })

    st.plotly_chart(fig)  # Use Streamlit's function to display Plotly chart

def plot_embeddings(game_csv):
    embeddings = np.vstack(game_csv['embedding'].apply(pd.Series))
    print(embeddings.shape)
    print(embeddings[0].shape)

    # Check if the embeddings have the minimum required shape
    if embeddings.shape[0] > 1 and embeddings.shape[1] > 1:
        # Dimensionality Reduction
        tsne = TSNE(n_components=2, random_state=0)
        reduced_embeddings = tsne.fit_transform(embeddings)

        # Plotting with Matplotlib
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], label='Topics')

        # Dimensionality Reduction
        tsne = TSNE(n_components=2, random_state=0)
        reduced_embeddings = tsne.fit_transform(embeddings)

        # Plotting with Matplotlib
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], label='Topics')

        # Highlight the first and last topic
        topics = game_csv['topic'].tolist()
        ax.scatter(reduced_embeddings[0, 0], reduced_embeddings[0, 1], color='red', label='First Topic')
        ax.scatter(reduced_embeddings[-1, 0], reduced_embeddings[-1, 1], color='green', label='Last Topic')

        # Annotate all points
        for i, word in enumerate(topics):
            ax.annotate(word, xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

        ax.legend()

        st.pyplot(fig)  # Use Streamlit's function to display Matplotlib figure

def plot_topic_clusters(game_csv):

    # Assuming 'game_csv' is your pandas DataFrame, 'embedding' and 'turn' are the columns
    embeddings = pd.DataFrame(game_csv['embedding'].tolist())

    # Function to calculate the optimal number of clusters
    def calculate_optimal_k(sse, K):
        # Calculate the second derivative
        second_derivative = np.diff(sse, 2)
        optimal_k = K[np.argmax(second_derivative) + 1]  # +1 due to the difference in indices
        return optimal_k

    # Elbow Method to find the optimal number of clusters
    sse = []
    max_clusters = min(len(embeddings), 10)  # Ensure max clusters do not exceed number of samples
    K = range(1, max_clusters + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        sse.append(kmeans.inertia_)

    optimal_k = calculate_optimal_k(sse, np.array(K))

    # Ensure optimal_k is not greater than the number of samples
    optimal_k = min(optimal_k, len(embeddings))

    # K-Means Clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    game_csv['topic'] = kmeans.fit_predict(embeddings)

    # # Plotting the clusters
    # plt.figure(figsize=(10, 6))
    # for topic in range(optimal_k):
    #     subset = game_csv[game_csv['topic'] == topic]
    #     plt.plot(subset['turn'], [topic] * len(subset), 'o', label=f'Topic {topic}')

    # plt.xlabel('Turn')
    # plt.ylabel('Topic')
    # plt.title('Topic Distribution Over Turns')
    # plt.legend()

    # # Display the plot in Streamlit
    # st.pyplot(plt)
    plot_game_csv = game_csv.groupby(['turn', 'topic']).size().reset_index(name='count')

    fig = px.line(plot_game_csv, x='turn', y='topic', markers=True, 
                  title='Topic Selection Over Time',
                  labels={'turn': 'Turn', 'topic': 'Topic'})

    fig.update_traces(hovertemplate='Turn: %{x}<br>Topic: %{y}<br>Count: %{customdata}<extra></extra>',
                      customdata=plot_game_csv['count'])

    # Update the layout for the title and the background colors
    fig.update_layout({
        'title': {
            'text': 'Topic Distribution Over Turns',
            'x': 0.5,  # Centers the title
            'xanchor': 'center'
        }
    })

    st.plotly_chart(fig)  # Use Streamlit's function to display Plotly chart
import streamlit as st
import plotly.express as px
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

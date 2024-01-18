import plotly.express as px
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def line_plot(game_csv):
    fig = px.line(game_csv, x = 'turn', y = 'similarity_to_target', markers = True,
                title = 'Line Plot of Similarity Over Time',
                labels = {'turn': 'Wiki Page Number', 'similarity_to_target': 'Similarity to Target'})

    # Customize tooltip content
    fig.update_traces(hovertemplate='Wiki Page Number: %{x}<br>Similarity: %{y}<br>Page Title: %{customdata[0]}<br>Turn Time (seconds): %{customdata[1]}<extra></extra>',
                    customdata = game_csv[['current_topic', 'turn_time']])

    # Update layout for background color
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',  # Transparent background
        'paper_bgcolor': 'rgba(200, 200, 200, 1)',  # Light grey background
    })

    # Show the figure
    fig.show()

def plot_embeddings(game_csv):
    embeddings = np.vstack(game_csv['embedding'].apply(pd.Series))

    # Dimensionality Reduction
    tsne = TSNE(n_components=2, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], label='Topics')  # Regular points

    # Highlight the first and last topic
    topics = game_csv['topic'].tolist()
    plt.scatter(reduced_embeddings[0, 0], reduced_embeddings[0, 1], color='red', label='First Topic')  # First point
    plt.scatter(reduced_embeddings[-1, 0], reduced_embeddings[-1, 1], color='green', label='Last Topic')  # Last point

    # Annotate all points
    for i, word in enumerate(topics):
        plt.annotate(word, xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

    # Adding legends
    plt.legend()

    plt.show()
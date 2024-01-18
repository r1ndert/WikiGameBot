# WikiGameBot

## Overview

WikiGameBot is a web application that lets you see a computer play the WikiGame, where the objective is to navigate from a starting Wikipedia topic to a target topic using as few turns as possible, relying solely on hyperlinks available on the current page for navigation.

## How it Works

### Embeddings Model

WikiGameBot utilizes HuggingFace's SentenceTransformer model, specifically 'sentence-transformers/all-MiniLM-L6-v2,' for generating embeddings of Wikipedia summaries. This model is employed to represent Wikipedia page content as high-dimensional vectors.

### Cosine Similarity

To determine the next page to navigate to, WikiGameBot calculates the cosine similarity between the embedding of the target summary and the embeddings of summaries on the current page. The page with the summary that is most similar to the target summary is chosen for the next move.

## Learn More

To dive deeper into how WikiGameBot was developed and explore the Python code behind it, check out the detailed article: How I Made a Bot That Can Play the Wiki Game (Python Code Included).

Feel free to explore the code, contribute, or provide feedback to enhance this project further!

Enjoy the WikiGame adventure!

## Contact

You can reach out to me via one of the following ways to connect about this project or anything else.

Email: kmaurinjones@gmail.com

LinkedIn: https://www.linkedin.com/in/kmaurinjones/
import random
import time
import string
# !pip install Wikipedia-API -q
import wikipediaapi
import streamlit as st
from IPython.display import clear_output
from funcs import *

class WikiGameBot():
    """
    A class to simulate playing the 'Wiki Game', an online game where players start from one Wikipedia article 
    and try to navigate to another target article using as few hyperlinks as possible within Wikipedia.

    This class connects to the Wikipedia API to fetch and analyze Wikipedia pages. It keeps track of the 
    game's progress, including the start and target topics, the current topic, turn number, time taken per turn, 
    and the similarity between the current and target topics.

    Attributes
    ----------
    wiki_wiki : WikipediaAPI
        The Wikipedia API connection instance.
    game_log : dict
        A dictionary to log the game's progress, including the start and target topics, turn number, 
        time taken per turn, current topic and summary, and similarity to the target.
    start_topic : str
        The starting topic of the game.
    target_topic : str
        The target topic of the game.
    current_summary : str
        The summary of the current topic.
    target_summary : str
        The summary of the target topic.

    Methods
    -------
    log_turn(turn_dict)
        Logs the details of a turn in the Wiki game.
    get_topics(start_topic=None, target_topic=None)
        Retrieves and sets the start and target topics for the Wiki game.
    take_turn(current_topic, visited)
        Processes a turn in the Wiki game, finding the most similar topic to the target on the current page.
    play_game(verbose=True, clear_cell=False)
        Starts and manages the Wiki game until the target topic is reached.

    Raises
    ------
    AssertionError
        If the start and target topics are the same, an assertion error is raised.
    """
    def __init__(self, wiki_wiki, start_topic = None, target_topic = None):
        """
        Initializes the WikiGameBot with the start and target topics.

        This constructor establishes a connection to the Wikipedia API, generates a unique user agent for 
        the session, and initializes the game log. It also retrieves the summaries for the start and target 
        topics, ensuring that they are not the same.

        Parameters
        ----------
        start_topic : str, optional
            The starting topic for the Wiki game. If not provided, a random Wikipedia page is chosen.
        
        target_topic : str, optional
            The target topic for the Wiki game. If not provided, a random Wikipedia page is chosen.

        Raises
        ------
        AssertionError
            If the start and target topics are the same, an assertion error is raised.
        """

        self.wiki_wiki = wiki_wiki
        
        # game log
        self.game_log = {
            'starting_topic': [],   # to be able to calculate distance from starting topic to current topic
            'target_topic': [],     # to be able to calculate distance from target topic to current topic
            'turn': [],             # int of this turn (so things are sortable)
            'turn_time': [],        # time in seconds to perform steps at this turn
            'current_topic': [],    # current topic
            'current_summary': [],  # summary for current topic
            'similarity_to_target': [],  # similarity to target
        }

        # get start and target topics and their respective summaries
        self.get_topics(start_topic, target_topic) # defines self.start_topic and self.target_topic
        assert self.start_topic != self.target_topic, "Please enter different start and target topics."
        
        target_page = self.wiki_wiki.page(self.target_topic)
        self.current_summary = "" # this is overwritten after game starts
        self.target_summary = get_page_summary(target_page)

    def log_turn(self, turn_dict):
        """
        Logs the details of a turn in the Wiki game.

        This method updates the game log with details from the current turn, such as the turn number, time 
        taken, current topic, and similarity to the target topic.

        Parameters
        ----------
        turn_dict : dict
            A dictionary containing the details of the current turn to be logged.
        """
        for key, val in turn_dict.items():
            self.game_log[key].append(val)

    def get_topics(self, start_topic = None, target_topic = None):
        """
        Retrieves and sets the start and target topics for the Wiki game.

        This method obtains the URLs and Wikipedia page names for the start and target topics. If the topics 
        are not specified, random Wikipedia pages are chosen.

        Parameters
        ----------
        start_topic : str, optional
            The starting topic for the Wiki game.
        
        target_topic : str, optional
            The target topic for the Wiki game.
        """
        if start_topic:
            self.start_topic = search_wiki(start_topic)
        else:
            self.start_topic = get_random_wiki_page(self.wiki_wiki)
        if target_topic:
            self.target_topic = search_wiki(target_topic)
        else:
            self.target_topic = get_random_wiki_page(self.wiki_wiki)

    def take_turn(self, current_topic, visited):
        """
        Processes a turn in the Wiki game.

        This method finds the most similar topic to the target on the current page. It returns the topic 
        whose summary is most similar to the target summary.

        Parameters
        ----------
        current_topic : str
            The current topic being analyzed in the game.

        visited : list
            A list of already visited topics to avoid repetition.

        Returns
        -------
        tuple
            A tuple containing the most similar topic to the target and its similarity score.
        """
        # get API page object for topic
        current_page = self.wiki_wiki.page(current_topic)
        self.current_summary = get_page_summary(current_page) # not the best but this works

        # get top n valid pages from all linked pages
        top_n = 20
        pages = [page for page in validate_pages(current_page) if page not in visited]
        if self.target_topic in pages:
            return self.target_topic, 1
        top_n_pages, _ = get_most_similar_strings(self.target_summary, pages, n = top_n)

        # get the summary of top_n // 2 pages and get the most similar summary to target summary
        top_n_summaries = {get_page_summary(self.wiki_wiki.page(page)).strip() : page for page in top_n_pages[: top_n // 2] if get_page_summary(self.wiki_wiki.page(page)).strip()}
        top_n_pages, top_n_similaries = get_most_similar_strings(self.target_summary, list(top_n_summaries.keys()), n = top_n)
        most_similar_topic, similarity_to_target = top_n_summaries[top_n_pages[0]], top_n_similaries[0]

        return most_similar_topic, similarity_to_target # return page topic whose summary that is most similar to target summary
    
    def play_game(self, verbose = True, clear_cell = False):
        """
        Starts and manages the Wiki game until the target topic is reached.

        This method keeps track of the game, including turn number, time, and similarity scores. It outputs 
        the game progress if verbose is True. The game continues until the target topic is reached.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints detailed information about each turn. Defaults to True.

        clear_cell : bool, optional
            If True, clears the output cell every four turns (useful in interactive environments). Defaults to False.
        """
        # turn number
        turn_num = 1

        # first 'current' topic is starting topic
        current_topic = self.start_topic

        # to prevent duplicates
        visited = set()

        # keep playing until target is reached
        while True:

            # if same as target topic, game is done
            if current_topic.lower().strip() == self.target_topic.replace('_', ' ').lower().strip():
                # print(f"Congratulations! WikiGameBot has finished the game in {turn_num} turns, in {round(sum(self.game_log['turn_time']), 2)} seconds!")
                # print(f"Average topic similarity was {round(sum(self.game_log['similarity_to_target']) / len(self.game_log['similarity_to_target']), 2)}.")
                st.write(f"Congratulations! WikiGameBot has finished the game in {turn_num} turns, in {round(sum(self.game_log['turn_time']), 2)} seconds!")
                st.write(f"Average topic similarity was {round(sum(self.game_log['similarity_to_target']) / len(self.game_log['similarity_to_target']), 2)}.")
                break

            # for turn time tracking
            turn_start = time.time()

            # find most similar topic on current page to target topic
            visited.add(current_topic)
            next_topic, similarity_to_target = self.take_turn(current_topic, list(visited))

            # for turn time tracking
            turn_time = time.time() - turn_start

            self.log_turn(
                {
                    'starting_topic': self.start_topic,
                    'target_topic': self.start_topic,
                    'turn': turn_num,             
                    'current_topic': current_topic,
                    'current_summary': self.current_summary,
                    'similarity_to_target': similarity_to_target,
                    'turn_time': round(turn_time, 2),
                }
            )

            if verbose:
                if clear_cell:
                    if turn_num % 4 == 0:
                        clear_output(wait=True)
                # print("-" * 50)
                # print(f"Turn: {turn_num}")
                # print(f"Turn time: {round(turn_time, 2)}s")
                # print(f"Total time: {round(sum(self.game_log['turn_time']), 2)}s")
                # print(f"Start topic: {self.start_topic.replace('_', ' ')}")
                # print(f"Current topic: {current_topic.replace('_', ' ')}")
                # print(f"Next topic: {next_topic.replace('_', ' ')}")
                # print(f"Target topic: {self.target_topic.replace('_', ' ')}")
                # print(f"Current similarity to target: {round(similarity_to_target, 2)}") # rounded to 2 digits to avoid gross printouts
                # print("-" * 50)
                st.write("-" * 50)
                st.write(f"Turn: {turn_num}")
                st.write(f"Turn time: {round(turn_time, 2)}s")
                st.write(f"Total time: {round(sum(self.game_log['turn_time']), 2)}s")
                st.write(f"Start topic: {self.start_topic.replace('_', ' ')}")
                st.write(f"Current topic: {current_topic.replace('_', ' ')}")
                st.write(f"Next topic: {next_topic.replace('_', ' ')}")
                st.write(f"Target topic: {self.target_topic.replace('_', ' ')}")
                st.write(f"Current similarity to target: {round(similarity_to_target, 2)}") # rounded to 2 digits to avoid gross printouts
                st.write("-" * 50)

            # else, set new next_topic to current topic and loop
            current_topic = next_topic

            # increment turn
            turn_num += 1

"""
Example usage:
start_topic = "test"
target_topic = "christmas"
print(f"Starting topic: '{start_topic}'")
print(f"Target topic: '{target_topic}'")
game_bot = WikiGameBot(start_topic, target_topic)
game_bot.play_game(verbose = True, clear_cell = False)
"""
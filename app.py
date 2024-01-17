import streamlit as st
from funcs import *
from bot import *

# doing this first to ensure things will work
def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

# connects to Wikipedia API and defines game log
random_string = generate_random_string(10)
wiki_wiki = wikipediaapi.Wikipedia(
    f'WikiBot-{random_string} (https://www.linkedin.com/in/kmaurinjones/)',
    'en',
    timeout = 30
    )

# Setting the page title
st.set_page_config(page_title = "WikiGameBot")

# Title of the web app
st.title("WikiGameBot")

# User input for Start Topic
st.write("Enter a topic to start on and/or a topic to end on. If left blank, a topic will be chosen randomly.")
start_topic = st.text_input("Start Topic", "Enter the starting topic here")

# User input for Target Topic
target_topic = st.text_input("Target Topic", "Enter the target topic here")

# if start not passed, get a random one
if not start_topic:
    start_topic = get_random_wiki_page(wiki_wiki)

# if target topic not passed, get a random one
if not target_topic:
    target_topic = get_random_wiki_page(wiki_wiki)

# User begins and game starts
if st.button("Begin"):
    st.divider()
    st.write("Start Topic: ", start_topic)
    st.write("Target Topic: ", target_topic)

    game = WikiGameBot(wiki_wiki = wiki_wiki, start_topic = start_topic, target_topic = target_topic)
    game.play_game()
    
    # printing progress throughout the game
    for i in game.printouts[::-1]: # reversing so the newest is always at the top
        st.write(i)

    st.divider()

outro_message = """
Thanks for checking out this app. If you have any questions or comments or would like to connect for any reason, you can reach me at:
- Email: kmaurinjones@gmail.com
- LinkedIn: https://www.linkedin.com/in/kmaurinjones/
""".strip()
st.write(outro_message)

disclaimer = """
Disclaimer: This creator of this webapp and author of this code is not responsible for the content produced by the app. The context produced by this app is a result of content sourced from Wikipedia and is property of Wikipedia.
""".strip()
st.write(disclaimer)
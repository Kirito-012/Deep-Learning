from api import *
import streamlit as st
EPISODE_API = '6b7ceb9ecfd54fc59648227ef321f6a6'

st.title("Welcome to Podcast Summaries")
episode_id = st.sidebar.text_input("Please Enter the episode ID")
button = st.sidebar.button("Get Summary!", on_click=save_transcript, args=(episode_id,))
if st.sidebar.button("Get ID"):
      st.sidebar.markdown("[Click here to redirect](https://www.listennotes.com/api/docs/?s=side_bottom&id=871d4f822c2f41928267abf4ed93524c#get-api-v2-episodes-id)", unsafe_allow_html=True)

def get_clean_time(start_ms):
      seconds = int((start_ms / 1000) % 60)
      minutes = int((start_ms / 1000 * 60) % 60)
      hours = int((start_ms /1000 * 60 * 60) % 24)

      if hours > 0:
            start_t = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
      else:
            start_t = f'{minutes:02d}:{seconds:02d}'
      
      return start_t
      

if button:
      filename = 'Chapters.json'
      with open(filename, 'r') as f:
            data = json.load(f)
            chapters = data['chapters']
            eps_thumbnail = data['eps_thumbnail'] 
            eps_title = data['eps_title'] 
            podcast_title = data['podcast_title']


      st.header(f"{podcast_title} - {eps_title}")
      st.image(eps_thumbnail)
      for chap in chapters:
            with st.expander(chap['gist'] + '-' + get_clean_time(chap['start'])):
                  chap['summary']
import streamlit as st
import youtube_transcript_api
import re
from collections import Counter
import re
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
# from multi_emotion import multi_emotion
from pysentimiento import create_analyzer
import pandas as pd
import json
from textblob import TextBlob
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock


# Function to get video transcript from YouTube

def get_youtube_transcript(video_url):
    try:
        video_id = video_url.split("v=")[1]
        video = youtube_transcript_api.YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([entry['text'] for entry in video])
        return transcript
    except Exception as e:
        return None


# Streamlit app
st.title("▶️ YouTube Video Transcript Processor")
st.write("Enter a YouTube video URL to extract and process the transcript:")

# Initialize with a text input field for the first YouTube video URL
# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

video_urls = [st.text_input(
    "YouTube Video URL",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
    placeholder="e.g. https://www.youtube.com/watch?v=rwkPCt77Mcs",
    key="placeholder",
)]

# Maximum number of URLs allowed
max_urls = 5

# Add more input fields and checkboxes if needed, up to a maximum of 20
for i in range(max_urls - 1):
    if i < len(video_urls) and st.checkbox(f"Add YouTube Video URL {i + 2}", value=False):
        video_urls.append(st.text_input(f"YouTube Video URL {i + 2}"))

process_button = st.button("Process Video Transcripts")

# Process transcripts only if the button is clicked
# st.sidebar.markdown("Process any Text🔡Web🕸️Page📄")
st.sidebar.link_button("NLP Web Content Analysis", "https://nlp-web-content-analysis.streamlit.app/",use_container_width=True)

# st.sidebar.markdown("Compare any Text🔡Web🕸️Page📄")
st.sidebar.link_button("NLP Compare Web Content", "https://nlp-compare-web-content.streamlit.app/",use_container_width=True)

# st.sidebar.markdown("Process any YouTube ▶️ Video of English Language")
st.sidebar.link_button("NLP YouTube Analysis (Selected 🎉 ✅)", "https://nlp-youtube-analysis.streamlit.app/",use_container_width=True)

# st.sidebar.markdown("Compare any YT ▶️ with captions")
st.sidebar.link_button("NLP Compare YouTube Videos", "https://nlp-compare-youtube-videos.streamlit.app/",use_container_width=True)
if process_button:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    nltk.download('stopwords')
    nltk.download('punkt')
    stopWords = set(stopwords.words("english"))
    # Function to summarize a given text
    def summarize_text(text):
        # Tokenizing the text
        words = word_tokenize(text)
    
        # Creating a frequency table to keep the score of each word
        freqTable = dict()
        for word in words:
            word = word.lower()
            if word in stopWords:
                continue
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1
    
        # Creating a dictionary to keep the score of each sentence
        sentences = sent_tokenize(text)
        sentenceValue = dict()
    
        for sentence in sentences:
            for word, freq in freqTable.items():
                if word in sentence.lower():
                    if sentence in sentenceValue:
                        sentenceValue[sentence] += freq
                    else:
                        sentenceValue[sentence] = freq
    
        sumValues = 0
        for sentence in sentenceValue:
            sumValues += sentenceValue[sentence]
    
        # Average value of a sentence from the original text
        average = int(sumValues / len(sentenceValue))
    
        # Storing sentences into our summary.
        summary = ''
        for sentence in sentences:
            if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
                summary += " " + sentence
    
        return summary

    # Function to extract common keywords from a given text
    def extract_keywords(text, top_n=10):
        # Preprocess and remove stopwords
        filtered_words = []
        words = nltk.word_tokenize(text)
        for word in words:
            word = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', word)
            if word.lower() not in stopWords and len(word) > 1:
                filtered_words.append(word.lower())
    
        # Extract keywords (non-stopwords as keywords)
        keywords = [word for word in filtered_words]
    
        # Count word frequencies
        word_count = Counter(keywords)
    
        # Extract the top N keywords based on their frequency
        top_keywords = [keyword for keyword, _ in word_count.most_common(top_n)]
    
        return top_keywords
    
    user_video_urls = [url for url in video_urls if url.strip()]
    user_transcripts = []

    for url in user_video_urls:
        transcript = get_youtube_transcript(url)

        if transcript:
            user_transcripts.append(transcript)
    # Convert the list of transcripts into a single string
    text = "\n".join(user_transcripts)
    
    # from transformers import AddedToken
    
    # # Define a custom hash function for tokenizers.AddedToken
    # def my_hash_func(token):
    #     try:
    #         return hash((token.ids, token.type_id))
    #     except AttributeError:
    #         # Handle cases where the token object is not as expected
    #         return hash(str(token))
    
    # @st.cache_resource(hash_funcs={AddedToken: my_hash_func})
    # def get_analyzers():
    #     from setup import analyzer, emotion_analyzer, hate_speech_analyzer
    #     return analyzer, emotion_analyzer, hate_speech_analyzer
    from my_module import get_analyzers
    
    # Load analyzers
    analyzers = get_analyzers()
    
    # Now you can use the analyzers for text analysis
    sentiment1 = analyzers[0].predict(text)
    emotion1 = analyzers[1].predict(text)
    hate_speech1 = analyzers[2].predict(text)
    # from setup import analyzer, emotion_analyzer, hate_speech_analyzer
    # from transformers import AddedToken
    
    # # Define a custom hash function for tokenizers.AddedToken
    # def my_hash_func(token):
    #     try:
    #         return hash((token.ids, token.type_id))
    #     except AttributeError:
    #         # Handle cases where the token object is not as expected
    #         return hash(str(token))
    
    # @st.cache(allow_output_mutation=True, hash_funcs={AddedToken: my_hash_func})
    # def create_analyzers():
    #     return analyzer, emotion_analyzer, hate_speech_analyzer
    
    # analyzers = create_analyzers()
    
    # sentiment1 = analyzers[0].predict(text)
    # emotion1 = analyzers[1].predict(text)
    # hate_speech1 = analyzers[2].predict(text)
    

    TOInews = re.sub("[^A-Za-z" "]+", " ", text).lower()
    TOInews_tokens = TOInews.split(" ")

    with open("en-stop-words.txt", "r") as sw:
        stop_words = sw.read()
                    
    stop_words = stop_words.split("\n")

    tokens = [w for w in TOInews_tokens if not w in stop_words]

    tokens_frequencies = Counter(tokens)

    # tokens_frequencies = tokens_frequencies.loc[tokens_frequencies.text != "", :]
    # tokens_frequencies = tokens_frequencies.iloc[1:]

    # Sorting
    tokens_frequencies = sorted(tokens_frequencies.items(), key = lambda x: x[1])

    # Storing frequencies and items in separate variables 
    frequencies = list(reversed([i[1] for i in tokens_frequencies]))
    words = list(reversed([i[0] for i in tokens_frequencies]))

    # Barplot of top 10 
    # import matplotlib.pyplot as plt
    
    
    # Create a figure and bar chart
    with _lock:
        plt.figure(1)
        plt.bar(height=frequencies[0:11], x=range(0, 11), color=['red', 'green', 'black', 'yellow', 'blue', 'pink', 'violet'], width=0.6)
        plt.title("Top 10 Tokens (Words)")
        plt.grid(True)
        # Customize the x-axis labels and rotation for visibility
        plt.xticks(range(0, 11), words[0:11], rotation=45)
        plt.xlabel("Tokens")
        plt.ylabel("Count")
        
        # Display the plot in Streamlit
        st.pyplot(plt.figure(1), use_container_width=True)
    ##########

    st.write("Please be patience for Amazing Results it will take few minutes")
    # Joinining all the tokens into single paragraph 
    cleanstrng = " ".join(words)

    with _lock:
        plt.figure(2)
        wordcloud_ip = WordCloud(background_color = 'White', width = 2800, height = 2400).generate(cleanstrng)
        plt.title("Normal Word Cloud")
        plt.axis("off")
        plt.grid(False)
        plt.imshow(wordcloud_ip)
        st.pyplot(plt.figure(2), use_container_width=True)


    #########################################################################################

    # positive words
    with open("en-positive-words.txt", "r") as pos:
        poswords = pos.read().split("\n")
    # Positive word cloud
    # Choosing the only words which are present in positive words
    pos_tokens = " ".join ([w for w in TOInews_tokens if w in poswords])

    with _lock:
        plt.figure(3)
        wordcloud_positive = WordCloud(background_color = 'White', width = 1800, height = 1400).generate(pos_tokens)
        plt.title("Positive Word Cloud")
        plt.axis("off")
        plt.grid(False)
        plt.imshow(wordcloud_positive)
        st.pyplot(plt.figure(3), use_container_width=True)

    # Negative words
    with open("en-negative-words.txt", "r") as neg:
        negwords = neg.read().split("\n")
    # Negative word cloud
    # Choosing the only words which are present in negwords
    neg_tokens = " ".join ([w for w in TOInews_tokens if w in negwords])
    with _lock:
        plt.figure(4)
        wordcloud_negative = WordCloud(background_color = 'black', width = 1800, height=1400).generate(neg_tokens)
        plt.title("Negative Word Cloud")
        plt.axis("off")
        plt.grid(False)
        plt.imshow(wordcloud_negative)
        st.pyplot(plt.figure(4), use_container_width=True)
    #########################################################################################
    
    
    # Word cloud with 2 words together being repeated

    # Extracting n-grams using TextBlob

    bigrams_list = list(nltk.bigrams(tokens))
    dictionary2 = [' '.join(tup) for tup in bigrams_list]

    # Using count vectorizer to view the frequency of bigrams
    
    vectorizer = CountVectorizer(ngram_range = (2, 2))
    bag_of_words = vectorizer.fit_transform(dictionary2)
    v1 = vectorizer.vocabulary_

    sum_words = bag_of_words.sum(axis = 0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in v1.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

    words_dict = dict(words_freq[:100])
    with _lock:
        plt.figure(5)
        wordcloud_2 = WordCloud(background_color = 'black', width = 1800, height = 1400)                 
        wordcloud_2.generate_from_frequencies(words_dict)
        plt.title("Bi-Gram based on Frequency")
        plt.axis("off")
        plt.grid(False)
        plt.imshow(wordcloud_2)
        st.pyplot(plt.figure(5), use_container_width=True)
    ##############################################################################################
    
    # Word cloud with 2 words together being repeated
    
    # Extracting n-grams using TextBlob

    bigrams_list2 = list(nltk.trigrams(tokens))
    dictionary3 = [' '.join(tup) for tup in bigrams_list2]

    # Using count vectorizer to view the frequency of bigrams
    
    vectorizer1 = CountVectorizer(ngram_range = (3, 3))
    bag_of_words1 = vectorizer1.fit_transform(dictionary3)
    v2 = vectorizer1.vocabulary_

    sum_words1 = bag_of_words1.sum(axis = 0)
    words_freq1 = [(word1, sum_words1[0, idx1]) for word1, idx1 in v2.items()]
    words_freq1 = sorted(words_freq1, key = lambda x: x[1], reverse = True)

    words_dict1 = dict(words_freq1[:100])
    with _lock:
        plt.figure(6)
        wordcloud_3 = WordCloud(background_color = 'black', width = 1800, height = 1400)                  
        wordcloud_3.generate_from_frequencies(words_dict1)
        plt.title("Tri-Gram based on Frequency")
        plt.grid(False)
        plt.axis("off")
        plt.imshow(wordcloud_3)
        st.pyplot(plt.figure(6), use_container_width=True)

    # eqn shift 1
    pattern = "[^A-Za-z.]+"

    # Perform text preprocessing without removing full stops
    sen = re.sub(pattern, " ", text).lower()

    # SENTANCE Tokenizer
    sen_t = sen.split(".")


    # Create a DataFrame with the sentences as lists
    df = pd.DataFrame(sen_t)

    # Display the DataFrame
    print(df)

    df.columns = ['text']
    

    # Number of words
    df['number_of_words'] = df['text'].apply(lambda x : len(TextBlob(x).words))

    # Detect presence of wh words
    wh_words = set(['why', 'who', 'which', 'what', 'where', 'when', 'how'])
    df['are_wh_words_present'] = df['text'].apply(lambda x : True if len(set(TextBlob(str(x)).words).intersection(wh_words)) > 0 else False)


    # Polarity
    df['polarity'] = df['text'].apply(lambda x : TextBlob(str(x)).sentiment.polarity)

    # Subjectivity
    df['subjectivity'] = df['text'].apply(lambda x : TextBlob(str(x)).sentiment.subjectivity)

    
    # Calculate the average number of words
    average_words = df['number_of_words'].mean()

    # Calculate the percentage of sentences that have WH words
    average_wh_presence = (df['are_wh_words_present'].sum() / len(df)) * 100

    # Calculate the average polarity
    average_polarity = df['polarity'].mean()

    # Calculate the average subjectivity
    average_subjectivity = df['subjectivity'].mean()

    # Display the calculated averages
    print("Average Number of Words:", average_words)
    print("Average Percentage of Sentences with WH Words:", average_wh_presence)
    print("Average Polarity:", average_polarity)
    print("Average Subjectivity:", average_subjectivity)

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Metric': ['Average Number of Words', 'Average Percentage of Sentences with WH Words', 'Average Polarity', 'Average Subjectivity'],
        'Value': [average_words, average_wh_presence, average_polarity, average_subjectivity]
    })
    st.table(results_df)
    
    # emo_in_txt = text
    # Define cache for the analyzers
    # from setup import analyzer, emotion_analyzer, hate_speech_analyzer
    # @st.cache(allow_output_mutation=True)
    # def create_analyzers():
    #     return analyzer, emotion_analyzer, hate_speech_analyzer
    
    # analyzers = create_analyzers()
    # sentiment1 = analyzers[0].predict(text)
    # emotion1 = analyzers[1].predict(text)
    # hate_speech1 = analyzers[2].predict(text)
    # analyzer = create_analyzer(task="sentiment", lang="en")
    # sentiment1 = analyzer.predict(text)
    st.subheader("Sentiment Analysis")
    st.write(sentiment1)
    print(sentiment1)
    sentiment_output = sentiment1.output
    probas_sentiment = sentiment1.probas
    NEU = probas_sentiment.get("NEU")
    POS = probas_sentiment.get("POS")
    NEG = probas_sentiment.get("NEG")
    

    # Create labels and values for the pie chart
    labels = ['NEU', 'POS', 'NEG']
    values = [NEU, POS, NEG]
    colors = ['blue', 'green', 'red']
    
    with _lock:
        # Create a figure with the figure number 7
        plt.figure(7, figsize=(6, 6))
        
        # Create a pie chart with custom colors
        wedges, _ = plt.pie(values, colors=colors, startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
        
        # Create a legend with labels and values
        legend_labels = [f"{label}: {value:.1%}" for label, value in zip(labels, values)]
        plt.legend(wedges, legend_labels, title="Sentiments", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.show()
        st.pyplot(plt.figure(7), use_container_width=True)

    st.write("Sentiment Output:", sentiment_output)
    st.write("Probas Sentiment:")
    st.write("NEU:", NEU)
    st.write("POS:", POS)
    st.write("NEG:", NEG)
    
    # emotion_analyzer = create_analyzer(task="emotion", lang="en")
    # emotion1 = emotion_analyzer.predict(text)
    st.subheader("Emotion Analysis")
    st.write(emotion1)
    print(emotion1)
    emotion_output = emotion1.output
    probas_emotion = emotion1.probas
    others = probas_emotion.get("others")
    joy = probas_emotion.get("joy")
    disgust = probas_emotion.get("disgust")
    fear = probas_emotion.get("fear")
    sadness = probas_emotion.get("sadness")
    surprise = probas_emotion.get("surprise")
    anger = probas_emotion.get("anger")
    

    # Create a dictionary for the emotion probabilities
    emotions101 = {
        "Others": others,
        "Joy": joy,
        "Disgust": disgust,
        "Fear": fear,
        "Sadness": sadness,
        "Surprise": surprise,
        "Anger": anger
    }
    # Extract emotion labels and probabilities
    emotions = emotions101.keys()
    probabilities = emotions101.values()
    
    # Create a bar plot
    with _lock:
        plt.figure(8,figsize=(10, 6))
        plt.bar(emotions, probabilities, color=['blue', 'green', 'red', 'purple', 'orange', 'pink'])
        plt.xlabel("Emotion")
        plt.ylabel("Probability")
        plt.title("Emotion Probabilities")
        plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1
        plt.show()
        st.pyplot(plt.figure(8), use_container_width=True)

    st.write("Emotion Output:", emotion_output)
    st.write("Probas Emotion: ⤵️")
    st.write("Others:", others)
    st.write("Joy:", joy)
    st.write("Disgust:", disgust)
    st.write("Fear:", fear)
    st.write("Sadness:", sadness)
    st.write("Surprise:", surprise)
    st.write("Anger:", anger)
    # Show the plot
    
    # st.bar_chart(emotions101)
    # with _lock:
    #     plt.figure(8)
    #     plt.barh(list(emotions101.keys()), list(emotions101.values()))
    #     plt.xlabel('Probability')
    #     plt.title('Emotion Analysis')
    #     st.pyplot(plt.figure(8), use_container_width=True)
    
    # hate_speech_analyzer = create_analyzer(task="hate_speech", lang="en")

    # hate_speech1 =  hate_speech_analyzer.predict(text)
    st.subheader("Hate Speech Analysis")
    st.write(hate_speech1)
    print(hate_speech1)
    hate_speech1_output = hate_speech1.output
    probas_hate_speech1 = hate_speech1.probas
    # Extract the values
    hateful = probas_hate_speech1.get("hateful")
    targeted = probas_hate_speech1.get("targeted")
    aggressive = probas_hate_speech1.get("aggressive")
    
 
    
    # Create a dictionary for the hate speech probabilities
    hate_speech = {
        "Hateful": hateful,
        "Targeted": targeted,
        "Aggressive": aggressive
    }
    
    # Extract hate speech labels and probabilities
    labels = hate_speech.keys()
    probs = hate_speech.values()
    
    # Create a bar plot
    with _lock:
        plt.figure(9,figsize=(10, 6))
        plt.bar(labels, probs, color=['red', 'green', 'blue'])
        plt.xlabel("Category")
        plt.ylabel("Probability")
        plt.title("Hate Speech Probabilities")
        plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1
        # Show the plot
        plt.show()
        st.pyplot(plt.figure(9), use_container_width=True) 
        st.write("Hate Speech Output:", hate_speech1_output)

    st.write("Probas:")
    st.write("Hateful:", hateful)
    st.write("Targeted:", targeted)
    st.write("Aggressive:", aggressive) 
    # Now, text contains all the video transcripts as a single string

    for idx, transcript in enumerate(user_transcripts, start=1):
        st.subheader(f"Processed Transcript {idx}:")
        st.write("Transcript Text:")
        st.write(transcript)  # Display the transcript

        # Generate and display the summary
        summary = summarize_text(transcript)
        st.subheader(f"Summary for Transcript {idx}:")
        st.write(summary)  # Display the summary

        # Extract keywords from each transcript
        keywords = extract_keywords(transcript, top_n=10)
        st.subheader(f"Keywords for Transcript {idx}:")
        st.write(keywords)

    # Extract common keywords from all user transcripts
    common_keywords = extract_keywords(" ".join(user_transcripts), top_n=10)
    print(common_keywords)
    st.subheader("Common Keywords:")
    st.write(common_keywords)
    st.balloons()
    st.cache_resource.clear()
st.write("Note: This app uses the YouTube Transcript API to retrieve captions.")

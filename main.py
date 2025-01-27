import pandas as pd
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe
import speech_recognition as sr

# Download NLTK data for sentiment analysis
nltk.download("vader_lexicon")

# Google Sheets authentication
def authenticate_google_sheets():
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
    client = gspread.authorize(creds)
    return client

# Function to create or access a Google Sheet for storing processed product data
def create_or_access_output_google_sheet(client, sheet_name):
    try:
        sheet = client.open(sheet_name)
    except gspread.exceptions.SpreadsheetNotFound:
        sheet = client.create(sheet_name)
    worksheet = sheet.get_worksheet(0) or sheet.add_worksheet(title="Sheet1", rows="1000", cols="20")
    return worksheet

# Function to load product data from the Excel file
def load_excel_data(file_path):
    try:
        df_products = pd.read_excel(file_path, sheet_name="Sheet1")
        return df_products
    except Exception as e:
        print(f"Failed to load Excel file: {e}")
        return None

# Sentiment Analysis using VADER
def analyze_sentiment_vader(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_score = sid.polarity_scores(text)
    compound_score = sentiment_score['compound']
    
    if compound_score >= 0.05:
        sentiment = "Positive"
    elif compound_score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return sentiment, compound_score

# Function to get voice input
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening for your query. Please speak now...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=10)
            query = recognizer.recognize_google(audio)
            st.success(f"You said: {query}")
            return query
        except Exception as e:
            st.error(f"Error: {e}")
    return ""

# Function to recommend products based on multiple columns
def recommend_products_using_multiple_columns(query, df_products):
    # Combine relevant columns into a single text column for vectorization
    df_products['combined_features'] = df_products['name'].astype(str) + " " + \
                                       df_products['main_category'].astype(str) + " " + \
                                       df_products['sub_category'].astype(str) + " " + \
                                       df_products['ratings'].astype(str) + " " + \
                                       df_products['no_of_ratings'].astype(str) + " " + \
                                       df_products['discount_price'].astype(str) + " " + \
                                       df_products['actual_price'].astype(str)

    # Vectorize the combined column
    vectorizer = TfidfVectorizer(stop_words='english')
    combined_matrix = vectorizer.fit_transform(df_products['combined_features'])

    # Vectorize the user's query
    query_vector = vectorizer.transform([query])

    # Compute cosine similarity between the query and the combined column
    cosine_similarities = cosine_similarity(query_vector, combined_matrix).flatten()

    # Get the indices of the top 5 most similar products
    top_indices = cosine_similarities.argsort()[-5:][::-1]

    # Prepare recommendations
    recommendations = []
    for idx in top_indices:
        recommendations.append({
            "name": df_products.iloc[idx]["name"],
            "main_category": df_products.iloc[idx]["main_category"],
            "sub_category": df_products.iloc[idx]["sub_category"],
            "ratings": df_products.iloc[idx]["ratings"],
            "no_of_ratings": df_products.iloc[idx]["no_of_ratings"],
            "discount_price": df_products.iloc[idx]["discount_price"],
            "actual_price": df_products.iloc[idx]["actual_price"],
            "image": df_products.iloc[idx]["image"],
            "link": df_products.iloc[idx]["link"]
        })

    return recommendations

# Store query and sentiment in Google Sheets
def store_query_in_google_sheet(query, sentiment, sentiment_score):
    client = authenticate_google_sheets()
    worksheet = create_or_access_output_google_sheet(client, "Query Sentiment History")
    
    data = [[query, sentiment, sentiment_score]]
    existing_data = pd.DataFrame(worksheet.get_all_records())
    if existing_data.empty:
        header = ["Query", "Sentiment", "Sentiment Score"]
        set_with_dataframe(worksheet, pd.DataFrame(data, columns=header))
    else:
        new_data = pd.DataFrame(data, columns=existing_data.columns)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        set_with_dataframe(worksheet, updated_data)

# Streamlit UI and Main App Logic
st.title("Sentiment Analysis and Product Recommendation System")

def main():
    file_path = r'C:\Users\user\OneDrive\Desktop\info project\Updated_Combined_Sample.xlsx'
    df_products = load_excel_data(file_path)
    if df_products is None:
        st.error("Failed to load the product dataset.")
        return

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page:", ["Home", "View Sentiment Data", "Sentiment Analysis Dashboard"])

    if page == "Home":
        st.subheader("Product Search and Sentiment Analysis")
        if st.button("Speak Now"):
            search_query = get_voice_input()
            if search_query:
                sentiment, sentiment_score = analyze_sentiment_vader(search_query)
                st.write(f"Sentiment: {sentiment} (Score: {sentiment_score:.2f})")
                store_query_in_google_sheet(search_query, sentiment, sentiment_score)
                
                if sentiment == "Negative":
                    st.write("Negative sentiment detected. No recommendations.")
                else:
                    recommendations = recommend_products_using_multiple_columns(search_query, df_products)
                    st.subheader("Recommendations")
                    if recommendations:
                        for rec in recommendations:
                            st.image(rec['image'], width=150)
                            st.write(f"**{rec['name']}**")
                            st.write(f"Category: {rec['main_category']} > {rec['sub_category']}")
                            st.write(f"Price: {rec['discount_price']} (Actual: {rec['actual_price']})")
                            st.write(f"Ratings: {rec['ratings']} ({rec['no_of_ratings']} reviews)")
                            st.write(f"[Buy Now]({rec['link']})")
                            st.write("---")
                    else:
                        st.write("No products match your query.")

    elif page == "View Sentiment Data":
        st.subheader("Sentiment Data from Google Sheets")
        client = authenticate_google_sheets()
        worksheet = create_or_access_output_google_sheet(client, "Query Sentiment History")
        google_sheet_data = pd.DataFrame(worksheet.get_all_records())
        if not google_sheet_data.empty:
            st.write(google_sheet_data)
        else:
            st.write("No sentiment data available.")

    elif page == "Sentiment Analysis Dashboard":
        st.subheader("Sentiment Analysis Dashboard")
        client = authenticate_google_sheets()
        worksheet = create_or_access_output_google_sheet(client, "Query Sentiment History")
        google_sheet_data = pd.DataFrame(worksheet.get_all_records())
        if not google_sheet_data.empty:
            sentiment_counts = google_sheet_data['Sentiment'].value_counts()
            st.write("Sentiment distribution of queries:")
            st.bar_chart(sentiment_counts)
        else:
            st.write("No sentiment data available.")

if __name__ == "__main__":
    main()

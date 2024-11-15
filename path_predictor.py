import streamlit as st
import pandas as pd
# import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

df_all = pd.read_csv("majors-list.csv").fillna(" ")
alljbs = pd.read_csv("Jobs.csv")
descrip = pd.read_csv("major_description.csv", delimiter=';')
salary = pd.read_csv("salary.csv")

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

stop_words_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()
tfidf_vectorizer = TfidfVectorizer()


def clean_txt(text):
    text = re.sub("'", "", text)
    text = re.sub("(\\d|\\W)+", " ", text)
    clean_text = [wn.lemmatize(word) for word in word_tokenize(text.lower())
                  if word not in stop_words_ and word not in string.punctuation]
    return " ".join(clean_text)


def get_recommendation(top, df, scores):
    recommendation = pd.DataFrame(columns=['major', 'description', 'rough salary (can highly vary)', 'score'])
    for count, i in enumerate(top):
        recommendation.at[count, 'major'] = df['Major'][i]
        recommendation.at[count, 'score'] = scores[count]
        for j in range(len(descrip)):
            if recommendation.at[count, 'major'].lower() == descrip.at[j, 'major'].lower():
                recommendation.at[count, 'description'] = descrip.at[j, ' description']
        for j in range(len(salary)):
            if recommendation.at[count, 'major'].lower() == salary.at[j, 'major'].lower():
                recommendation.at[count, 'rough salary (can highly vary)'] = salary.at[j, 'salary']
    return recommendation


def TFIDFRUN(user_text):
    df_all['clean_major'] = df_all['Major'].apply(clean_txt)
    tfidf_major = tfidf_vectorizer.fit_transform(df_all['clean_major'])
    user_tfidf = tfidf_vectorizer.transform([user_text])

    cos_similarity_tfidf = cosine_similarity(user_tfidf, tfidf_major)
    top_indices = cos_similarity_tfidf[0].argsort()[-10:][::-1]
    list_scores = [cos_similarity_tfidf[0][i] for i in top_indices]
    return get_recommendation(top_indices, df_all, list_scores)


def KNNRUN(user_text):
    df_all['clean_major'] = df_all['Major'].apply(clean_txt)
    tfidf_major = tfidf_vectorizer.fit_transform(df_all['clean_major'])
    user_tfidf = tfidf_vectorizer.transform([user_text])

    KNN = NearestNeighbors(n_neighbors=11, p=2)
    KNN.fit(tfidf_major)
    NNs = KNN.kneighbors(user_tfidf, return_distance=True)
    top_indices = NNs[1][0][1:]
    index_scores = NNs[0][0][1:]

    return get_recommendation(top_indices, df_all, index_scores)


def main():
    st.title("Path Predictor")

    grade = st.number_input("What is your current grade level?", min_value=1, max_value=12)
    subject = st.selectbox("Which subject do you enjoy the most?",
                           options=["Math", "English", "History", "Foreign Languages",
                                    "Computer Science", "Chemistry", "Physics", "Biology", "Art", "Geography"])
    hobby = st.selectbox("What is your main hobby or interest?",
                         options=["Reading and Writing", "Coding", "Drawing and Painting",
                                  "Cooking", "Gardening", "Teaching Others"])
    category = st.selectbox("Which of the following category would you be most interested in studying?",
                         options=["Agriculture & Natural Resources", "Arts", "Biology & Life Science", "Business",
                                  "Communications & Journalism", "Computers & Mathematics", "Education", "Engineering",
                                  "Health", "Humanities & Liberal Arts", "Industrial Arts & Consumer Services",
                                  "Interdisciplinary", "Law & Public Policy", "Physical Sciences", "Psychology & Social Work",
                                  "Social Science"])

    resume = st.text_area("Please enter your resume")

    if st.button("Get Recommendations"):
        words = f"{subject} {hobby} {category}"
        user_text = clean_txt(resume + " " + words)

        # st.subheader("TF-IDF Recommendations")
        tfidf_recommendations = TFIDFRUN(user_text)
        tfidf_recommendations['score'] = 1 - tfidf_recommendations['score']
        # st.write(tfidf_recommendations)

        # st.subheader("KNN Recommendations")
        knn_recommendations = KNNRUN(user_text)
        # st.write(knn_recommendations)

        merged_recommendations = pd.merge(
            knn_recommendations, tfidf_recommendations, on='major', how='inner', suffixes=('_knn', '_tfidf')
        )

        merged_recommendations['combined_score'] = (
            merged_recommendations['score_knn'] + merged_recommendations['score_tfidf']
        )
        merged_recommendations.sort_values(by='combined_score', ascending=True, inplace=True)

        # st.subheader("Combined Recommendations")
        st.subheader("Recommendations")
        # columns_to_display = ['major', 'combined_score']
        merged_recommendations.rename(columns={'description_knn': 'description'}, inplace=True)
        merged_recommendations.rename(columns={'rough salary (can highly vary)_knn': 'rough salary (can highly vary)'}, inplace=True)
        merged_recommendations.index = merged_recommendations.index + 1
        columns_to_display = ['major', 'description', 'rough salary (can highly vary)']
        st.table(merged_recommendations[columns_to_display])

        st.subheader("Jobs Recommendations")
        st.write("Based on the major that fits you the best, here are some job opportunities that would suit your major:")
        major_to_check = merged_recommendations.iloc[0]["major"]
        major_to_check = major_to_check.lower()
        matching_jobs = []

        for i in range(len(alljbs)):
            if alljbs.iloc[i]["Job_Major"].lower() == major_to_check:
                matching_jobs.append((alljbs.iloc[i]["Job_Title"], alljbs.iloc[i]["Job_Description"]))

        if matching_jobs:
            for pair in matching_jobs:
                job_title, job_description = pair
                st.write(f"**{job_title}**:")
                st.write(f"Here is an overview of the job: {job_description}")
        else:
            st.write("No matching job titles found.")

if __name__ == "__main__":
    main()

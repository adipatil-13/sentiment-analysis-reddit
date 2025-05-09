import streamlit as st
import boto3
import pandas as pd
import praw
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from decimal import Decimal
import plotly.express as px

nltk.download("vader_lexicon")
sid = SentimentIntensityAnalyzer()

st.set_page_config(layout="wide")
st.title("🧠 Reddit Post Comment Sentiment Analyzer")
st.caption("Compare Amazon Comprehend & VADER on top 50 comments of a post")

reddit = praw.Reddit(
    client_id=st.secrets["reddit"]["client_id"],
    client_secret=st.secrets["reddit"]["client_secret"],
    user_agent=st.secrets["reddit"]["user_agent"]
)

session = boto3.Session(
    aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
    region_name=st.secrets["aws"]["region_name"]
)
comprehend = session.client("comprehend")

post = next(reddit.subreddit("technology").hot(limit=1))
st.subheader(f"Post: {post.title}")
st.write(f"🔗 [View on Reddit](https://www.reddit.com{post.permalink})")
st.markdown("---")

post.comments.replace_more(limit=0)
comments = post.comments[:50]

texts = [comment.body for comment in comments]
results = []

for text in texts:
    if len(text.strip()) < 10:
        continue

    comp = comprehend.detect_sentiment(Text=text, LanguageCode="en")
    comp_sentiment = comp["Sentiment"]
    comp_scores = comp["SentimentScore"]

    vader_score = sid.polarity_scores(text)["compound"]
    vader_sentiment = (
        "POSITIVE" if vader_score > 0.05 else
        "NEGATIVE" if vader_score < -0.05 else
        "NEUTRAL"
    )

    results.append({
        "comment": text,
        "comprehend_sentiment": comp_sentiment,
        "positive_score": comp_scores["Positive"],
        "negative_score": comp_scores["Negative"],
        "neutral_score": comp_scores["Neutral"],
        "mixed_score": comp_scores["Mixed"],
        "vader_score": vader_score,
        "vader_sentiment": vader_sentiment
    })

df = pd.DataFrame(results)
df["match"] = df["comprehend_sentiment"] == df["vader_sentiment"]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Amazon Comprehend Sentiment")
    st.plotly_chart(px.histogram(df, x="comprehend_sentiment", color="comprehend_sentiment"), use_container_width=True)
with col2:
    st.subheader("VADER Sentiment")
    st.plotly_chart(px.histogram(df, x="vader_sentiment", color="vader_sentiment"), use_container_width=True)

st.subheader("🧾 Sentiment Comparison Table")
st.dataframe(df[["comment", "comprehend_sentiment", "vader_sentiment", "vader_score"]].head(20))

st.subheader("⚠️ Mismatched Sentiments")
st.write(f"Total comments analyzed: {len(df)}")
st.write(f"Disagreements: {len(df[df['match'] == False])}")
st.dataframe(df[df["match"] == False][["comment", "comprehend_sentiment", "vader_sentiment", "vader_score"]].head(10))

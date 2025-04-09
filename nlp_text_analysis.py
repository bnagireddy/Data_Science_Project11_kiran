import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure required NLTK dependencies are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# STEP 1: Load the cleaned dataset
file_path = "cleaned_youtube.csv"
data = pd.read_csv(file_path)

# Check if required columns exist
if 'title' not in data.columns or 'tags' not in data.columns:
    print("Missing 'title' or 'tags' column in the dataset. Exiting.")
    exit()

# STEP 2: Combine text fields for analysis
data['combined_text'] = data['title'].astype(str) + " " + data['tags'].astype(str)

# STEP 3: Tokenize and clean text
stop_words = set(stopwords.words('english'))

def tokenize_and_clean(text):
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]  # Remove stopwords and non-alphanumeric tokens
    return tokens

data['tokens'] = data['combined_text'].apply(tokenize_and_clean)

# STEP 4: Generate word frequency
all_tokens = [token for tokens in data['tokens'] for token in tokens]
word_freq = Counter(all_tokens)

print("\nTop 10 Most Common Words:")
for word, freq in word_freq.most_common(10):
    print(f"{word}: {freq}")

# STEP 5: Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(all_tokens))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Titles and Tags")
plt.show()

# STEP 6: Sentiment Analysis on Titles
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

data['sentiment_score'] = data['title'].apply(analyze_sentiment)

# Display average sentiment score
average_sentiment = data['sentiment_score'].mean()
print(f"\nAverage Sentiment Score: {average_sentiment:.2f}")

# STEP 7: Save the analyzed data
data.to_csv("nlp_analysis_results.csv", index=False)
print("\n✅ NLP text analysis complete. Results saved as 'nlp_analysis_results.csv'.")
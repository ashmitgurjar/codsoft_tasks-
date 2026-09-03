import re
import string
import ssl
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Bypass SSL verification issue on macOS python when downloading NLTK datasets
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

for resource in ['stopwords', 'punkt', 'wordnet']:
    try:
        nltk.download(resource, quiet=True)
    except Exception as e:
        pass

try:
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
        "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
        "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
        "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
        "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
        "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the",
        "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
        "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
        "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
        "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't",
        "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
    }

stemmer = PorterStemmer()
try:
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("testing")
    HAS_WORDNET = True
except Exception:
    HAS_WORDNET = False

def clean_text(text, use_stemming=False, use_lemmatization=True):
    """
    Cleans raw text plot summary:
    - Lowercase
    - Strip HTML tags and special punctuation
    - Remove numbers
    - Remove English stopwords
    - Apply Lemmatization or Stemming
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Tokenize by whitespace
    tokens = text.split()

    # Remove stopwords and short words (< 2 characters)
    tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 2]

    # Stemming or Lemmatization
    if use_lemmatization and HAS_WORDNET:
        try:
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        except Exception:
            tokens = [stemmer.stem(word) for word in tokens]
    elif use_stemming:
        tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)

def preprocess_dataframe(df, text_column="plot", target_column="clean_plot"):
    """
    Applies text cleaning to an entire DataFrame column.
    """
    df[target_column] = df[text_column].apply(clean_text)
    return df

if __name__ == "__main__":
    raw_sample = "A team of special forces OPERATIVES must rescue a kidnapped diplomat in Tokyo! Packed with action & high-octane explosive chases."
    cleaned = clean_text(raw_sample)
    print("Raw sample:", raw_sample)
    print("Cleaned sample:", cleaned)

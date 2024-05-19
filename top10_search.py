import requests
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import faiss
import re
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the Universal Sentence Encoder's TF Hub module
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def preprocess_text(text):
    """
    Preprocess the input text by removing punctuations and numbers,
    converting to lowercase, and removing excess whitespace.
    """
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuations and numbers
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Remove excess whitespace
    return text

def fetch_data(url):
    """
    Fetch data from the specified URL and return it as a JSON object.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from the API: {e}")
        return None

def extract_items(data):
    """
    Extract items from the API response.
    """
    if data and 'meta' in data and 'listItems' in data['meta']:
        return [item['title'] for item in data['meta']['listItems']]
    else:
        print("Invalid data format received from API.")
        return []

def generate_embeddings(items):
    """
    Generate embeddings for the provided list of items.
    """
    preprocessed_items = [preprocess_text(item) for item in items]
    return np.vstack([embed([item]).numpy() for item in preprocessed_items])

def create_faiss_index(embeddings):
    """
    Create a FAISS index from the provided embeddings.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search(query_text, index, k=10):
    """
    Perform a search on the FAISS index using the query text.
    """
    preprocessed_query = preprocess_text(query_text)
    query_vector = embed([preprocessed_query]).numpy()
    distances, indices = index.search(query_vector.astype('float32'), k)
    return distances, indices

# Fetch data from the public API
url = "https://list.ly/api/v4/meta?url=http://google.com"
data = fetch_data(url)

# Extract and preprocess items
items = extract_items(data)
if items:
    embeddings = generate_embeddings(items)

    # Create FAISS index
    index = create_faiss_index(embeddings)

    # Example Query
    query_text = "search query"
    distances, indices = search(query_text, index)

    # Display the results
    for i, idx in enumerate(indices[0]):
        print(f"Rank {i+1}: (Distance: {distances[0][i]})\n{items[idx]}\n")

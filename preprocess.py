"""
Preprocess the dataset for training and evaluation.
"""

stopwods_spanish = [
    "de", "la", "que", "el", "en", "y", "a", "los", 
    "del", "se", "las", "con", "un", "por", "para"]


def preprocess_data(data):
    """
    Preprocess the dataset by normalizing and tokenizing the text data.
    
    Args:
        string data: The raw text data to preprocess.
    Returns:
        list: A list of preprocessed tokens.
    """
    # Normalize the text (e.g., lowercasing, removing punctuation)
    normalized_data = data.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '').replace('¡', '').replace('¿', '')

    # Tokenize the text into words
    tokens = normalized_data.split()

    # Remove stop words
    tokens = [token for token in tokens if token not in stopwods_spanish]

    # Remove empty tokens
    tokens = [token for token in tokens if token.strip()]

    # Remove duplicates while preserving order
    seen = set()
    
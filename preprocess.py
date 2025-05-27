"""
Preprocess the dataset for training and evaluation.
"""

stopwods_spanish = [
    "de", "la", "que", "el", "en", "y", "a", "los", 
    "del", "se", "las", "con", "un", "por", "para",
    "lo", "al", "o", "una"]

symbols = [
    "¡", "¿", ":", ";", ",", ".", "!", "?", "-", "_", "(", 
    ")", "[", "]", "{", "}", "%", "º", "ª", "@", "#", "$",
    "&", "*", "+", "=", "<", ">", "/", "'", "`", "~", "«", 
    "»", "¨", "°", "«", "»", "“", "”"
]

numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def preprocess_data(data):
    """
    Preprocess the dataset by normalizing and tokenizing the text data.
    
    Args:
        string data: The raw text data to preprocess.
    Returns:
        list: A list of preprocessed tokens.
    """
    # Normalize the text (e.g., lowercasing, removing punctuation)
    lower_data = data.lower()

    # Remove symbols and punctuation
    normalized_data = ''.join(
        char for char in lower_data if (char not in symbols and char not in numbers)
    )

    # Tokenize the text into words
    tokens = normalized_data.split()

    # Remove stop words
    tokens = [token for token in tokens if token not in stopwods_spanish]

    return tokens
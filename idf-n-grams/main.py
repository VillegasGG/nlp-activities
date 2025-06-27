import math
from input import read_files
from preprocess import preprocess_data
import math

def generar_trigramas(texto):
    trigramas = []
    for i in range(len(texto) - 2):
        trigrama = texto[i:i+3]
        trigramas.append(trigrama)
    return trigramas


def n_grams_trichar(tokenized_texts):
    """
    Generate trichar n-grams from a list of tokenized texts.
    """
    result = []
    vocabulary = set()

    # Sepate by characters
    for text in tokenized_texts:
        # Flatten the list of tokens into a single string for character trigrams
        if isinstance(text, list):
            text_str = ''.join(text)
        else:
            text_str = text
        ngrams = generar_trigramas(text_str)
        result.append(ngrams)
        vocabulary.update(ngrams)
    
    vocabulary = sorted(vocabulary)  # Sort for consistency
    return result, vocabulary

def n_grams(tokenized_texts, n):
    """
    Generate n-grams from a list of tokenized texts.
    
    Args:
        tokenized_texts (list): List of lists containing tokens from each text.
        n (int): The size of the n-grams to generate.
    
    Returns:
        list: A list of n-grams.
        vocabulary: A sorted list of unique n-grams.
    """
    result = []
    vocabulary = set()

    for text in tokenized_texts:
        text_ngrams = []
        for index, _ in enumerate(text):
            # Generate n-gram
            ngram = ' '.join(text[index:index+n])
            if len(ngram.split()) == n:
                text_ngrams.append(ngram)
                vocabulary.add(ngram)
        result.append(text_ngrams)
    vocabulary = sorted(vocabulary)  # Sort for consistency
    
    return result, vocabulary

def idf_calculation(tokenized_texts, vocabulary):
    """
    Calculate the Inverse Document Frequency (IDF) for each term in the vocabulary.
    
    Args:
        tokenized_texts (list): List of lists containing tokens from each text.
        vocabulary (list): List of unique terms.
    
    Returns:
        dict: A dictionary with terms as keys and their IDF values as values.
    """
    idf = {}
    total_documents = len(tokenized_texts)

    # Count occurrences of each term in the vocabulary across all documents
    term_counts = {}
    print("Vocabulary:" + str(vocabulary))
    for term in vocabulary:
        print("Term:", term)
    for term in vocabulary:
        term_counts[term] = 0
    for text in tokenized_texts:
        for term in set(text):
            if term in term_counts:
                term_counts[term] += 1

    print("Term counts:", term_counts)
    # Calculate IDF for each term
    for term, count in term_counts.items():
        if count > 0:
            idf[term] = math.log(total_documents / count)
        else:
            idf[term] = 0.0

    return idf

def tf_idf(tokenized_texts, vocabulary_idf_dict):
    """
    Calculate the Term Frequency-Inverse Document Frequency (TF-IDF) for each text.

    Args:
        tokenized_texts (list): List of lists containing tokens (n-grams) from each text.
        vocabulary_idf_dict (dict): Dictionary with terms as keys and their IDF values as values.

    Returns:
        list: A list of dictionaries representing the normalized TF-IDF vectors for each text.
    """
    tf_idf_vectors = []

    for text in tokenized_texts:
        for index, _ in enumerate(text):
            # Calculate term frequency (TF)
            tf = {}
            for term in text:
                print("Term:", term)
                if term in tf:
                    tf[term] += 1
                else:
                    tf[term] = 1
            
            # Normalize TF
            total_terms = len(text)
            for term in tf:
                tf[term] /= total_terms
            
            # Calculate TF-IDF
            tf_idf_vector = {}
            for term, tf_value in tf.items():
                idf_value = vocabulary_idf_dict.get(term, 0.0)
                tf_idf_vector[term] = tf_value * idf_value
            
            tf_idf_vectors.append(tf_idf_vector)

    return tf_idf_vectors

def cosine_similarity(tokenized_text1, tokenized_text2, voc):
    """
    Calculate the cosine similarity between two Bag of Words representations.
    
    Args:
        bow1 (dict): Bag of Words representation of the first text.
        bow2 (dict): Bag of Words representation of the second text.
    
    Returns:
        float: Cosine similarity value between 0 and 1.
    """
    # Vectors
    vec1 = [tokenized_text1.get(palabra, 0) for palabra in voc]
    vec2 = [tokenized_text2.get(palabra, 0) for palabra in voc]

    print("Vector 1:", vec1)
    print("Vector 2:", vec2)

    # Calculate the dot product
    dot = sum(a * b for a, b in zip(vec1, vec2))

    # Calculate norms
    norm1 = math.sqrt(sum(a ** 2 for a in vec1))
    norm2 = math.sqrt(sum(b ** 2 for b in vec2))

    # Check for zero vectors to avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Calculate cosine similarity
    cos_sim = dot / (norm1 * norm2)

    return cos_sim

def main():

    # Read and preprocess the text files
    data = read_files(['texts/perro.txt', 'texts/gato.txt', 'texts/perro2.txt'])

    if not data:
        print("No data to process.")
        return
    
    preprocessed_data = []
    for text in data:
        tokens = preprocess_data(text)
        preprocessed_data.append(tokens)
    
    # Vocabulary creation
    vocabulary = set()

    for tokens in preprocessed_data:
        vocabulary.update(tokens)
    vocabulary = sorted(vocabulary)  # Sort for consistency
    print("\nVocabulary:", vocabulary)

    # n-grams creation
    n_grams_texts, vocabulary = n_grams_trichar(preprocessed_data)
    print("\nN-grams:")
    print(vocabulary)
    print("\nN-grams for each text:")
    for i, ngram in enumerate(n_grams_texts):
        print(f"File {i+1} n-grams: {ngram}")

    print("\nVocabulary size:", len(vocabulary))
    print("\nVocabulary:", vocabulary)

    # idf calculation
    idf = idf_calculation(n_grams_texts, vocabulary)
    
    # vectors
    tf_idf_vectors = tf_idf(n_grams_texts, idf)
    print("\nTF-IDF Vectors:")
    for i, vector in enumerate(tf_idf_vectors):
        print(f"File {i+1} TF-IDF Vector: {vector}")

    # Calculate cosine similarity between the first two texts
    if len(tf_idf_vectors) >= 2:
        cos_sim = cosine_similarity(tf_idf_vectors[0], tf_idf_vectors[1], vocabulary)
        print(f"\nCosine Similarity between File 1 and File 2: {cos_sim:.4f}")
    else:
        print("\nNot enough texts to calculate cosine similarity.")

    # Calculate cosine similarity between the first and third texts
    if len(tf_idf_vectors) >= 3:
        cos_sim = cosine_similarity(tf_idf_vectors[0], tf_idf_vectors[2], vocabulary)
        print(f"Cosine Similarity between File 1 and File 3: {cos_sim:.4f}")
    else:
        print("\nNot enough texts to calculate cosine similarity.")

    # Calculate cosine similarity between the second and third texts
    if len(tf_idf_vectors) >= 3:
        cos_sim = cosine_similarity(tf_idf_vectors[1], tf_idf_vectors[2], vocabulary)
        print(f"Cosine Similarity between File 2 and File 3: {cos_sim:.4f}")
    else:
        print("\nNot enough texts to calculate cosine similarity.")

    


main()
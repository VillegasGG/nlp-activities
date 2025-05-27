import math
from BoW.input import read_files
from BoW.preprocess import preprocess_data

# BoW
def create_bow_representation(tokens, vocabulary):
    """
    Create a Bag of Words representation from the tokens and vocabulary.
    
    Args:
        tokens (list): List of preprocessed tokens.
        vocabulary (list): List of unique words in the dataset.
    
    Returns:
        dict: A dictionary representing the Bag of Words.
    """
    bow = {word: 0 for word in vocabulary} 
    for token in tokens:
        if token in bow:
            bow[token] += 1
    return bow

def cosine_similarity(bow1, bow2, voc):
    """
    Calculate the cosine similarity between two Bag of Words representations.
    
    Args:
        bow1 (dict): Bag of Words representation of the first text.
        bow2 (dict): Bag of Words representation of the second text.
    
    Returns:
        float: Cosine similarity value between 0 and 1.
    """
    # Vectors
    vec1 = [bow1.get(palabra, 0) for palabra in voc]
    vec2 = [bow2.get(palabra, 0) for palabra in voc]

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
    data = read_files(['texts/delfines.txt', 'texts/megalodon.txt', 'texts/tormenta-solar.txt'])

    if not data:
        print("No data to process.")
        return
    
    preprocessed_data = []
    for text in data:
        tokens = preprocess_data(text)
        preprocessed_data.append(tokens)

    print("Preprocessed Data:")
    for i, tokens in enumerate(preprocessed_data):
        print(f"File {i+1} tokens: {tokens}")

    # Bag of Words representation
    
    # Vocabulary creation
    vocabulary = set()

    for tokens in preprocessed_data:
        vocabulary.update(tokens)
    vocabulary = sorted(vocabulary)  # Sort for consistency
    print("\nVocabulary:", vocabulary)

    # Create Bag of Words representation
    bow_representations = []
    for tokens in preprocessed_data:
        bow = create_bow_representation(tokens, vocabulary)
        bow_representations.append(bow)

    print("\nBag of Words Representations:")
    for i, bow in enumerate(bow_representations):
        print(f"File {i+1} BoW: {bow}")

    # Calculate cosine similarity between the first two files
    if len(bow_representations) >= 2:
        similarity = cosine_similarity(bow_representations[0], bow_representations[1], vocabulary)
        print(f"\nCosine Similarity between File 1 and File 2: {similarity:.4f}")
    else:
        print("\nNot enough files to calculate cosine similarity.")

    # Calculate cosine similarity between the first and third files
    if len(bow_representations) >= 3:
        similarity = cosine_similarity(bow_representations[0], bow_representations[2], vocabulary)
        print(f"Cosine Similarity between File 1 and File 3: {similarity:.4f}")
    else:
        print("Not enough files to calculate cosine similarity between File 1 and File 3.")

    # Calculate cosine similarity between the second and third files
    if len(bow_representations) >= 3:
        similarity = cosine_similarity(bow_representations[1], bow_representations[2], vocabulary)
        print(f"Cosine Similarity between File 2 and File 3: {similarity:.4f}")
    else:
        print("Not enough files to calculate cosine similarity between File 2 and File 3.")

if __name__ == "__main__":
    main()


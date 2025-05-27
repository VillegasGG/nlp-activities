from input import read_files
from preprocess import preprocess_data

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

        

if __name__ == "__main__":
    main()


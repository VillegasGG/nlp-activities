from input import read_files
from preprocess import preprocess_data

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
    print("\nVocabulary:")
    print(vocabulary)

if __name__ == "__main__":
    main()


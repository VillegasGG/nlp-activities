"""
Open n files and read their contents
"""

def read_files(files: list):
    """
    Read the contents of each file return a list of their contents.
    """

    contents = []
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                contents.append(f.read())
        except FileNotFoundError:
            print(f"File {file} not found.")
        except IOError as e:
            print(f"Error reading file {file}: {e}")

    return contents
import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def create_vectorizer(csv_processed_file="processed_speeches.csv"):

    with open(csv_processed_file, "r", encoding='utf8') as speeches:
        # Skip the column names
        speeches.readline()

        all_lines = speeches.readlines()

        all_speeches = [line.split(',')[-1] for line in all_lines]

        transformer = TfidfTransformer()

        transformer.fit(all_speeches)

        # Save it for future uses
        pickle.dump(transformer, open("tfidf/transformer.pkl", "wb"))


def create_vector_per_member(csv_processed_file="processed_speeches.csv", transformer_file="tfidf/transformer.pkl",
                             group_dictionary_file="group_dictionary.pkl"):

    with open(csv_processed_file, "r") as speeches:

        # List with vector per member
        vector_per_member = []

        # Load transformer.
        with open(transformer_file, "rb") as file:
            transformer = pickle.load(file)

        # Load the group dictionary.
        with open(group_dictionary_file, "rb") as file:
            group_dictionary = pickle.load(file)

        member_dictionary = group_dictionary["Member"]

        for member, doc_ids in member_dictionary.items():
            temp = []
            for doc_id in doc_ids:
                temp.append()

import csv
import pickle
import numpy as np
import pandas as pd
import re
import chardet
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from utils.preprocess import preprocess_text
from utils.general import create_folder


def read_pre_file(name="processed_speeches.csv"):
    with open(name, encoding='utf-8') as csv_file_read:
        csv_reader = csv.reader(csv_file_read, delimiter=',', quotechar='"')

        for i, row in enumerate(csv_reader):
            print(row)


def create_pre_processed_file(csv_file_name, processed_file_name="processed_speeches/processed_speeches.csv", limit=2):
    """
        Create a new processed .csv based on the given file, where the "speeches" column has taken preprocess
        (remove punctuation, stopwords, accentuation and do stemming). Additionally, a dictionary is created with the
        offset of each speech id.

        Parameters:
        csv_file_name (str): The original .csv file, containing the speeches.
        processed_file_name (str): The name of the processed .csv file.

    """
    create_folder('processed_speeches')
    with open(csv_file_name, "r", encoding='utf-8') as csv_file_read, open(processed_file_name, 'w', encoding='utf-8',
                                                                           newline='') as csv_file_write:

        csv.field_size_limit(1000000)
        doc_id_to_offset = {}

        # Read all lines at once and then iterate for each one.
        lines = csv_file_read.readlines()
        offset = 0
        for doc_id, row in enumerate(lines):

            if limit == doc_id:
                break

            # Save the first row (column row).
            if doc_id == 0:
                offset += len(row.encode("utf-8")) + len('\n'.encode("utf-8"))
                csv_file_write.write(f"{row}")
                continue

            # Save the current offset and create the offset for the next speech row.
            doc_id_to_offset[doc_id-1] = offset
            offset += len(row.encode("utf-8")) + len('\n'.encode("utf-8"))

            # Process speeches.
            row = re.split(',(?=(?:[^"]*"[^"]*")*[^"]*$)', row.strip())
            text = preprocess_text(row[-1])
            temp = row[:10]

            # Concatenate the new processed speech to the old row and save.
            temp.append(text)
            temp = ",".join(temp)
            csv_file_write.write(f'{temp}\n')

        # Save the id offsets of the original speeches (for the search procedure).
        with open("id_to_offset.pkl", "wb") as f:
            pickle.dump(doc_id_to_offset, f)


def test_offsets(k = 10, all_speeches=[]):
    dict = pickle.load(open('id_to_offset.pkl', "rb"))

    with open('Greek_Parliament_Proceedings_1989_2020.csv', "r", encoding="utf8") as f:
        for i in range(k):
            f.seek(dict[i+1])
            print(f'doc_id {i} offset {dict[i+1]}')
            line = f.readline()
            print(f' line {line} : {all_speeches[i]}\n\n\n')

def create_inverted_catalog_file(csv_processed_file, inverted_catalog_file="inverted_catalog.txt", limit=-1):
    '''
    This function creates the inverted catalog of the project.
    Inverted Catalog has the format: term, total_frequency, doc_id1, freq1, doc_id2, freq2, ...
    We calculate the frequencies and the doc_ids by calculating the CountVectorizer, that counts frequency of all terms
    in all documents. Then transform this matrix to a compressed version specialized in Columns in order to calculate the
    frequencies and in Rows in order to calculate each document unique term length.


    :param csv_processed_file: Processed speeches file.
    :param inverted_catalog_file: Output Inverted Catalog file name.
    '''
    create_folder('inverted_catalog')
    sample_text = 100
    with open(inverted_catalog_file, "w+", encoding='utf8') as inverted_catalog:
        # Read dataframe.
        df = pd.read_csv(csv_processed_file, nrows=sample_text)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', None)
        print(df)
        # Create dictionary that maps terms to file index.
        term_indexes = {}

        # Read all speeches.
        df['speech'].fillna(" ", inplace=True)
        all_speeches = df['speech']

        # Fit and save for future use.
        transformer = TfidfVectorizer(lowercase=False)
        transformer.fit(all_speeches)
        tfidf_per_speech = transformer.transform(all_speeches)

        vectorizer = CountVectorizer(lowercase=False)


        # Create nxm matrix, where n is the number of documents and m the number of total distinct terms.
        count_matrix = vectorizer.fit_transform(all_speeches)

        terms = vectorizer.get_feature_names_out()
        num_of_docs, num_of_terms = count_matrix.shape

        # Compress the sparse count_matrix.
        compressed_matrix = count_matrix.tocsc()
        print(f'compressed {compressed_matrix}')

        print(f'count matrix shape {count_matrix.shape}')



        for term_pos in range(num_of_terms):

            # Get the document indexes and frequency that term shows up.
            term_col = compressed_matrix.getcol(term_pos).tocoo()
            
            # Create Inverted Row for the term (term_name, n_term, (doc1, freq1), (doc5, freq5),...,(docn, freqn))
            term_name = terms[term_pos]
            term_array = [str(term_name)]
            tern_n = len(term_col.data)
            for doc_id, term_freq in zip(term_col.row, term_col.data):
                term_array.extend([str(doc_id), str(term_freq)])

            term_array.insert(1, str(tern_n))
            term_array = ','.join(term_array)

            term_indexes[term_name] = inverted_catalog.tell()
            inverted_catalog.write(term_array + "\n")

        doc_id_to_length = {}
        compressed_matrix = count_matrix.tocsr()

        # Calculate lengths of each doc.
        for doc_id in range(num_of_docs):
            doc_row = compressed_matrix.getrow(doc_id).tocoo()
            doc_length = np.linalg.norm(doc_row.data)
            doc_id_to_length[doc_id] = doc_length

    # Save the indexes.
    create_folder('inverted_catalog')
    pickle.dump(term_indexes, open("inverted_catalog/inverted_offsets.pkl", "wb"))

    # Save lengths.
    pickle.dump(doc_id_to_length, open("inverted_catalog/doc_lengths.pkl", "wb"))

    # Save transformer from all speeches.
    create_folder('transformer')
    pickle.dump(transformer, open("transformer/transformer.pkl", "wb"))

    # Save tfidf per speech.
    create_folder('group')
    pickle.dump(tfidf_per_speech, open("group/tf idf_per_speech.pkl", "wb"))

import pickle
from sklearn.neighbors import KDTree
from utils.general import create_folder
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

def calculate_top_k(tfidf_per_member="group/tfidf_per_member.pkl", k=100):
    '''
    This function load the dictionary with the csr transformed tfidf matrix for each member. Then concatenate all of them
    while transforming them to the original dense form, and then calculate the full csr for all tfidf matrices combined.
    This way we can calculate cosine similarity extremely fast without having any zeros in the equation.
    Finally the similarity matrix is calculated and the most similar pairs are extracted (excluding themselves)

    :param tfidf_per_member: This dictionary maps each member of the parliament to the tfidf matrix calculated
    from all his speeches
    :param k: The number of top pairs saved in disk.
    '''

    # Load the dict and transform the values to CSR matrices again.
    tfidf_per_member = pickle.load(open(tfidf_per_member, "rb"))
    keys = list(tfidf_per_member.keys())
    values = sp.csr_matrix([tfidf_per_member[k].toarray().astype(np.float32).flatten() for k in keys])
    print('Values ready')
    del tfidf_per_member
    # Calculate cosine similarities for each document pair.
    similarities = cosine_similarity(values)
    print('Similarity matrix calculated')
    # Get the indices of the most similar pairs excluding self-similarities.
    np.fill_diagonal(similarities, -np.inf)
    indices = np.argsort(similarities, axis=None)[::-1][:k]
    i_indices, j_indices = np.unravel_index(indices, similarities.shape)
    print(f'i_ind {i_indices} j_ind {j_indices}')
    top_k_pairs = []
    for i, j in zip(i_indices, j_indices):
        if i < j:
            top_k_pairs.append([keys[i], keys[j], str(round(similarities[i, j],2))])
    print(f'Pair names and similarity scores calculated.. time to save {top_k_pairs}')
    # Save top-k similar pairs.
    create_folder('similarity')
    pickle.dump(top_k_pairs, open('similarity/top_k_pairs.pkl', "wb"))









'''def calculate_top_k(tfidf_per_member="group/tfidf_per_member.pkl", k=100):
    
    This function gets a dictionary that maps each member to the Tfidf matrix of all speeches and then finds the pairs
    of most similar Tfidf matrices, translate it into the keys of these matrices and returns the most similar key pairs.
    :param tfidf_per_member: The dictionary that maps each member to its Tfidf matrix calculated from all his
    concatenated speeches.
    :param k: Store the top-k pairs.
    :return:
    
    create_folder('similarity')

    # Load the dict and transform the values to dense matrices again.
    tfidf_per_member = pickle.load(open(tfidf_per_member, "rb"))
    keys = list(tfidf_per_member.keys())
    values = []
    for key, value in tfidf_per_member.items():
        values.append(value.toarray().reshape(-1).astype(np.float16))

    print(f'Compressed form of dictionary reseted back to dense form.')
    # Get the values of the dict, build the KDTree and find the indices of the most similar value pairs.
    #keys = list(tfidf_per_member.keys())
    #print(f'Keys')
    #values = list(tfidf_per_member.values())
    #print(f'Values')
    del tfidf_per_member
    print(f'Deleted')
    tree = KDTree(values)
    indices = tree.query(values, k=3)[1]
    print(f'Indices returned')
    del values
    # Get the indices and map the value pairs to key pairs in order to show it to the user in the future. (The first
    # index is skipped due to being itself).
    indices = indices[:, 1:]
    top_k_pairs = []

    for i, row in enumerate(indices):
        row_keys = [keys[j] for j in row]
        top_k_pairs.append(row_keys)

    # Save top-k similar pairs.
    pickle.dump(top_k_pairs, open('similarity/top_k_pairs.pkl', "wb"))'''

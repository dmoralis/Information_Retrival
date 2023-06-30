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

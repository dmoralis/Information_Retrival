import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from utils.general import create_folder
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def create_lsa(tfdf_per_speech="tfidf_per_speech.pkl"):
    '''
    This function take the tfidf matrices of all speeches and transform them to lsi

    :param tfdf_per_speech: Tfidf matrices of all speeches
    '''
    tfidf_per_speech = pickle.load(open("group/tfidf_per_speech.pkl", "rb"))

    svd = TruncatedSVD(n_components=100)
    svd.fit(tfidf_per_speech)
    svd_matrix = svd.transform(tfidf_per_speech)[:, :60]
    print('SVD Created')
    create_folder('lsa')
    pickle.dump(svd_matrix, open("lsa/svd_matrix.pkl", "wb"))
    pickle.dump(svd.components_, open("lsa/Vt.pkl", "wb"))


def create_clusters(svd_matrix_file="lsa/svd_matrix.pkl", lsa_components_file='lsa/Vt.pkl', n=9, calculate_clusters=False):
    '''
    This function calculate the Elbow method in order to figure out the best k in kmeans algorithm.
    In case of "calculate_clusters=False' the elbow method is applies and in case of true the given param n
    is used in the final kmeans and the results is savd

    :param svd_matrix_file: LSI matrix with dimension reduction
    :param n:
    :param calculate_clusters:
    :return:
    '''

    svd_matrix = pickle.load(open(svd_matrix_file, "rb"))

    # Elbow Method.
    if not calculate_clusters:
        wcss = []
        for i in range(1, 100):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(svd_matrix)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 100), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()

        silhouette_scores = []
        for k in range(2, 100):
            # Fit K-means clustering model
            kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(svd_matrix)

            # Compute average silhouette score
            labels = kmeans.labels_
            score = silhouette_score(svd_matrix, labels)
            silhouette_scores.append(score)

        # Plot silhouette scores for different values of K
        plt.plot(range(2, 100), silhouette_scores)
        plt.title('Silhouette Score')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette score')
        plt.show()
    else:
        # Clusters construction.
        kmeans = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=10, random_state=0)
        clustered_svd = kmeans.fit_predict(svd_matrix)

        # Components.
        components = pickle.load(open(lsa_components_file, "rb"))

        # Tfidf Matrix.
        tfidf_model = pickle.load(open('transformer/transformer.pkl', "rb"))
        terms = tfidf_model.get_feature_names_out()
        print(f'terms {terms}')

        # Compute the centroid of each cluster in LSA space
        centroids = np.array([svd_matrix[clustered_svd == i].mean(axis=0) for i in range(kmeans.n_clusters)])

        # Identify the most relevant dimensions for each cluster
        num_terms = 2
        relevant_dims = [np.argsort(-np.abs(centroid))[:num_terms] for centroid in centroids]

        # Extract the top terms for each cluster based on the most relevant dimensions
        top_terms = []
        for i in range(kmeans.n_clusters):
            cluster_top_terms = []
            for dim in relevant_dims[i]:
                weights = components[dim]
                sorted_indices = np.argsort(weights)[::-1]
                top_indices = sorted_indices[:num_terms]
                cluster_top_terms.extend([terms[idx] for idx in top_indices])
            top_terms.append(cluster_top_terms)
            print(f"Cluster {i} top terms: {cluster_top_terms}")


        print(clustered_svd)
        pickle.dump(clustered_svd, open("lsa/clustered_svd.pkl", "wb"))

#create_lsa()
#create_clusters(n=9, calculate_clusters=True)


'''def calculate_top_k(tfidf_per_member="group/tfidf_per_member.pkl", k=100):
   

    # Load the dict and transform the values to CSR matrices again.
    tfidf_per_member = pickle.load(open(tfidf_per_member, "rb"))

    # Load keys and transform the combined tfidf matrix to csr.
    keys = list(tfidf_per_member.keys())
    values = sp.csr_matrix([tfidf_per_member[k].toarray().flatten() for k in keys])
    del tfidf_per_member

    # Calculate cosine similarities for each document pair.
    member_similarities = []
    for i in range(values.shape[0]):
        similarity = calculate_cosine_similarity((i, values))
        print(f'Simialarity {similarity} i {i}')
        member_similarities.append(similarity)
    print('Document similarities calculated')
    # Convert the list of cosine similarity arrays to a 2D numpy array.
    similarities = np.vstack(member_similarities)
    print('Stacked')
    # Get the indices of the most similar pairs excluding self-similarities.
    indices = np.argsort(similarities, axis=None)[::-1][:k]
    i_indices, j_indices = np.unravel_index(indices, similarities.shape)
    mask = i_indices != j_indices
    i_indices, j_indices = i_indices[mask], j_indices[mask]

    print('Sorted')
    # Get the keys of the most similar pairs.
    top_k_pairs = [(keys[i_indices[i]], keys[j_indices[i]]) for i in range(k)]
    print('Pair names calculated.. time to save')
    # Save top-k similar pairs.
    print(top_k_pairs)
    pickle.dump(top_k_pairs, open('similarity/top_k_pairs.pkl', "wb"))'''
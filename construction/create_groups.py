import csv
import os
import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp

from collections import defaultdict
from utils.general import create_folder


def calculate_keywords(grouped, transformer, one_group_filter=True, save_tfidf=False,
                       file_name=''):
    """
    This function combines all the speeches for each group of the given filter and calculate the Tfidf Matrix.
    Then the Tfidf Matrix is sorted along with all terms and the top 20 keywords are saved with its Tfidf weights.
    The results are saved in a .txt file and the offsets of the particular group are saved in a dictionary with the format:
    {(group1:offset1), ...} in case of one group (member or party) or {((group1, group2):offset), ...}
    in case of two groups (member and date or party and date).
    :param grouped: The groups created by a particular filter in the pandas dataframe.
    :param transformer: Saved transformer in order to calculate the Tfidf Matrix for each group.
    :param one_group_filter: In case of one group (member or party)
    :param save_tfidf: In case of saving the Tfidf in a dictionary in compressed csr form.
    :param file_name: Wanted .txt file name.
    :return:
    """
    with open(file_name, "w", encoding='utf-8') as file:

        file_offset_dict = {}

        if save_tfidf:
            tfidf_offset_dict = {}

        feature_names = transformer.get_feature_names_out()
        for name, group in grouped:
            speeches = ' '.join(group["speech"])
            # Calculate most important keywords per year
            tfidf_matrix = transformer.transform([speeches]).toarray().reshape(-1, 1)

            if save_tfidf:
                tfidf_offset_dict[name] = sp.csr_matrix(tfidf_matrix)

            scores = zip(feature_names, tfidf_matrix)
            sorted_words = sorted(scores, key=lambda x: x[1], reverse=True)

            if one_group_filter:
                file_offset_dict[name] = file.tell()
            else:
                file_offset_dict[(name[1], name[0])] = file.tell() #dict with (name, date) = offset
            file.write(','.join(f'{str(k)},{str(float(v))}' for k, v in sorted_words[:20]) + '\n')

        if save_tfidf:
            pickle.dump(tfidf_offset_dict, open('group/tfidf_per_member.pkl', "wb"))

        # Save file's offset dict
        pickle.dump(file_offset_dict, open(f'group/{file_name.split("/")[-1].split(".")[0]}_dict.pkl', "wb"))




def create_groups(speeches="processed_speeches.csv", transformer_file="transformer.pkl"):
    '''
    This functions read all csv files, and create group based on member and party column.
    Then it creates a new column named 'Year' and create groups based on (member, year) ,(party, year) and (speech, year)
    For each of these filters calculate_keyword() is called in order to save the top keywords for each group created by
    the filters.
    :param speeches: Processed .csv file.
    :param transformer_file: Saved transformers file, trained on the while corpus.
    '''
    with open(transformer_file, "rb") as file:
        transformer = pickle.load(file)

    if not os.path.exists('group'):
        os.makedirs('group')

    df = pd.read_csv(speeches, encoding='utf8')

    # Drop the rows that have blank speeches
    df.dropna(subset=['speech'], inplace=True)

    member_grouped = df.groupby(["member_name"])
    print(member_grouped)

    calculate_keywords(member_grouped, transformer, save_tfidf=True, file_name='group/member_all_time.txt')
    print('First completed')

    party_grouped = df.groupby(["political_party"])
    calculate_keywords(party_grouped, transformer, file_name='group/party_all_time.txt')
    print('Second completed')

    # Per year keyword calculation
    df["Year"] = pd.to_datetime(df["sitting_date"], dayfirst=True).dt.year

    member_grouped_per_year = df.groupby(["Year", "member_name"])
    calculate_keywords(member_grouped_per_year, transformer, one_group_filter=False, file_name='group/member_per_year.txt')
    print('Third completed')

    party_grouped_per_year = df.groupby(["Year", "political_party"])
    calculate_keywords(party_grouped_per_year, transformer, one_group_filter=False, file_name='group/party_per_year.txt')
    print('Fourth completed')

    speech_grouped_per_year = df.groupby(["Year"])
    calculate_keywords(speech_grouped_per_year, transformer, file_name='group/speech_per_year.txt')
    print('Fifth completed')

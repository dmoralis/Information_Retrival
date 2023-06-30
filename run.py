import time
import pickle

import chardet
import pandas as pd
import os
import re

from math import log
from collections import defaultdict
from construction.create_inverted_catalog import create_pre_processed_file, create_inverted_catalog_file
from construction.create_groups import create_groups
from construction.create_top_k_similar import calculate_top_k
from construction.create_lsa import create_lsa, create_clusters
from utils.preprocess import preprocess_text
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool, freeze_support


class EngineManager:
    def __init__(self):
        self.inverted = None
        self.id_to_offset = None
        if os.path.exists('inverted_catalog/inverted_catalog.txt') and os.path.exists(
                'inverted_catalog/inverted_offsets.pkl'):
            self.inverted = open('inverted_catalog/inverted_catalog.txt', "r")
            self.id_to_offset = pickle.load(open('inverted_catalog/inverted_offsets.pkl', "rb"))

    def construct(self):
        t1 = time.time()
        create_pre_processed_file("Greek_Parliament_Proceedings_1989_2020.csv",
        processed_file_name="processed_speeches.csv", limit=-1)
        print(f'Preprocess done')
        t2 = time.time()
        create_inverted_catalog_file("processed_speeches.csv", "inverted_catalog/inverted_catalog.txt")
        print(f'Inverted Created')
        t3 = time.time()
        create_groups("processed_speeches.csv", "transformer/transformer.pkl")
        print(f'Groups Created')
        t4 = time.time()
        calculate_top_k('group/tfidf_per_member.pkl')
        t5 = time.time()
        print(f'Top-k Calculated')
        create_lsa(tfdf_per_speech="group/tfidf_per_speech.pkl")
        print(f'LSA Created')
        t6 = time.time()
        create_clusters("lsa/svd_matrix.pkl", calculate_clusters=True, n=80)
        #print(f'Clusters Created')
        #t7 = time.time()
        #print(
        #    f'Create Groups:{t4 - t3}\nCreate top-k similar:{t5 - t4}'
        #    f'\nCreate lsa:{t6 - t5}\nCreate Clusters{t7 - t6}\nTotal time:{t7 - t1}s')

    def calculate_idf(self, total_doc, term_n):
        return log(1 + total_doc / term_n)

    def calculate_tf(self, term_freq):
        return 1 + log(term_freq)

    def get_speech(self, doc_id, speeches, id_to_offset, i=0):

        offset = id_to_offset[doc_id]
        speeches.seek(offset)
        row = speeches.readline()
        ##if i:
        ##    print(f'doc_id {doc_id} file {row}')
        #x = ",".join(k.split(',')[10:]).strip().replace('"', '')
        speech = re.split(',(?=(?:[^"]*"[^"]*")*[^"]*$)', row.strip())
        return [speech[-1], speech[0], speech[1]] # speech, name, date

    def get_speeches(self, doc_ids):
        speeches_array = []
        with open('Greek_Parliament_Proceedings_1989_2020.csv', "r", encoding="utf-8") as speeches:
            id_to_offset = pickle.load(open("id_to_offset.pkl", "rb"))
            #print(f'id_to_offset {id_to_offset}')
            for i, doc_id in enumerate(doc_ids):
                if i < 20:
                    _ = self.get_speech(doc_id, speeches, id_to_offset, i=1)
                speeches_array.append(self.get_speech(doc_id, speeches, id_to_offset))

        return speeches_array

    def filter_results(self, attr, scores):
        threshold_score = 0.2
        length = 0
        for i in range(len(scores)):
            if scores[i] > threshold_score:
                length = i
            else:
                break
        return attr[:length], scores[:length]

    def query_search(self, query):
        
        t1 = time.time()
        # Load dictionaries
        with open('inverted_catalog/inverted_catalog.txt', "r", encoding='utf8') as inverted_catalog:
            inverted_catalog_index = pickle.load(open('inverted_catalog/inverted_offsets.pkl', "rb"))
            doc_lengths = pickle.load(open('inverted_catalog/doc_lengths.pkl', "rb"))
            t2 = time.time()
            # Preprocess query and get keywords
            query = preprocess_text(query)
            ##print(f'Query {query}')
            keywords = query.split(' ')
            ##print(f'keyowrds {keywords}')
            total_doc = len(inverted_catalog_index)
            #print(f'Dict {inverted_catalog_index}\n Keywords {keywords}')
            doc_rankings = defaultdict(float)

            for term in keywords:
                if inverted_catalog_index.get(term, -1) != -1:
                    offset = inverted_catalog_index[term]
                    ##print(f'Offset {offset}')
                    inverted_catalog.seek(offset)
                    term_line = inverted_catalog.readline().split(',')
                    ##print(f'Term line {term_line}')
                    # print(f'Term line {term_line}')
                    _, term_n, doc_and_frequency = term_line[0], term_line[1], term_line[2:]
                    idf = self.calculate_idf(total_doc, int(term_n))

                    for i in range(0, len(doc_and_frequency), 2):
                        doc_id = int(doc_and_frequency[i])
                        doc_freq = int(doc_and_frequency[i + 1])
                        #print(f'doc_id {doc_id} doc_freq {doc_freq}')
                        # if doc_id not in doc_rankings.keys():
                        #    doc_rankings[doc_id] = 0

                        tf_td = self.calculate_tf(doc_freq)

                        doc_rankings[doc_id] += tf_td * idf

            for doc_id, score in doc_rankings.items():
                doc_rankings[doc_id] = score / int(doc_lengths[doc_id])
            #print(sorted(doc_rankings.items(), key=lambda x: x[1], reverse=True)[:20])

            doc_rankings = dict(sorted(doc_rankings.items(), key=lambda x: x[1], reverse=True))

            # print(f'Doc rankings {doc_rankings}')

            t3 = time.time()
        t4 = time.time()
        print(f'Total {t4 - t1} Query match {t3 - t2}')

        return self.filter_results(self.get_speeches(doc_rankings.keys()), list(doc_rankings.values()))

    def return_line_from_offset(self, file_name, offset):
        with open(file_name, "r", encoding="utf8") as file:
            file.seek(offset)
            line = file.readline()
        return line

    def top_keywords(self, name, category="Member"):

        top_keyword_dict = None
        top_keyword_per_year_dict = None

        if category.lower() == "member":
            top_keyword_dict = pickle.load(open('group/member_keywords_all_time.pkl', "rb"))
            top_keyword_per_year_dict = pickle.load(open('group/member_keywords_per_year.pkl', "rb"))
        elif category.lower() == "party":
            top_keyword_dict = pickle.load(open('group/party_keywords_all_time.pkl', "rb"))
            top_keyword_per_year_dict = pickle.load(open('group/party_keywords_per_year.pkl', "rb"))
        else:
            # keyword_dict = pickle.load(open('group/'))
            print('dummy')

        if top_keyword_dict.get(name, 0):
            top_keywords = top_keyword_dict[name][:10]
            top_keyword_per_year_dict = top_keyword_per_year_dict[name]
            #print(top_keywords)
            #print(top_keyword_per_year_dict)
        else:
            print(f'{category} {name} not found\n Try one of these: {top_keyword_dict.keys()}')

    '''def search_lsa(self, query):
        query = preprocess_text(query)

        svd_matrix = pickle.load(open('lsa/svd_matrix.pkl', "rb"))
        Vt = pickle.load(open('lsa/Vt.pkl', "rb"))
        transformer = pickle.load(open('transformer/transformer.pkl', "rb"))

        query = transformer.transform([query])
        query = query @ Vt.T

        similarity_scores = cosine_similarity(query, svd_matrix)

        most_similar_ids = similarity_scores.argsort()[:, :][0]

        print(most_similar_ids)

        return self.get_speeches(most_similar_ids)'''

    def search_lsa(self, query):
        query = preprocess_text(query)

        svd_matrix = pickle.load(open('lsa/svd_matrix.pkl', "rb"))
        Vt = pickle.load(open('lsa/Vt.pkl', "rb"))
        transformer = pickle.load(open('transformer/transformer.pkl', "rb"))

        query = transformer.transform([query])
        query = query @ Vt.T[:, :60]

        similarity_scores = cosine_similarity(query, svd_matrix)

        # Get the indices of the most similar documents
        most_similar_ids = similarity_scores.argsort()[0][::-1]
        # Get the similarity scores of the most similar documents
        similarity_scores = similarity_scores[0][most_similar_ids]

        # Create a list of tuples containing the document ids and their similarity scores
        # results = [(self.get_speech_id(most_similar_ids[i]), similarity_scores[i]) for i in
        #           range(len(most_similar_ids))]

        return self.filter_results(self.get_speeches(most_similar_ids), similarity_scores)

    def top_k_pairs(self, k=5):
        top_k = pickle.load(open('similarity/top_k_pairs.pkl', "rb"))
        return top_k[:k]


    def summary(self, text, max_length=512):
        from transformers import pipeline
        import re
        
        # translator = pipeline("translation", "Helsinki-NLP/opus-mt-grk-en")
        translator = pipeline("translation", "Helsinki-NLP/opus-mt-grk-en")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        translator_el = pipeline("translation", model="Helsinki-NLP/opus-mt-en-el")
    
        translated_text = ""
        text_length = len(text.split(' '))
        text = text.replace('"','').replace("'",'')
        text = re.split(r'\.(?!\d)', text)
        t1 = time.time()
        dot_char=''
        for i, sentence in enumerate(text):
            translated_text += dot_char + translator(sentence, max_length=512)[0]["translation_text"]
            dot_char='.'
        t2 = time.time()
        summarized_text = summarizer(translated_text, min_length=int(text_length/2), max_length=int(text_length/1.5))[0]['summary_text']
        t3 = time.time()
        greek_text = translator_el(summarized_text)[0]["translation_text"]
        t4 = time.time()
        print(f"Length before {text_length} Lenght after {len(greek_text.split(' '))}")
        print(f'EL-TO-EN TIME : {t2 - t1}s SUMMARY TIME: {t3 - t2}s EN-TO-EL: {t4 - t3}s')
        return greek_text
if __name__ == '__main__':
    manager = EngineManager()
    #manager.construct()
    #manager.summary('''Η τεχνητή νοημοσύνη (ΤΝ) αναδύεται ως μια από τις πιο επαναστατικές και επηρεαστικές τεχνολογίες των τελευταίων δεκαετιών. Οι επιπτώσεις της ΤΝ στην κοινωνία είναι ευρείας κλίμακας και πολυδιάστατες, καθιστώντας την ένα θέμα ενδιαφέροντος και συζήτησης. Παρακάτω, θα εξετάσουμε μερικές από τις κύριες επιπτώσεις της ΤΝ στην κοινωνία.
    #    Στον τομέα της απασχόλησης, η ΤΝ έχει ήδη αρχίσει να επηρεάζει τις αγορές εργασίας. Οι αυτοματοποιημένες διαδικασίες και οι ρομποτικές τεχνολογίες έχουν ήδη αντικαταστήσει ορισμένες εργασίες που παλαιότερα διεκπεραιώνονταν από ανθρώπους. Αυτό δημιουργεί ανησυχίες για τη μελλοντική απασχόληση και την ανισότητα στην πρόσβαση στην απασχόληση.
    #    Η ΤΝ επηρεάζει επίσης την υγεία και την ιατρική. Η ανάπτυξη αλγορίθμων μηχανικής μάθησης και το βαθύ μάθημα έχουν επιφέρει σημαντική πρόοδο σε πεδία όπως η ανίχνευση ασθενειών, η διάγνωση και οι θεραπευτικές προσεγγίσεις. Οι αλγόριθμοι μηχανικής μάθησης μπορούν να αναλύσουν μεγάλες ποσότητες δεδομένων και να αναγνωρίσουν πρότυπα και τάσεις που είναι δύσκολο να ανιχνευθούν από τον ανθρώπινο εγκέφαλο.
    #    Αυτό επιτρέπει στους ιατρούς να έχουν πιο ακριβείς διαγνώσεις και να επιλέξουν την κατάλληλη θεραπεία για τον κάθε ασθενή.
    #    Επιπλέον, η ΤΝ έχει επίσης επιδράσει στην κοινωνία μέσω των αλληλεπιδράσεων με τους ανθρώπους. Οι εικονικοί βοηθοί και οι ρομποτικές συσκευές παίζουν σημαντικό ρόλο στην καθημερινή ζωή, παρέχοντας υπηρεσίες όπως η ενημέρωση, η ψυχαγωγία και η βοήθεια σε ηλικιωμένους και ανθρώπους με αναπηρία. Αυτή η αλληλεπίδραση με την ΤΝ ανοίγει νέους τρόπους επικοινωνίας και αλληλεπίδρασης με τεχνητούς συνομιλητές, προσφέροντας μια πιο εξατομικευμένη εμπειρία για τον χρήστη.
    #    Ωστόσο, η ΤΝ έχει επίσης προκαλέσει ανησυχίες και διλήμματα.
    #    Η θέση της στην προστασία της ιδιωτικότητας και των δεδομένων αποτελεί ένα σημαντικό ζήτημα.
    #    Η συλλογή και ανάλυση μεγάλων ποσοτήτων προσωπικών δεδομένων μπορεί να δημιουργήσει ανησυχίες για την ανεπάρκεια των μέτρων προστασίας και την πιθανότητα κατάχρησης των πληροφοριών.''')
# manager.get_speech(0)
# manager.get_speech(1)
# manager.get_speech(3)

# manager.query_search('Ο Ελληνικός λαός δεν θα πεινάσει. Πεινάς ρε;')
# manager.top_keywords('δημοκρατικη ανανεωση', 'Party')
# manager.top_k_pairs(k=5)
# manager.search_lsa('Ο Ελληνικός λαός δεν θα πεινάσει. Πεινάς ρε;')

import json

from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from run import EngineManager
from builtins import len
import pickle
import numpy as np

manager = EngineManager()

app = Flask(__name__)
app.secret_key = 'information_retrieval'

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/top_keywords')
def top_keywords():
    return render_template('top_keywords.html')

@app.route('/top_similar')
def top_similar():
    return render_template('top_similar.html')

@app.route('/get_options', methods=['POST'])
def get_options():
    selected_field = request.form.get('selected_field')

    if selected_field == 'Μέλος της βουλής':
        dict = pickle.load(open('group/member_all_time_dict.pkl', "rb"))
    elif selected_field == 'Κόμμα της βουλής':
        dict = pickle.load(open('group/party_all_time_dict.pkl', "rb"))
    else:
        dict={}

    field2 = dict.keys()

    option_values = field2
    #print(f'field2 {field2}')

    return json.dumps(list(option_values))

@app.route('/get_dates', methods=['POST'])
def get_dates():
    selected_field1 = request.form.get('selected_field1')
    selected_field2 = request.form.get('selected_field2')

    if selected_field1 == 'Μέλος της βουλής':
        dict = pickle.load(open('group/member_per_year_dict.pkl', "rb"))
    elif selected_field1 == 'Κόμμα της βουλής':
        dict = pickle.load(open('group/party_per_year_dict.pkl', "rb"))
    else:
        dict = []

    name = selected_field2
    results = [str(key[1]) for key in dict.keys() if key[0] == name]
    results.insert(0, f'{results[0]}-{results[-1]}')
    return json.dumps(results)

@app.route('/get_keywords', methods=['POST'])
def get_keywords():
    selected_field1 = request.form.get('selected_field1')
    selected_field2 = request.form.get('selected_field2')
    selected_field3 = request.form.get('selected_field3')
    print(f'Get keywords for {selected_field2}:{selected_field3}')
    results = []
    if '-' in selected_field3:
        if selected_field1 == 'Μέλος της βουλής':
            dict = pickle.load(open('group/member_all_time_dict.pkl', "rb"))
            print(f'member all {dict}')
            results = manager.return_line_from_offset('group/member_all_time.txt', dict[selected_field2])
        elif selected_field1 == 'Κόμμα της βουλής':
            dict = pickle.load(open('group/party_all_time_dict.pkl', "rb"))
            print(f'party all {dict}')
            results = manager.return_line_from_offset('group/party_all_time.txt', dict[selected_field2])
        else:
            dict = []

    else:
        if selected_field1 == 'Μέλος της βουλής':
            dict = pickle.load(open('group/member_per_year_dict.pkl', "rb"))
            results = manager.return_line_from_offset('group/member_per_year.txt', dict[selected_field2, int(selected_field3)])
        elif selected_field1 == 'Κόμμα της βουλής':
            dict = pickle.load(open('group/party_per_year_dict.pkl', "rb"))
            results = manager.return_line_from_offset('group/party_per_year.txt',
                                                dict[selected_field2, int(selected_field3)])
        else:
            dict = []



        '''results = []
        for result in all_results:
            if int(result[1]) == int(selected_field3):
                print(f'Found {result[1]}')
                results = result[0]
                break'''
    results = results.split(',')
    keywords = []
    #values = []
    for i in range(0, len(results), 2):
        keywords.append([results[i], round(float(results[i+1]),2)])
    print(keywords)
    return json.dumps(keywords)


@app.route('/get_top_k_values', methods=['POST'])
def get_top_k_values():
    selected_field = request.form.get('selected_field')
    print('enabled')
    top_k_pairs = pickle.load(open('similarity/top_k_pairs.pkl', "rb"))
    top_k_pairs = top_k_pairs[:int(selected_field)]

    print(top_k_pairs)

    return json.dumps(top_k_pairs)

@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.form.get('query')
    if not query:
        return render_template('base.html')

    lsa_enabled = request.form.get('lsa_enabled')
    print(f'lsa enabled {lsa_enabled}')
    return redirect(url_for('search_results', lsa=lsa_enabled, query=query, page='1'))

'''@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.form.get('query')
    if not query:
        return render_template('base.html')

    return redirect(url_for('search_results', query=query, page='1'))'''


#def top_k_keywords():
#@app.route('/search_results_lsa', methods=['GET', 'POST'])
'''def search_results_lsa():
    query = request.args.get('query')
    page = int(request.args.get('page'))
    lsa_value = request.args.get('lsa')
    print(f'Query {query}')

    # Make a request to the search engine API with the query
    speeches, values = manager.search_lsa(query)
    if speeches:
        total_length = len(speeches)
        print(f'Total length {total_length}')
        speeches = np.array(speeches[(page-1)*10:page*10])
        values = values[(page - 1) * 10:page * 10]

        names = speeches[:, 1]
        dates = speeches[:, 2]
        speeches = speeches[:, 0]
        num_pages = total_length // 10
        if total_length % 10:
            num_pages += 1

        return render_template('search_results.html', lsa=lsa_value, query=query, results=speeches, values=values, names=names,
                               dates=dates, page=str(page), num_pages=str(num_pages))
    else:
        return render_template('search_results.html', lsa=lsa_value, query=query, results=['Δεν βρέθηκαν ομιλίες'], values=[], names=[]
                               , dates=[], page='0', num_pages='0')'''

@app.route('/search_results', methods=['GET', 'POST'])
def search_results():
    query = request.args.get('query')
    page = int(request.args.get('page'))
    lsa_value = request.args.get('lsa')
    print(f'Query {query} ')

    # Make a request to the search engine API with the query
    if lsa_value == 'off':
        print(f'Normal query')
        speeches, values = manager.query_search(query)
    else:
        print(f'LSA query')
        speeches, values = manager.search_lsa(query)
    if speeches:

        total_length = len(values)
        print(f'Total length {total_length}')
        speeches = np.array(speeches[(page-1)*10:page*10])
        values = values[(page - 1) * 10:page * 10]

        names = speeches[:, 1]
        dates = speeches[:, 2]
        speeches = speeches[:, 0]
        num_pages = total_length // 10
        if total_length % 10:
            num_pages += 1

        return render_template('search_results.html', lsa=lsa_value, query=query, results=speeches, values=values, names=names,
                               dates=dates, page=str(page), num_pages=str(num_pages))
    else:
        return render_template('search_results.html', lsa=lsa_value, query=query, results=['Δεν βρέθηκαν ομιλίες'], values=[], names=[]
                               , dates=[], page='0', num_pages='0')



if __name__ == '__main__':
    app.run(debug=True)

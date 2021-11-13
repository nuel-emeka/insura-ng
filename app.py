from flask import Flask, render_template, request, Markup
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import requests
from io import StringIO



"""def get_data(filename):
    token = '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' 
    owner = 'nuel-emeka'
    repo = 'RECOMMENDER'
    path = 'Data/{}'.format(filename)

    # send a request
    r = requests.get(
        'https://api.github.com/repos/{owner}/{repo}/contents/{path}'.format(
        owner=owner, repo=repo, path=path),
        headers={
            'accept': 'application/vnd.github.v3.raw',
            'authorization': 'token {}'.format(token)
                }
        )

    # convert string to StringIO object
    string_io_obj = StringIO(r.text)

    # Load data to df
    df = pd.read_csv(string_io_obj, sep=",", index_col=0)
    df = df.reset_index()
    return df """

def get_data_public(filename):
    filename = filename.replace(' ', '%20')
    owner = 'nuel-emeka'
    repo = 'RECOMMENDER'
    path = 'Data/{}'.format(filename)
    # get link
    r = 'https://raw.githubusercontent.com/{owner}/{repo}/main/{path}'.format(owner=owner, repo=repo, path=path)
    # Load data to df
    df = pd.read_csv(r)
    return df

def get_ready(data):
    for col in data.iloc[:, 4:].columns:
        data[col] = data[col].apply(yesNo_encode)
    data = pd.get_dummies(data, columns=['geographical coverage'], prefix=['location'])
    data['Premium Tier'] = OrdinalEncoder().fit_transform(data[['Premium Tier']])
    data = data.iloc[:, 3:]
    return data

def yesNo_encode(value):
    if value.lower().strip() == 'yes':
        return 1
    elif value.lower().strip() == 'no':
        return 0
    else:
        return value

#getting all data ready...
data = get_data_public('Clean_data.csv')
data_copy = data.copy()
rating_data = get_data_public("HMO ratings - Form responses 1.csv")
df = get_ready(data)

def cosine_sim(response, data):
    """ largest value signifies great similarity"""
    df = data
    test = [response]
    index_ = []
    results = []
    for index in df.index:
        value = cs([df.loc[index,:].values], test)[0][0]
        index_.append(index)
        results.append(value)
    
    df_ = pd.DataFrame({'HMO index': index_ ,'cosine similarity': results})
    df_ = df_.sort_values('cosine similarity', ascending=False).set_index('HMO index')

    return df_.index[:5].values

def clean_ratings(ratings_data, hmo):
    ratings = ratings_data
    hmo = [i.upper() for i in hmo]
    
    ratings.rename(columns={'What is the name of your HMO?': 'Name'}, inplace=True)
    ratings.dropna(subset=['Name'], inplace=True)
    ratings['Name'] = ratings['Name'].apply(lambda x: x.upper().strip())
    ratings['sum ratings'] = ratings.iloc[:, [3,4,5,6]].sum(axis=1)
    ratings = ratings.groupby('Name').mean()[['sum ratings']]
    
    hmo = [hmo_ for hmo_ in hmo if hmo_ in ratings.index]
    hmo_ratings = ratings.loc[hmo, ['sum ratings']].sort_values(by='sum ratings', ascending=False)

    return hmo_ratings.index[:3].to_list()

def top_5_dict(result):
    top_5 = result
    
    hmo_names = [hmo.upper().strip() for hmo in data.loc[top_5, 'Name']]
    hmo_dict = {hmo: [] for hmo in set(hmo_names)}
    for hmo, index in zip(hmo_names, top_5):
        hmo_dict.get(hmo).append(index)
    
    return hmo_dict, hmo_names

def top_3_index(top5):
    top5_dict = top_5_dict(top5)
    rating = clean_ratings(rating_data, top5_dict[0].keys())
    for hmo in top5_dict[1]:
        if hmo not in rating:
            rating.append(hmo)
        else:
            pass

    index = []
    for name in rating[:3]:
        index.extend(top5_dict[0].get(name))
    
    return index[:3]

def print_top_3(result):
    top_3 = result
    df = data_copy.loc[top_3, :]
    df.index = [1,2,3]
    html = df.to_html()
    return Markup(html)
    

def recommend(test, df):
    tier = test[0]
    loc = test[-2]
    
    if loc == 0:
        rows_drop = df[(df['location_Lagos']==1)].index
        df = df.drop(rows_drop, axis=0)
    else:
        pass
    
    if tier == 0:
        rows_drop = df[df['Premium Tier']>0].index
        df = df.drop(rows_drop, axis=0)
        results = cosine_sim(test, df)
    elif tier == 1:
        rows_drop = df[df['Premium Tier']>1].index
        df = df.drop(rows_drop, axis=0)
        results = cosine_sim(test, df)
    elif tier == 2:
        rows_drop = df[df['Premium Tier']>2].index
        df = df.drop(rows_drop, axis=0)
        results = cosine_sim(test, df)
    else:
        results = cosine_sim(test, df)
        
    index = top_3_index(results)
    
    return print_top_3(index)

def user_ready(data):
    test = []
    for dt in data:
        if dt.strip() in ['TIER 1', 'NO', 'NATIONWIDE']:
            test.append(0)
        elif dt.strip() in ['TIER 2', 'YES', 'LAGOS']:
            test.append(1)
        elif dt.strip()=='TIER 3':
            test.append(2)
        elif dt.strip()=='TIER 4':
            test.append(3)
    if test[-1]==1:
        test.append(0)
    else:
        test.append(1)

    return test


#print_top_3(top_3_index(top_5))
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    color = {
        'blue': '002aff',
        'yellow': 'e1eb34',
        'green': '28fc03',
        'red': 'fc1703', 
        'purple': 'b503fc'}
    user = [value for value in request.form.values()]
    if 'SELECT AN OPTION' in user:
        return render_template('index.html', prediction_text=Markup('KINDLY ANSWER ALL QUESTIONS<br>THANK'))
    user_data = user_ready(user)
    if len(user_data)<12 :
        return render_template('index.html', prediction_text=Markup('KINDLY ANSWER ALL QUESTIONS<br>THANK'))
    else:
        return render_template('index.html', prediction_text=user_data)

if __name__=='__main__':
    app.run(debug=True)


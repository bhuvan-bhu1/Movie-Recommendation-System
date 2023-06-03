import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib
from sklearn.metrics.pairwise import cosine_similarity
import requests

df9 = pd.read_csv('Final.csv')
def getindex(given_id):
    for i in range(len(df9)):
        if df9['id'][i] == given_id:
            return i


movies_list = df9['title_x'].to_list()
vector = TfidfVectorizer()
combined_data = df9['genres'] + ' ' + df9['keywords'] + ' ' + df9['tagline'] + ' ' + df9['cast'] + ' ' + df9['directors']
features = vector.fit_transform(combined_data.astype('U'))
similarity = cosine_similarity(features)




def get_movie_id(given):

    url = "https://online-movie-database.p.rapidapi.com/auto-complete"

    querystring = {"q":given}

    headers = {
        "X-RapidAPI-Key": "dd56ce9576msh6080c645b21880ap13976djsn974779118eef",
        "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    movie_id = response.json()['d'][0]['id']
    image_url = response.json()['d'][0]['i']['imageUrl']
    # print(movie_id)
    # print(data)
    return([given,movie_id,image_url])



def find_movie(movie_name):
    try:


        movie_name_crrt = difflib.get_close_matches(movie_name,movies_list)[0]
        movie_id = df9[df9['title_x'] == movie_name_crrt].values[0][3]
        index_of_the_movie = getindex(movie_id)
        score_of_similarity = list(enumerate(similarity[index_of_the_movie]))
        similar_movies_list = sorted(score_of_similarity,key=lambda x:x[1],reverse=True)[:12]
        final_output = []
        total = []
        for i in similar_movies_list:
            final_output.append(df9.iloc[i[0]]['title_x'])
        for i in final_output:
            total.append(get_movie_id(i))
            # print(i)
        return total
    except:
        return [['None','None']]

if __name__ == '__main__':
    print('Welcome to the Module Page')
    # print(find_movie('Avengers'))
    print(get_movie_id('Thor'))
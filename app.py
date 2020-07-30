import pandas as pd
import numpy as np
import sklearn.metrics.pairwise as pw
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances
from flask import Flask, render_template, request


def itembased_movie_recommender(movie_name):
    final_movies=pd.read_csv(r"movies_final.csv")
    movie_name=movie_name.lower()
    if movie_name not in final_movies['title'].unique():
        return "This movie is not in Database. Please enter an other movie"
    else:
        
        pivot_ibr = pd.pivot_table(final_movies,
                                          index='title',
                                          columns=['userId'], values='rating')  
        sparse_ibr = sparse.csr_matrix(pivot_ibr.fillna(0))
        recommender_ibr = pw.cosine_similarity(sparse_ibr)
        recommender_ibr_df=pd.DataFrame(recommender_ibr, 
                                      columns=pivot_ibr.index,
                                      index=pivot_ibr.index)
        cosine_ibr_df = pd.DataFrame(recommender_ibr_df[movie_name].sort_values(ascending=False))
        cosine_ibr_df.reset_index(level=0, inplace=True)
        cosine_ibr_df.columns = ['title','cosine_sim']
        movie_list=list((cosine_ibr_df['title'].values))
        return movie_list[1:6]
        
app = Flask(__name__)
@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    movie_list=itembased_movie_recommender(movie)
    movie=movie.upper()
    if type(movie)==type('string'):
        return render_template('recommend.html',movie=movie,movie_list=movie_list,t='s')
    else:
        return render_template('recommend.html',movie=movie,movie_list=movie_list,t='l')

if __name__=='__main__':
    app.run()
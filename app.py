from flask import Flask, render_template, request
from recommender import recommend

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    movie_name = request.form['movie']

    recommendations = recommend(movie_name)

    return render_template(
        'index.html',
        recommendations=recommendations,
        movie_name=movie_name
    )

if __name__ == "__main__":
    app.run(debug=True)
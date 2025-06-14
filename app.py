from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# CSV 파일 로드
data = pd.read_csv('steam_games.csv', encoding='cp949')
data = data[~data['Tags'].isna()]

# 중복을 제거한 태그 리스트
tag_options = list(set(tag.strip() for tags in data['Tags'] for tag in tags.split(',')))

# 사용자 입력 태그 초기화
user_input_tags = []

# 사용자가 선택한 값을 저장할 변수
selected_popularity = 'notFamous'
selected_num_games = 5   # 기본값: 5

def preprocess_data(data):
    data['Supported languages'] = data['Supported languages'].apply(lambda x: x.strip("[]").replace("'", ""))
    
    # 'Tags' 열 전처리
    tags_corpus = [" ".join(str(tags).split(',')) for tags in data['Tags']]

    # CountVectorizer를 사용하여 벡터화
    vectorizer = CountVectorizer()
    tags_matrix = vectorizer.fit_transform(tags_corpus).toarray()

    return vectorizer, tags_matrix

def calculate_cosine_similarity(user_input_vector, game_matrix):
    # 코사인 유사도 계산
    similarities = cosine_similarity(user_input_vector, game_matrix)

    return similarities

def recommend_games(user_input_tags, data, vectorizer, min_total_reviews=500, max_total_reviews=10000, threshold_percentage=50, max_recommendations=5):
    # 최소 리뷰 수 이상인 게임만 선택
    data = data[(data['Total'] >= min_total_reviews) & (data['Total'] <= max_total_reviews)]

    # 사용자 입력 태그에 대한 벡터 생성
    user_input_vector = vectorizer.transform([" ".join(user_input_tags)]).toarray()

    # 게임 태그에 대한 벡터 생성
    game_matrix = vectorizer.transform(data['Tags']).toarray()

    # 코사인 유사도 계산
    similarities = calculate_cosine_similarity(user_input_vector, game_matrix)

    # 코사인 유사도 기준으로 내림차순 정렬
    similar_games_indices = similarities.argsort()[0][::-1]

    # 상위 threshold_percentage 이상의 유사도를 가진 데이터만 선택
    threshold_index = int(len(similar_games_indices) * (threshold_percentage / 100))
    similar_games_selected = data.iloc[similar_games_indices[:threshold_index]]
    similar_games_selected = similar_games_selected.sort_values(by=['Score', 'Total'], ascending=[False, False])

    # 사용자 입력 태그의 개수부터 1개까지의 경우에 대해 검색하고 출력
    num_recommendations = 0
    num_tags = len(user_input_tags)
    recommended_games = []
    while num_tags > 0 and num_recommendations < max_recommendations:
        # 선택된 게임을 출력
        for idx, game in similar_games_selected.iterrows():
            # 게임의 태그와 사용자 입력 태그를 비교하여 공통된 태그의 수를 계산
            common_tags = sum(tag in game['Tags'] for tag in user_input_tags)
            # 만약 공통된 태그의 수가 현재 검사 중인 태그의 수와 같다면 출력
            if common_tags == num_tags:
                release_date = datetime.strptime(game['Release date'], '%d-%b-%y').strftime('%Y, %b, %d')
                supported_languages = ', '.join([lang.strip('\'') for lang in game['Supported languages'].strip('[]').split(',')])
                recommended_games.append({
                    'AppID': game['AppID'],
                    'Name': game['Name'],
                    'Tags': game['Tags'],
                    'Header image': game['Header image'],
                    'About the game': game['About the game'],
                    'Release date': release_date,
                    'Price': game['Price'],
                    'Supported languages': supported_languages,
                    'Positive': game['Positive'],
                    'Negative': game['Negative'],
                    'Total': game['Total'],
                    'Score': game['Score']
                })
                # 해당 게임 삭제
                data = data[data.index != idx]
                # num_recommendations 증가
                num_recommendations += 1

                # num_recommendations가 원하는 값에 도달하면 반복문 종료
                if num_recommendations >= max_recommendations:
                    break

        # n개의 태그를 가진 게임으로 max_recommendations을 채우지 못했다면 n을 1 감소하여 재검색
        num_tags -= 1

    return recommended_games

# 사용자가 선택할 수 있는 태그 옵션
@app.route('/')
def index():
    return render_template('index.html', tag_options=tag_options, user_input_tags=user_input_tags)

# 선택한 태그를 받아오는 부분
@app.route('/select_tag', methods=['POST'])
def select_tag():
    selected_tag = request.form.get('tag')
    
    # 검색 결과에서 선택한 태그가 아니라면 추가
    if selected_tag and selected_tag not in user_input_tags:
        user_input_tags.append(selected_tag)

    return render_template('index.html', tag_options=tag_options, user_input_tags=user_input_tags)

# 선택한 태그를 제거하는 부분
@app.route('/remove_tag', methods=['POST'])
def remove_tag():
    removed_tag = request.form.get('tag')
    if removed_tag in user_input_tags:
        user_input_tags.remove(removed_tag)

    return render_template('index.html', tag_options=tag_options, user_input_tags=user_input_tags)


# 라우트에 사용자 선택값을 전달하는 엔드포인트 추가
@app.route('/update_popularity', methods=['POST'])
def update_popularity():
    global selected_popularity, selected_num_games
    selected_popularity = request.json['popularity'] if request.json['popularity'] else 'notFamous'
    selected_num_games = int(request.json['numGames'] if request.json['numGames'] else 5)
    
    return jsonify({'message': 'success'})

# 최종적으로 선택된 인기도에 따라서 minTotalReviews와 maxTotalReviews 값을 계산하여 반환
def calculate_review_range():
    if selected_popularity == 'notFamous':
        return 500, 1000
    elif selected_popularity == 'aLittleNotFamous':
        return 1000, 5000
    elif selected_popularity == 'ordinary':
        return 5000, 10000
    elif selected_popularity == 'aLittleFamous':
        return 10000, 30000
    elif selected_popularity == 'veryFamous':
        return 30000, 700000
    else:
        return 0, 1000000
    
# 추천 결과를 보여주는 부분
@app.route('/recommend', methods=['POST'])
def recommend():
    vectorizer, _ = preprocess_data(data)

    # 사용자가 입력한 게임 수 가져오기
    min_total_reviews, max_total_reviews = calculate_review_range()
    recommended_games_data = recommend_games(user_input_tags, data, vectorizer, min_total_reviews, max_total_reviews, max_recommendations=selected_num_games)

    return render_template('result.html', recommended_games=recommended_games_data[:selected_num_games])

# 태그 검색 기능 추가
@app.route('/search_tags', methods=['POST'])
def search_tags():
    input_text = request.form.get('input', '').lower()
    filtered_tags = [tag for tag in tag_options if input_text in tag.lower()]

    return render_template('tag_list.html', tags=filtered_tags)


if __name__ == '__main__':
    app.run(debug=True)

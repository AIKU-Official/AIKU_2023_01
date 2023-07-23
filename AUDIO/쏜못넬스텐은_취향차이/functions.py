import csv
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

songs=[]


def set_API(client_id,client_secret):
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)


def song_extract(uri):
    global feature
    d = sp.audio_features(uri)[0]
    new = [d['acousticness'], d['danceability'], d['energy'], d['instrumentalness'], d['liveness'], d['valence']]
    feature.loc[len(feature)] = new


def get_playlist(playlist_link):
    global feature
    global songs
    playlist_URI = playlist_link.split("/")[-1].split("?")[0]
    for track in sp.playlist_tracks(playlist_URI)["items"]:
        songs.append(track["track"]["name"])
    track_uris = [x["track"]["uri"] for x in sp.playlist_tracks(playlist_URI)["items"]]
    for track in track_uris:
        song_extract(track)


def map_features():
    global feature
    f_e = open(r'C:\Users\foldd\PycharmProjects\AIKU_Project\echonest.csv', "r", encoding="UTF-8")
    reader_e = csv.reader(f_e)
    reader_e = list(reader_e)[4:]

    df = []
    for i in range(len(reader_e)):
        df.append(reader_e[i][1:9])

    df = pd.DataFrame(df,
                      columns=['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness',
                               'tempo', 'valence'])
    feature = df[['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']]
    label_to_color = {0: 'red', 1: 'orange', 2: 'yellow', 3: 'green', 4: 'blue', 5: 'navy', 6: 'purple', 7: 'black',
                      8: 'pink', 9: 'lime', 10: 'cyan', 11: 'peru', 12: 'blueviolet', 13: 'silver', 14: 'plum'}

    playlists = ["https://open.spotify.com/playlist/6ofwdZgbppeVWCxTYrAntU?si=Z-LCQYOPQZq0MSX5o1cQ8g",
                 "https://open.spotify.com/playlist/2Ri1vUcJoQOezyKdzCOlaA?si=b_9T2fuASlW0sqcP7UUWXQ",
                 "https://open.spotify.com/playlist/37i9dQZF1DZ06evO3vxzPU?si=29jKOp1FSbCw10Ot1CXEbQ",
                 "https://open.spotify.com/playlist/5ixmk7Y0npdkcKDYdJnzUk?si=93QttA8XS_m1b0wAESP2cQ"]

    for playlist in playlists:
        get_playlist(playlist)


def fit_model(uri):
    global feature
    d = sp.audio_features(uri)[0]
    new = [d['acousticness'], d['danceability'], d['energy'], d['instrumentalness'], d['liveness'], d['valence']]
    feature.loc[len(feature)] = new

    k_model = KMeans(n_clusters=6)
    k_model.fit(feature)
    result = k_model.fit_predict(feature)

    model = TSNE(learning_rate=1000)
    transformed = model.fit_transform(feature)
    xs = transformed[:, 0]
    ys = transformed[:, 1]

    thorn = [(songs[i - 13129], xs[i], ys[i], result[i]) for i in range(13129, 13165)]
    sten = [(songs[i - 13129], xs[i], ys[i], result[i]) for i in range(13165, 13216)]
    nell = [(songs[i - 13129], xs[i], ys[i], result[i]) for i in range(13216, 13266)]
    mot = [(songs[i - 13129], xs[i], ys[i], result[i]) for i in range(13266, 13290)]

    artists = [thorn, mot, nell, sten]
    names = ["쏜애플", "못", "넬", "국카스텐"]
    results=[result[13129:13290].tolist().count(i) for i in range(6)]
    key_x, key_y, key_c = xs[-1], ys[-1], result[-1]
    print(results)
    if results[key_c]<=20:
        recommendation="노래가 취향에 맞지 않으실 수 있습니다. 주의해주세요.\n그럼에도 듣고 싶으시다면, "
    else:
        recommendation="노래가 취향에 맞으실 것 같아요. "
    similar_songs = []
    for artist in artists:
        artist_sim = 999999999999
        sim_s = False
        for song in artist:
            name, x, y, cluster = song
            if abs(x - key_x) + abs(y - key_y) < artist_sim:
                artist_sim = abs(x - key_x) + abs(y - key_y)
                sim_s = name
        similar_songs.append([artist_sim, sim_s])
    for i in range(4):
        similar_songs[i].append(names[i])
    similar_songs.sort()
    for i in range(4):
        similar_songs[i].pop(0)
    a=similar_songs[0][1]
    s=similar_songs[0][0]

    recommendation+="제일 비슷한 노래는 "+str(a)+"의 "+str(s)+"입니다.\n다른 가수들의 비슷한 노래들로는\n"
    for i in range(1,3):
        recommendation+=str(similar_songs[i][1])+"의 "+str(similar_songs[i][0])+",\n"
    recommendation+=str(similar_songs[3][1])+"의 "+str(similar_songs[3][0])+"가 있습니다."

    return recommendation
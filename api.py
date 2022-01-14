from flask import Flask
from flask import jsonify
from flask_cors import CORS
import pickle
import datetime
import numpy as np

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return "Hello, World!"


@app.route('/GBRT')
def yes_GBRT():
    return "yes GBRT"


@app.route('/dt')
def yes_dt():
    return "yes dt"


@app.route('/time/<string:publishAt>')
def compute_time(publishAt):
    get_time = datetime.datetime.strptime(publishAt, "%Y-%m-%dT%H:%M:%S.000Z")
    result = []
    result.append(get_time.year)
    result.append(get_time.month)
    result.append(get_time.day)
    return jsonify(result)


@app.route('/GBRT/<int:duration>/<int:viewcount>/<int:subscribercount>/<int:videocount>/<string:time>')
def pred_GBRT(duration, viewcount, subscribercount, videocount, time):
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d %H.%M.%S")
    vid1t = datetime.datetime.strptime(time, "%Y-%m-%d %H.%M.%S")
    vid2t = datetime.datetime.strptime(now, "%Y-%m-%d %H.%M.%S")
    timepass = (vid2t-vid1t).total_seconds()

    avgviewcount = viewcount/videocount
    data = []
    data.append(duration)
    data.append(viewcount)
    data.append(subscribercount)
    data.append(videocount)
    data.append(avgviewcount)
    data.append(timepass)

    with open('model/GBRT.pickle', 'rb')as f:
        gbrt = pickle.load(f)
        result = gbrt.predict(
            [[duration, viewcount, subscribercount, videocount, avgviewcount, timepass]])
        r = []
        r.append(result[0])
    return jsonify(r[0])


@app.route('/dt/<int:duration>/<int:viewcount>/<int:subscribercount>/<int:videocount>/<string:time>')
def pred_dt(duration, viewcount, subscribercount, videocount, time):
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d %H.%M.%S")
    vid1t = datetime.datetime.strptime(time, "%Y-%m-%d %H.%M.%S")
    vid2t = datetime.datetime.strptime(now, "%Y-%m-%d %H.%M.%S")
    timepass = (vid2t-vid1t).total_seconds()
    avgviewcount = viewcount/videocount
    data = []
    data.append(duration)
    data.append(viewcount)
    data.append(subscribercount)
    data.append(videocount)
    data.append(avgviewcount)
    data.append(timepass)

    with open('model/dt.pickle', 'rb')as f:
        dtm = pickle.load(f)
        result = dtm.predict(
            [[duration, viewcount, subscribercount, videocount, avgviewcount, timepass]])
        r = []
        r.append(result[0])
    return jsonify(r[0])


app.run(port=5000, debug=False)

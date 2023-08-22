from flask import Flask, render_template, request
import logging
import pickle
import numpy as np

logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(asctime)s  %(levelname)s  %(message)s")

model = pickle.load(open("Ridge.pkl", "rb"))

app = Flask(__name__, template_folder="templates")

class Flaskapp:
    """
    This is a flask app package
    """
    def __init__(self, app):
        self.app = app

    def configure_home_routes(self):
        @self.app.route("/")
        def home():
            logging.info("You are now on the home page")
            return render_template("home.html")
        
        @self.app.route("/predict", methods=["POST"])
        def predict():
            data1 = float(request.form["a"])
            data2 = float(request.form["b"])
            data3 = float(request.form["c"])
            data4 = float(request.form["d"])
            data5 = float(request.form["e"])
            data6 = float(request.form["f"])
            data7 = float(request.form["g"])
            data8 = float(request.form["h"])
            data9 = float(request.form["i"])
            data10 = float(request.form["j"])

            arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10]])
            pred = model.predict(arr)
            return render_template("result.html", data=np.round(pred, 2))

flask_app = Flaskapp(app)
flask_app.configure_home_routes()  

if __name__ == "__main__":
    app.run(debug=True)
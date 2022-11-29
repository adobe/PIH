from flask import Flask
import flask


BASE_URL = "/flask"  # Make sure your port on the devbox has this name
# E.g., --port flask:5000
app = Flask(
    __name__,
    static_url_path=f"{BASE_URL}/static",
)


@app.route(BASE_URL + "/")
@app.route(BASE_URL + "/index.html", strict_slashes=False)
# def hello_world():
#     return "<p>Hello, World!</p>"
def index():
    """Displays the index page accessible at '/'"""
    return flask.render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=True)

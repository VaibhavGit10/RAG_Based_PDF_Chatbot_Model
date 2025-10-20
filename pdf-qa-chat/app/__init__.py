from flask import Flask
from .config import Config

def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.from_object(Config)

    # Register routes
    from .routes import bp as main_bp
    app.register_blueprint(main_bp)

    return app


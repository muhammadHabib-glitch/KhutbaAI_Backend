# app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)

    # ← your existing config.py values
    from config import DATABASE_URI
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # initialize ORM
    db.init_app(app)

    # register your routes blueprint
    from app.routes import routes
    app.register_blueprint(routes)

    return app

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager
import os
import re

db = SQLAlchemy()
DB_NAME = "heart_new.db"

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'justRandomStringForEncryption'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'

    # config with postgresql
    # uri = os.getenv('DATABASE_URL')
    # if uri.startswith("postgres://"):
    #     uri = uri.replace("postgres://", "postgresql://",1)
    db.init_app(app)

    from .views import views
    from .auth import auth
    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    from .models import User, Note

    create_database(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    return app

def create_database(app):
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')
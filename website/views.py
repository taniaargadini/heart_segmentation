from flask import Blueprint, render_template
from flask_login import login_required, current_user

views = Blueprint('views', __name__)

@views.route('/')
# @login_required
def home():
    return render_template("index.html", user=current_user) # user=current_user -> check if the user is authenticated

@views.route('/main')
@login_required
def main():
    return render_template('home.html', user=current_user)
from flask import render_template
from . import views

@views.route('/')
def home():
    return render_template('home.html')
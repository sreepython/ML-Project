from flask import Flask
from .views import views as views_blueprint

def create_app():
    app = Flask(__name__)
    
    # register blueprints
    app.register_blueprint(views_blueprint)
  
    return app
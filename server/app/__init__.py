from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os

load_dotenv()

def create_app():
    return Flask(__name__)
   
app = create_app()

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///estilista_virtual.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.getenv("FLASK_SECRET")

db = SQLAlchemy(app)

from app.models.models import User  # Importa os modelos para criar as tabelas

with app.app_context():
    db.create_all()

from app.controllers.routes import init_routes
init_routes(app)

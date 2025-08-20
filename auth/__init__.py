# Blueprint de autenticação
from flask import Blueprint

auth_bp = Blueprint('auth', __name__)

from . import routes
from .routes import google_bp

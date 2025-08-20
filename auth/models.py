from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'users.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def create_user_table():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password TEXT,
        name TEXT,
        google_id TEXT
    )''')
    conn.commit()
    conn.close()

create_user_table()

def add_user(email, password=None, name=None, google_id=None):
    conn = get_db()
    c = conn.cursor()
    c.execute('INSERT INTO users (email, password, name, google_id) VALUES (?, ?, ?, ?)',
              (email, generate_password_hash(password) if password else None, name, google_id))
    conn.commit()
    conn.close()

from pymongo import MongoClient
from datetime import datetime

# Modelo de usuário para MongoDB
def criar_usuario(email, senha, nome):
    mongo_client = MongoClient()
    mongo_db = mongo_client['estilista_virtual']
    mongo_usuarios = mongo_db['usuarios']
    if mongo_usuarios.find_one({'email': email}):
        return False  # Usuário já existe
    mongo_usuarios.insert_one({
        'email': email,
        'senha': senha,
        'nome': nome,
        'criado_em': datetime.now().strftime('%d/%m/%Y %H:%M')
    })
    return True

def autenticar_usuario(email, senha):
    mongo_client = MongoClient()
    mongo_db = mongo_client['estilista_virtual']
    mongo_usuarios = mongo_db['usuarios']
    usuario = mongo_usuarios.find_one({'email': email, 'senha': senha})
    return usuario
def get_user_by_email(email):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = c.fetchone()
    conn.close()
    return user

def get_user_by_google_id(google_id):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE google_id = ?', (google_id,))
    user = c.fetchone()
    conn.close()
    return user

def check_password(user, password):
    return check_password_hash(user['password'], password)

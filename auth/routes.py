from flask import render_template, request, redirect, url_for, flash, session, Blueprint
from .models import add_user, get_user_by_email, check_password, get_user_by_google_id
from flask import current_app as app
from flask_dance.contrib.google import make_google_blueprint, google
import os


from . import auth_bp

google_bp = make_google_blueprint(
    client_id=os.getenv('GOOGLE_CLIENT_ID', 'GOOGLE_CLIENT_ID_AQUI'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET', 'GOOGLE_CLIENT_SECRET_AQUI'),
    scope=["profile", "email"],
    redirect_url='/auth/google/callback'
)
print('Flask-Dance Google redirect_url configurado:', '/auth/google/callback')
print('GOOGLE_CLIENT_ID:', os.getenv('GOOGLE_CLIENT_ID'))
print('GOOGLE_CLIENT_SECRET:', os.getenv('GOOGLE_CLIENT_SECRET'))
from .models import criar_usuario, autenticar_usuario

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        senha = request.form['senha']
        if not email or not senha:
            flash('Preencha todos os campos.')
            return render_template('login.html')
        usuario = autenticar_usuario(email, senha)
        if usuario:
            session['user_id'] = str(usuario['_id'])
            session['email'] = usuario['email']
            session['nome'] = usuario.get('nome', '')
            return redirect(url_for('index'))
        else:
            flash('Usuário ou senha incorretos.')
            return render_template('login.html')
    return render_template('login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        senha = request.form['senha']
        confirm_senha = request.form.get('confirm_senha')
        nome = request.form['nome']
        if not email or not senha or not confirm_senha or not nome:
            flash('Preencha todos os campos.')
            return render_template('register.html')
        if senha != confirm_senha:
            flash('As senhas não coincidem.')
            return render_template('register.html')
        sucesso = criar_usuario(email, senha, nome)
        if sucesso:
            usuario = autenticar_usuario(email, senha)
            session['user_id'] = str(usuario['_id'])
            session['email'] = usuario['email']
            session['nome'] = usuario.get('nome', '')
            return redirect(url_for('index'))
        else:
            flash('E-mail já cadastrado.')
            return render_template('register.html')
    return render_template('register.html')

@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('auth.login'))

@auth_bp.route('/login/google')
def login_google():
    return redirect(url_for('google.login'))

@auth_bp.route('/google/callback')
def google_callback():
    if not google.authorized:
        return redirect(url_for('auth.login'))
    resp = google.get('/oauth2/v2/userinfo')
    if not resp.ok:
        flash('Erro ao autenticar com Google.')
        return redirect(url_for('auth.login'))
    info = resp.json()
    user = get_user_by_google_id(info['id'])
    if not user:
        add_user(info['email'], None, info.get('name'), info['id'])
        user = get_user_by_email(info['email'])
    session['user_id'] = user['id']
    session['user_email'] = user['email']
    return redirect(url_for('index'))

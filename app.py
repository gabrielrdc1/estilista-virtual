import json
import os
from pymongo import MongoClient
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from datetime import datetime
from dotenv import load_dotenv
from auth import auth_bp
from auth.routes import google_bp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from modelo.yolo_utils import segmentar_pecas
from modelo.avaliador_look import carregar_combinacoes, avaliar_combinacao
from sklearn.cluster import KMeans
import cv2


MODEL_PATH = 'modelo/cnn_model_mobilenetv2.h5'
CLASS_NAMES = ['blusa_feminino', 'blusa_masculino', 'calca_feminino', 'calca_masculino', 'tenis_feminino', 'tenis_masculino']
model_cnn = load_model(MODEL_PATH)
# Carregar variáveis de ambiente
load_dotenv()

# Conexão MongoDB
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client['estilista_virtual']
mongo_historico = mongo_db['historico']
mongo_usuarios = mongo_db['usuarios']
mongo_perfis = mongo_db['perfis']
mongo_imagens = mongo_db['imagens']

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'segredo-tcc')
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(google_bp, url_prefix='/auth')
UPLOAD_FOLDER = 'static/uploads'
## Removido variável antiga de histórico em arquivo
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_user_profile_path():
    user_id = session.get('user_id')
    if not user_id:
        return None
    return user_id

# --- Endpoints de perfil do usuário ---
@app.route('/perfil/preferencias', methods=['GET', 'POST'])
def perfil_preferencias():
    if 'user_id' not in session:
        return jsonify({'error': 'Não autenticado'}), 401
    user_id = get_user_profile_path()
    if request.method == 'GET':
        perfil = mongo_perfis.find_one({'user_id': user_id}, {'_id': 0})
        if perfil:
            return jsonify(perfil)
        else:
            return jsonify({'estilo': '', 'cores': '', 'ocasioes': ''})
    elif request.method == 'POST':
        data = request.get_json()
        perfil = {
            'user_id': user_id,
            'estilo': data.get('estilo', ''),
            'cores': data.get('cores', ''),
            'ocasioes': data.get('ocasioes', '')
        }
        mongo_perfis.replace_one({'user_id': user_id}, perfil, upsert=True)
        return jsonify({'success': True})

def get_user_upload_folder():
    user_id = session.get('user_id')
    if not user_id:
        return UPLOAD_FOLDER
    user_folder = os.path.join(UPLOAD_FOLDER, f'user_{user_id}')
    os.makedirs(user_folder, exist_ok=True)
    return user_folder

def classificar(image_path):
    # Recebe sexo via variável global ou parâmetro
    from flask import request
    sexo = request.form.get('sexo', '').lower()
    print(f"Classificando imagem com CNN: {image_path} | Sexo: {sexo}")
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model_cnn.predict(x)[0]
    # Filtrar classes pelo sexo
    if sexo == 'masculino':
        idxs_validos = [CLASS_NAMES.index(c) for c in ['blusa_masculino', 'calca_masculino', 'tenis_masculino']]
    elif sexo == 'feminino':
        idxs_validos = [CLASS_NAMES.index(c) for c in ['blusa_feminino', 'calca_feminino', 'tenis_feminino']]
    else:
        idxs_validos = list(range(len(CLASS_NAMES)))
    # Zerar probabilidades das classes não válidas
    mask = np.ones_like(preds, dtype=bool)
    mask[idxs_validos] = False
    preds[mask] = 0
    class_idx = np.argmax(preds)
    classe = CLASS_NAMES[class_idx]
    confianca = float(preds[class_idx])
    return classe, confianca



def avaliar_look(pecas_detectadas, combinacoes=None):
    tipos = [p.get('tipo', '').lower() for p in pecas_detectadas]
    tem_blusa = any('blusa' in t for t in tipos)
    tem_calca = any('calca' in t for t in tipos)
    tem_tenis = any('tenis' in t for t in tipos)
    pecas_str = ', '.join([f"{p['tipo'].capitalize()} ({p['cor']})" for p in pecas_detectadas])
    if tem_blusa and tem_calca and tem_tenis and combinacoes is not None:
        # Usa o avaliador inteligente
        avaliacao = avaliar_combinacao(pecas_detectadas, combinacoes)
        valido = avaliacao.get('valido', False)
        mensagem = avaliacao.get('mensagem', '')
        return valido, mensagem
    elif len(pecas_detectadas) == 0:
        return False, "Nenhuma peça foi detectada. Tente enviar uma imagem mais nítida."
    else:
        return False, f"Peças detectadas: {pecas_str}. Para uma avaliação completa do look, envie uma blusa, uma calça e um tênis."


def ler_historico():
    user_id = session.get('user_id')
    if not user_id:
        return list(mongo_historico.find({}, {'_id': 0}))
    return list(mongo_historico.find({'user_id': user_id}, {'_id': 0}))

def salvar_historico(historico):
    user_id = session.get('user_id')
    mongo_historico.delete_many({'user_id': user_id})
    for item in historico:
        item['user_id'] = user_id
        mongo_historico.insert_one(item)

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            # Redireciona para a tela de login centralizada
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated


# Página principal pública: mostra histórico só se logado
@app.route('/', methods=['GET'])
def index():
    user_id = session.get('user_id')
    usuario_logado = bool(user_id)
    historico = ler_historico() if usuario_logado else []
    return render_template('index.html', historico=historico[::-1], resultado=None, usuario_logado=usuario_logado)


def extrair_cor_dominante(img_path, n_clusters=4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(img)
    counts = np.bincount(kmeans.labels_)
    cores = kmeans.cluster_centers_.astype(int)
    idx = np.argmax(counts)
    r, g, b = cores[idx]
    # Mapeamento para nomes de cores comuns
    def cor_nomeada(r, g, b):
        if r > 220 and g > 220 and b > 220:
            return "Branco"
        if r < 40 and g < 40 and b < 40:
            return "Preto"
        if abs(r-g) < 20 and abs(r-b) < 20 and abs(g-b) < 20:
            if r > 180:
                return "Cinza Claro"
            elif r > 100:
                return "Cinza"
            else:
                return "Cinza Escuro"
        if r > 180 and g > 180 and b < 100:
            return "Amarelo"
        if r > 180 and b > 180 and g < 100:
            return "Rosa"
        if g > 180 and b > 180 and r < 100:
            return "Ciano"
        if r > 180 and g < 100 and b < 100:
            return "Vermelho"
        if g > 180 and r < 100 and b < 100:
            return "Verde"
        if b > 180 and r < 100 and g < 100:
            return "Azul"
        if r > 150 and g > 100 and b < 80:
            return "Laranja"
        if r > 100 and g < 80 and b > 100:
            return "Roxo"
        # Marrom: tons médios a escuros, r > g > b, g não muito baixo, b baixo
        if (r > 90 and g > 40 and b < 90 and r > g > b) or (r > 80 and g > 50 and b < 70 and r-g < 60 and g-b > 10):
            return "Marrom"
        return f"RGB({r},{g},{b})"
    return cor_nomeada(r, g, b)

@app.route('/analisar_ajax', methods=['POST'])
def analisar_ajax():
    if 'imagens' not in request.files:
        return jsonify({"success": False, "message": "Nenhum arquivo enviado"}), 400
    
    imagens = request.files.getlist('imagens')
    sexo = request.form.get('sexo', '')
    ocasiao = request.form.get('ocasiao', '')
    if not imagens or not any(img.filename for img in imagens):
        return jsonify({"success": False, "message": "Nenhuma imagem válida foi enviada"}), 400
    
    caminhos = []  # caminhos das imagens originais
    pecas_detectadas = []  # peças detectadas após segmentação e classificação
    user_folder = get_user_upload_folder()

    fallback_used = False
    for img in imagens:
        if img and img.filename:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            filename, file_extension = os.path.splitext(img.filename)

            # Validar tipo de arquivo
            if file_extension.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                return jsonify({"success": False, "message": f"Formato de arquivo não suportado: {file_extension}. Use JPG, PNG ou WEBP."}), 400

            unique_filename = f"{filename}_{timestamp}{file_extension}"
            path = os.path.join(user_folder, unique_filename)

            try:
                img.save(path)
                caminho_rel = os.path.relpath(path, 'static').replace('\\', '/')
                while caminho_rel.startswith('uploads/uploads/'):
                    caminho_rel = caminho_rel.replace('uploads/uploads/', 'uploads/')
                caminhos.append(caminho_rel)
                if 'user_id' in session:
                    mongo_imagens.insert_one({
                        'user_id': session.get('user_id'), 'filename': unique_filename,
                        'path': caminho_rel, 'data': datetime.now().strftime('%d/%m/%Y %H:%M')
                    })
                # Segmentar e classificar cada peça detectada pelo YOLO
                crops = segmentar_pecas(path)
                if crops:
                    for crop in crops:
                        classe, confianca = classificar(crop)
                        cor = extrair_cor_dominante(crop)
                        pecas_detectadas.append({
                            "tipo": classe,
                            "cor": cor,
                            "sexo": sexo
                        })
                else:
                    # Fallback: classificar imagem inteira
                    fallback_used = True
                    classe, confianca = classificar(path)
                    cor = extrair_cor_dominante(path)
                    pecas_detectadas.append({
                        "tipo": classe,
                        "cor": cor,
                        "sexo": sexo
                    })
            except Exception as e:
                print(f"Erro ao processar {unique_filename}: {e}")

    # Avaliação do look usando pipeline inteligente
    combinacoes = carregar_combinacoes()
    avaliacao_dict = avaliar_combinacao(pecas_detectadas, combinacoes, ocasiao.lower() if ocasiao else None)
    valido = avaliacao_dict.get('valido', False)
    mensagem = avaliacao_dict.get('mensagem', '')
    status = avaliacao_dict.get('status', 'analise')
    resultado = {
        'imagens': caminhos,
        'pecas_detectadas': pecas_detectadas,
        'avaliacao': mensagem,
        'ok': valido,
        'sexo': sexo,
        'ocasiao': ocasiao,
        'status': status
    }

    # --- LOGGING DETALHADO ---
    user_id = session.get('user_id', 'anon')
    try:
        log_analysis(user_id, resultado, pecas_detectadas, 'success', fallback_used)
    except Exception as e:
        print(f'Erro ao logar análise: {e}')

    # Salvar no histórico (MongoDB) apenas se logado
    if 'user_id' in session:
        user_id = session.get('user_id')
        # Gerar novo id incremental por usuário
        ultimo = mongo_historico.find_one({'user_id': user_id}, sort=[('id', -1)])
        new_id = 1
        if ultimo and 'id' in ultimo:
            new_id = ultimo['id'] + 1

        novo_item = {
            "id": new_id,
            "data": datetime.now().strftime('%d/%m/%Y %H:%M'),
            "resultado": resultado,
            "nome": "",
            "user_id": user_id
        }
        mongo_historico.insert_one(novo_item)
        return jsonify({
            "success": True,
            "resultado": resultado,
            "novo_historico_item": {
                "id": novo_item["id"],
                "data": novo_item["data"],
                "nome": novo_item["nome"]
            }
        })
    else:
        return jsonify({
            "success": True,
            "resultado": resultado
        })

@app.route('/historico/<int:item_id>')
@login_required
def ver_historico(item_id):
    user_id = session.get('user_id')
    usuario_logado = bool(user_id)
    historico = list(mongo_historico.find({'user_id': user_id}, {'_id': 0}))
    item = mongo_historico.find_one({'user_id': user_id, 'id': item_id}, {'_id': 0})
    if not item:
        return redirect(url_for('index'))
    return render_template('index.html', historico=historico[::-1], resultado=item["resultado"], usuario_logado=usuario_logado)

@app.route('/historico/renomear/<int:item_id>', methods=['POST'])
@login_required
def renomear_historico(item_id):
    user_id = session.get('user_id')
    data = request.get_json()
    novo_nome = data.get("nome", "")
    mongo_historico.update_one({'user_id': user_id, 'id': item_id}, {'$set': {'nome': novo_nome}})
    return jsonify(success=True)

@app.route('/historico/deletar/<int:item_id>', methods=['POST'])
@login_required
def deletar_historico(item_id):
    user_id = session.get('user_id')
    mongo_historico.delete_one({'user_id': user_id, 'id': item_id})
    return jsonify(success=True)

# --- Rota protegida do dashboard supervisor ---
from functools import wraps

def supervisor_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        print('DEBUG SESSION:', dict(session))  # <-- Debug temporário
        user_id = session.get('user_id')
        email = session.get('email') or session.get('user_email')
        if user_id == 'supervisor' or email == 'supervisor@gmail.com':
            return f(*args, **kwargs)
        return redirect(url_for('index'))
    return decorated

@app.route('/dashboard')
@supervisor_required
def dashboard():
    return render_template('dashboard.html')

# --- Redirecionar supervisor para dashboard após login ---
from flask import redirect, url_for

@app.after_request
def redirecionar_supervisor(response):
    try:
        if request.endpoint == 'auth.login' and request.method == 'POST':
            email = session.get('email') or session.get('user_email')
            if email == 'supervisor@gmail.com':
                return redirect(url_for('dashboard'))
    except Exception:
        pass
    return response

# --- Ajustar dashboard para mostrar análises de todos os usuários ---
@app.route('/dashboard_data')
@supervisor_required
def dashboard_data():
    import collections
    dados = []
    try:
        with open(ANALYSIS_LOG_PATH, encoding='utf-8') as f:
            for line in f:
                try:
                    dados.append(json.loads(line))
                except:
                    pass
    except FileNotFoundError:
        pass
    # Mostra dados de todos os usuários (não filtra por user_id)
    classes = collections.Counter()
    cores = collections.Counter()
    status = collections.Counter()
    fallback = collections.Counter()
    ultimas = []
    for item in dados[-50:][::-1]:
        pecas = ', '.join([p.get('tipo','') for p in item.get('pecas_detectadas',[])])
        cores_str = ', '.join([p.get('cor','') for p in item.get('pecas_detectadas',[])])
        resultado = item.get('resultado',{}).get('avaliacao','')
        ultimas.append({
            'timestamp': item.get('timestamp',''),
            'user_id': item.get('user_id',''),
            'pecas': pecas,
            'cores': cores_str,
            'status': item.get('status',''),
            'fallback_used': item.get('fallback_used', False),
            'resultado': resultado
        })
        for p in item.get('pecas_detectadas', []):
            classes[p.get('tipo','')] += 1
            cores[p.get('cor','')] += 1
        status[item.get('status','')] += 1
        fallback['Fallback' if item.get('fallback_used', False) else 'Normal'] += 1
    return jsonify({
        'classes': {
            'labels': list(classes.keys()),
            'datasets': [{ 'label': 'Peças', 'data': list(classes.values()), 'backgroundColor': '#4e79a7' }]
        },
        'cores': {
            'labels': list(cores.keys()),
            'datasets': [{ 'label': 'Cores', 'data': list(cores.values()), 'backgroundColor': ['#a7c7e7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc949','#af7aa1','#ff9da7','#9c755f','#bab0ab'] }]
        },
        'status': {
            'labels': list(status.keys()),
            'datasets': [{ 'label': 'Status', 'data': list(status.values()), 'backgroundColor': ['#59a14f','#e15759','#bab0ab'] }]
        },
        'fallback': {
            'labels': list(fallback.keys()),
            'datasets': [{ 'label': 'Fallback', 'data': list(fallback.values()), 'backgroundColor': ['#edc949','#4e79a7'] }]
        },
        'ultimas': ultimas
    })

ANALYSIS_LOG_PATH = 'data/analysis_log.jsonl'

def log_analysis(user_id, resultado, pecas_detectadas, status, fallback_used, erro=None):
    import json
    from datetime import datetime
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'resultado': resultado,
        'pecas_detectadas': pecas_detectadas,
        'status': status,
        'fallback_used': fallback_used,
        'erro': erro
    }
    try:
        with open(ANALYSIS_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f'Erro ao registrar análise: {e}')

if __name__ == '__main__':
    app.run(debug=True)
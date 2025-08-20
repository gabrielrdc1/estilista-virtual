import json
import os

def carregar_combinacoes(path='data/combinacoes/combinations.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

COMBINACOES = carregar_combinacoes()

def avaliar_look(pecas_detectadas):
    """
    pecas_detectadas = [
        {"tipo": "Blouse", "cor": "Black"},
        {"tipo": "Jeans", "cor": "Blue"},
        {"tipo": "Shoes", "cor": "White"}
    ]
    """
    tipos_detectados = [p["tipo"] for p in pecas_detectadas]
    cores_detectadas = [p["cor"] for p in pecas_detectadas]

    for combinacao in COMBINACOES:
        tipos_validos = [p["tipo"] for p in combinacao["pecas"]]
        cores_validas = [p["cor"] for p in combinacao["pecas"]]

        if set(tipos_detectados).issubset(set(tipos_validos)) and set(cores_detectadas).issubset(set(cores_validas)):
            return {
                "valido": True,
                "mensagem": f"✅ Look aprovado! {combinacao['descricao']}"
            }

    sugestoes = []
    for combinacao in COMBINACOES:
        tipos_validos = [p["tipo"] for p in combinacao["pecas"]]
        cores_validas = [p["cor"] for p in combinacao["pecas"]]

        tipos_faltando = list(set(tipos_validos) - set(tipos_detectados))
        tipos_sobrando = list(set(tipos_detectados) - set(tipos_validos))

        cores_faltando = list(set(cores_validas) - set(cores_detectadas))
        cores_sobrando = list(set(cores_detectadas) - set(cores_validas))

        if len(tipos_sobrando) <= 1 and len(cores_sobrando) <= 1:
            sugestoes.append({
                "sugestao": combinacao["descricao"],
                "trocar_tipo": tipos_sobrando,
                "trocar_cor": cores_sobrando
            })

    mensagem = "⚠️ Look não recomendado.\n"

    if sugestoes:
        mensagem += "Sugestão de melhoria:\n"
        melhor = sugestoes[0] 
        if melhor["trocar_tipo"]:
            mensagem += f" • Trocar peça: {', '.join(melhor['trocar_tipo'])}\n"
        if melhor["trocar_cor"]:
            mensagem += f" • Mudar cor: {', '.join(melhor['trocar_cor'])}\n"
        mensagem += f" • Sugestão de look: {melhor['sugestao']}"
    else:
        mensagem += " • Nenhuma sugestão próxima encontrada."

    return {
        "valido": False,
        "mensagem": mensagem
    }

def avaliar_combinacao(pecas_detectadas, combinacoes, ocasiao=None):
    tipos = {p['tipo'].lower(): p['cor'] for p in pecas_detectadas}
    blusa = None
    calca = None
    tenis = None
    for t in tipos:
        if 'blusa' in t:
            blusa = tipos[t]
        elif 'calca' in t:
            calca = tipos[t]
        elif 'tenis' in t:
            tenis = tipos[t]
    if blusa and calca and tenis:
        # Filtra por ocasião se informada
        combinacoes_filtradas = [c for c in combinacoes if not ocasiao or c.get('ocasiao') == ocasiao]
        # Procura combinação exata
        for c in combinacoes_filtradas:
            if c['blusa'].lower() == blusa.lower() and c['calca'].lower() == calca.lower() and c['tenis'].lower() == tenis.lower():
                return {"valido": c['valido'], "mensagem": c['mensagem']}
        # Procura combinação quase válida (só 1 diferença)
        melhor = None
        melhor_dif = 4
        for c in combinacoes_filtradas:
            dif = 0
            if c['blusa'].lower() != blusa.lower():
                dif += 1
            if c['calca'].lower() != calca.lower():
                dif += 1
            if c['tenis'].lower() != tenis.lower():
                dif += 1
            if dif < melhor_dif:
                melhor = c
                melhor_dif = dif
        if melhor and melhor_dif == 1:
            # Só um ajuste necessário
            ajustes = []
            if melhor['blusa'].lower() != blusa.lower():
                ajustes.append(f"Troque a blusa para {melhor['blusa']}")
            if melhor['calca'].lower() != calca.lower():
                ajustes.append(f"Troque a calça para {melhor['calca']}")
            if melhor['tenis'].lower() != tenis.lower():
                ajustes.append(f"Troque o tênis para {melhor['tenis']}")
            return {
                "valido": False,
                "mensagem": f"Seu look está quase perfeito para a ocasião! {', '.join(ajustes)}.\nSugestão: {melhor['mensagem']}"
            }
        # Se não achou próximo, sugere uma referência
        if melhor:
            return {
                "valido": False,
                "mensagem": f"Look detectado: Blusa {blusa}, Calça {calca}, Tênis {tenis}.\nSugestão de referência para a ocasião: Blusa {melhor['blusa']}, Calça {melhor['calca']}, Tênis {melhor['tenis']}. {melhor['mensagem']}"
            }
    else:
        faltando = []
        if not blusa: faltando.append('blusa')
        if not calca: faltando.append('calça')
        if not tenis: faltando.append('tênis')
        if faltando:
            return {
                "valido": False,
                "mensagem": f"Para uma avaliação completa do look, envie também: {', '.join(faltando)}."
            }
        return {
            "valido": False,
            "mensagem": "Não foi possível avaliar o look. Tente novamente com outra imagem."
        }
    # Garantir retorno seguro
    return {
        "valido": False,
        "mensagem": "Não foi possível avaliar o look. Tente novamente com outra imagem."
    }

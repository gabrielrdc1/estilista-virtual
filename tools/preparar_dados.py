import os
import shutil
import pandas as pd

CSV_PATH = 'data/raw/fashion-dataset/styles.csv'
IMAGES_PATH = 'data/raw/fashion-dataset/images'
DEST_PATH = 'data/processed/por_categoria'

def copiar_imagem(img_id, categoria):
    origem = os.path.join(IMAGES_PATH, f"{img_id}.jpg")
    destino_dir = os.path.join(DEST_PATH, categoria)
    destino = os.path.join(destino_dir, f"{img_id}.jpg")

    if not os.path.exists(origem):
        return False

    os.makedirs(destino_dir, exist_ok=True)
    shutil.copyfile(origem, destino)
    return True

def preparar_dados():
    df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
    df = df.dropna(subset=['id', 'articleType'])

    print(f"[INFO] Total de registros: {len(df)}")
    count_ok = 0

    for _, row in df.iterrows():
        img_id = str(row['id'])
        categoria = row['articleType'].strip().replace(' ', '_')

        if copiar_imagem(img_id, categoria):
            count_ok += 1

    print(f"[OK] Imagens copiadas com sucesso: {count_ok}")

if __name__ == '__main__':
    preparar_dados()

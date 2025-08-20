import os
from PIL import Image

root = r'data/processed/por_categoria'
removidos = 0
for dirpath, _, filenames in os.walk(root):
    for fname in filenames:
        fpath = os.path.join(dirpath, fname)
        try:
            with Image.open(fpath) as img:
                img.verify()
        except Exception:
            print(f"Removendo arquivo inválido: {fpath}")
            os.remove(fpath)
            removidos += 1
print(f"Total de arquivos removidos: {removidos}")

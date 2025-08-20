import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
import glob
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix



BATCH_SIZE = 32
IMG_SIZE = (224, 224)
DATA_DIR = os.path.join('data', 'train')
VAL_DIR = os.path.join('data', 'val')
EPOCHS = 15  # Reduzido para acelerar
MODEL_PATH = os.path.join('modelo', 'cnn_model_mobilenetv2.h5')


print("[INFO] Carregando imagens de treino:", DATA_DIR)
train_ds = image_dataset_from_directory(
    DATA_DIR,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)
print("[INFO] Carregando imagens de validação:", VAL_DIR)
val_ds = image_dataset_from_directory(
    VAL_DIR,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)


# Detectar classes
class_names = train_ds.class_names
print(f"[INFO] Classes encontradas: {class_names}")

# Diagnóstico de shapes dos dados
for x, y in train_ds.take(1):
    print("[DIAG] Shape X treino:", x.shape, "Shape Y treino:", y.shape)
for x, y in val_ds.take(1):
    print("[DIAG] Shape X val:", x.shape, "Shape Y val:", y.shape)

# Validação cruzada k-fold
# Carregar caminhos e rótulos
file_paths = []
labels = []
for idx, class_name in enumerate(class_names):
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        for path in glob.glob(os.path.join(DATA_DIR, class_name, ext)):
            file_paths.append(path)
            labels.append(idx)
file_paths = np.array(file_paths)
labels = np.array(labels)

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
AUTOTUNE = tf.data.AUTOTUNE
fold_metrics = []
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
print("[INFO] Iniciando validação cruzada k-fold...")
for fold, (train_idx, val_idx) in enumerate(kf.split(file_paths, labels), 1):
    print(f"[INFO] Fold {fold}")
    train_paths, train_labels = file_paths[train_idx], labels[train_idx]
    val_paths, val_labels = file_paths[val_idx], labels[val_idx]
    def process_path(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3)
        img.set_shape([None, None, 3])  # Garante shape conhecido
        img = tf.image.resize(img, IMG_SIZE)
        img = preprocess_input(img)
        return img, label
    train_ds_fold = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))\
        .map(process_path).shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    val_ds_fold = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))\
        .map(process_path).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    # Recriar modelo para cada fold
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = False
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(len(class_names), activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_ds_fold, validation_data=val_ds_fold, epochs=EPOCHS, callbacks=[early_stop], verbose=0)
    eval_res = model.evaluate(val_ds_fold, verbose=0)
    print(f"[INFO] Fold {fold} - Loss: {eval_res[0]:.4f}, Acc: {eval_res[1]:.4f}")
    fold_metrics.append(eval_res)

avg_loss = np.mean([m[0] for m in fold_metrics])
avg_acc = np.mean([m[1] for m in fold_metrics])
print(f"[INFO] Validação cruzada - Loss médio: {avg_loss:.4f}, Acc médio: {avg_acc:.4f}")


# Data augmentation para treino final
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.GaussianNoise(0.05)  # Robustez contra ruído
], name='data_augmentation')

def preprocess_train(x, y):
    x = preprocess_input(x)
    x = data_augmentation(x, training=True)
    return x, y

def preprocess_val(x, y):
    x = preprocess_input(x)
    return x, y

# Treino final em toda base
print("[INFO] Treinando modelo final em toda a base...")
train_ds_final = train_ds.map(preprocess_train).cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds_final = val_ds.map(preprocess_val).cache().prefetch(AUTOTUNE)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False
inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = models.Model(inputs, outputs)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(train_ds_final, validation_data=val_ds_final, epochs=EPOCHS, callbacks=[early_stop])

# Avaliação detalhada no conjunto de validação (precisão/recall/F1 por classe)
print("[INFO] Gerando relatório de classificação na validação...")
y_true_all, y_pred_all = [], []
for x_batch, y_batch in val_ds_final:
    preds = model.predict(x_batch, verbose=0)
    y_pred_all.extend(np.argmax(preds, axis=1))
    y_true_all.extend(y_batch.numpy())

try:
    print(classification_report(y_true_all, y_pred_all, target_names=class_names, digits=4))
    cm = confusion_matrix(y_true_all, y_pred_all)
    print("[INFO] Matriz de confusão:\n", cm)
except Exception as e:
    print("[WARN] Falha ao gerar relatório de classificação:", e)
model.save(MODEL_PATH)
print(f"[INFO] Modelo final salvo em: {MODEL_PATH}")

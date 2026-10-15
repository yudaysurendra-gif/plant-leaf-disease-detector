"""Optional ML version: train a CNN on the PlantVillage dataset.

1. Download PlantVillage (Kaggle: "plantvillage dataset") and arrange it as:
       data/<class_name>/*.jpg      e.g. data/Tomato_Late_blight/..., data/Tomato_healthy/...
2. pip install tensorflow
3. python train_leaf_cnn.py
Then predict with: python train_leaf_cnn.py path/to/leaf.jpg
"""
import sys
import numpy as np
import tensorflow as tf

IMG, BATCH, EPOCHS = (128, 128), 32, 8

def train():
    ds_args = dict(validation_split=0.2, seed=42, image_size=IMG, batch_size=BATCH)
    train = tf.keras.utils.image_dataset_from_directory("data", subset="training", **ds_args)
    val = tf.keras.utils.image_dataset_from_directory("data", subset="validation", **ds_args)
    classes = train.class_names
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1 / 255, input_shape=IMG + (3,)),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.Conv2D(32, 3, activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(len(classes), activation="softmax"),
    ])
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=EPOCHS)
    model.save("leaf_model.keras")
    open("classes.txt", "w").write("\n".join(classes))

def predict(path):
    model = tf.keras.models.load_model("leaf_model.keras")
    classes = open("classes.txt").read().split("\n")
    img = tf.keras.utils.load_img(path, target_size=IMG)
    p = model.predict(np.expand_dims(tf.keras.utils.img_to_array(img), 0))[0]
    print(f"{classes[p.argmax()]}  (confidence {p.max():.1%})")

if __name__ == "__main__":
    predict(sys.argv[1]) if len(sys.argv) > 1 else train()

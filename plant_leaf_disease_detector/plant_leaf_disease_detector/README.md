# Plant Leaf Disease Detector

- index.html : open in any browser. Upload a leaf photo (or try a sample) to get a colour-based damage estimate.
- train_leaf_cnn.py : optional CNN trained on the PlantVillage dataset (needs TensorFlow).

Run the ML version:
    pip install tensorflow
    python train_leaf_cnn.py            # train (data/<class_name>/*.jpg)
    python train_leaf_cnn.py leaf.jpg   # predict

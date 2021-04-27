import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy, BinaryAccuracy, AUC, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from medicalgan.models.architecture.binary_vgg import VGG
from medicalgan.models.architecture.cnn_oasis import CNN_OASIS
from medicalgan.models.architecture.cnn_batchnorm import CNN_BatchNorm
from medicalgan.utils.utils import get_dataset, get_aug_dataset, get_model, get_model_path, get_tb_dir, normround, f1_score

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)
tf.random.set_seed(42)

DATASET = "oasis"
MODEL_TYPE = "ad_classifier"
MODEL_NOTE = "cross_validation"

def train(plane, depth):
    """
    Build and train CNN with crossvalidation for alzheimer's detection.
    """
    if plane == "transverse":
        rows, cols, channels = (208, 176, 1)
    elif plane == "coronal":
        rows, cols, channels = (176, 176, 1)
    else:
        print("Incompatable plane")
        return
    img_shape = (rows, cols, channels)
    batch_size = 12
    
    # get train data as array for k-fold cross-validation
    train_dir = "resources/data/oasis/{0}/{1}/crossval_train".format(depth, plane)  
    neg_dir = os.path.join(train_dir, "0")
    pos_dir = os.path.join(train_dir, "1")
    X_neg = [img_to_array(load_img(os.path.join(neg_dir, path), color_mode="grayscale", target_size=img_shape)) for path in sorted(os.listdir(neg_dir))]
    X_pos = [img_to_array(load_img(os.path.join(pos_dir, path), color_mode="grayscale", target_size=img_shape)) for path in sorted(os.listdir(pos_dir))]
    
    # subjects should be kept together (6 slices per subject)
    X = np.asarray(X_neg + X_pos)
    y = np.asarray([0] * len(X_neg) + [1] * len(X_pos))
    X_subjects = np.asarray(np.split(X, len(X) // 6))
    y_subjects = np.asarray(np.split(y, len(X) // 6))

    # get the test data
    test_dir = "resources/data/oasis/{0}/{1}/test".format(depth, plane)
    # test_data = get_dataset(test_dir, rows, cols, batch_size, label_mode="binary")
    neg_dir = os.path.join(test_dir, "0")
    pos_dir = os.path.join(test_dir, "1")
    X_neg = [img_to_array(load_img(os.path.join(neg_dir, path), color_mode="grayscale", target_size=img_shape)) for path in sorted(os.listdir(neg_dir))]
    X_pos = [img_to_array(load_img(os.path.join(pos_dir, path), color_mode="grayscale", target_size=img_shape)) for path in sorted(os.listdir(pos_dir))]    
    X_test = np.asarray(X_neg + X_pos)
    y_test = np.asarray([0] * len(X_neg) + [1] * len(X_pos))
    # test_shuffle_idx = np.random.permutation(len(X_test))
    # X_test, y_test = X_test[test_shuffle_idx], y_test[test_shuffle_idx]

    # delete unused large lists
    for var in [X, y, X_neg, X_pos]:
        del(var)
    
    # callbacks
    earlystopping = EarlyStopping(
        monitor="val_loss", 
        patience=10, 
        restore_best_weights=True,
    )
    tensorboard = TensorBoard(get_tb_dir(DATASET, MODEL_TYPE, MODEL_NOTE))
    # callbacks = [earlystopping, tensorboard]
    callbacks = [earlystopping]

    # opt = Adam(learning_rate=1e-5)
    opt = SGD(learning_rate=1e-5, momentum=0.9, nesterov=True)
    loss = BinaryCrossentropy(from_logits=True)
    metrics = [BinaryAccuracy()]#, AUC(), Precision(), Recall(), f1_score]

    # set model hyperparameters
    epochs = 100
    workers=4
    max_queue_size=10
    use_multiprocessing=True

    # list of results
    crossval_results = []

    # cross validation splitter, per-subject labels are the same so take mean as subject label for stratified sampling
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    for i, (train_idx, val_idx) in enumerate(kf.split(X_subjects, y_subjects.mean(axis=1))):

        print(f"\n------------\n\nTraining fold {i+1}:\n")

        # get train and val sets - collapse subjects together
        X_train_subjects, X_val_subjects = X_subjects[train_idx], X_subjects[val_idx]
        y_train_subjects, y_val_subjects = y_subjects[train_idx], y_subjects[val_idx]
        X_train, X_val = np.concatenate(X_train_subjects), np.concatenate(X_val_subjects)
        y_train, y_val = np.concatenate(y_train_subjects), np.concatenate(y_val_subjects)

        # shuffle subject slices throughout sets
        train_shuffle_idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[train_shuffle_idx], y_train[train_shuffle_idx]
        val_shuffle_idx = np.random.permutation(len(X_val))
        X_val, y_val = X_val[val_shuffle_idx], y_val[val_shuffle_idx]

        # compile new model
        architecture = CNN_OASIS(img_shape)
        cnn = architecture.cnn
        cnn.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics,
        )

        # fit model and validate on fold
        cnn.fit(
            X_train, y_train,
            validation_data = (X_val, y_val),
            epochs=epochs,
            # workers=workers,
            # max_queue_size=max_queue_size,
            # use_multiprocessing=use_multiprocessing,
            callbacks=callbacks,
        )
        
        # save model
        # cnn_path = get_model_path(DATASET, MODEL_TYPE, MODEL_NOTE)
        # cnn.save(cnn_path)

        # fast evalutate
        print("\nFold training complete!\n\nEvaluating...")        
        crossval_results.append(cnn.evaluate(X_test, y_test))

        evaluate(cnn, X_test, y_test)

        return

    avg_results = np.asarray(crossval_results).mean(axis=0)
    print("\n------------\n\nCrossvalidation complete!\n")
    print(f"loss = {avg_results[0]:.4f}\nacc  = {avg_results[1]:.4f}\n")


def evaluate(model, X, y):
    """
    Custom evaluation of model on test set.
    """
    preds = model.predict(X).flatten().round()
    # set correct if prediction is the same as label
    correct = np.array([1 if pred == label else 0 for (pred, label) in zip(preds, y)])
    # split per subject
    samples_per_subject = 6
    num_subjects = len(y) // samples_per_subject
    subjects_correct = np.asarray(np.split(correct, num_subjects))
    # get average prediction
    subjects_score = [normround(np.sum(subject_correct) / samples_per_subject) for subject_correct in subjects_correct]
    # get overall subject_accuracy
    subject_accuracy = np.sum(subjects_score) / num_subjects

    # evaluate
    model.evaluate(X, y)
    print(f"\nSubject accuracy: {100*subject_accuracy:.2f}%\n")
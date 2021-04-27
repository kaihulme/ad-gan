import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy, BinaryAccuracy, AUC, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from medicalgan.models.architecture.binary_vgg import VGG
from medicalgan.models.architecture.cnn_oasis import CNN_OASIS
from medicalgan.models.architecture.cnn_batchnorm import CNN_BatchNorm
from medicalgan.utils.utils import get_dataset, get_aug_dataset, get_model, get_model_path, get_tb_dir, normround, f1_score

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

DATASET = "oasis"
MODEL_TYPE = "ad_classifier"
MODEL_NOTE = "optimised_cnn"


def train(plane, depth):
    """
    Build and train CNN for tumour detection in fMRI images.
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
    
    # data generators
    train_dir = "resources/data/oasis/{0}/{1}/train".format(depth, plane)    
    val_dir = "resources/data/oasis/{0}/{1}/val".format(depth, plane)
    test_dir = "resources/data/oasis/{0}/{1}/test".format(depth, plane)

    # train_data = get_dataset(train_dir, rows, cols, batch_size, label_mode="binary")
    train_data = get_dataset(train_dir, rows, cols, batch_size, label_mode="binary")
    val_data = get_dataset(val_dir, rows, cols, batch_size, label_mode="binary")
    test_data = get_dataset(test_dir, rows, cols, batch_size, label_mode="binary")
    
    # callbakcs
    earlystopping = EarlyStopping(
        monitor="val_loss", 
        patience=10, 
        restore_best_weights=True,
    )
    reducelronplateau = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.9,
        patience=3,
        verbose=1,
    )
    tensorboard = TensorBoard(
        get_tb_dir(DATASET, MODEL_TYPE, MODEL_NOTE)
    )
    callbacks = [earlystopping, tensorboard]
    # callbacks = [earlystopping, reducelronplateau, tensorboard]

    expdecay_lr = ExponentialDecay(1e-4, decay_steps=1000, decay_rate=0.95)
    opt = SGD(learning_rate=expdecay_lr, momentum=0.9, nesterov=True)
    loss = BinaryCrossentropy(from_logits=True)
    metrics = [BinaryAccuracy(), AUC(), Precision(), Recall(), f1_score]

    architecture = CNN_OASIS(img_shape, show_summary=True)
    cnn = architecture.cnn
    cnn.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics,
    )

    epochs = 1000
    workers=4
    max_queue_size=10
    use_multiprocessing=True

    cnn.fit(
        train_data,
        validation_data = val_data,
        epochs=epochs,
        workers=workers,
        max_queue_size=max_queue_size,
        # use_multiprocessing=use_multiprocessing,
        callbacks=callbacks,
    )
    
    # save model
    cnn_path = get_model_path(DATASET, MODEL_TYPE, MODEL_NOTE)
    cnn.save(cnn_path)
    # fast evalutate
    print("\nTraining complete!\n\nEvaluating...")
    cnn.evaluate(test_data)


def evaluate(plane, depth):
    """
    Custom evaluation of model on test set.
    """
    # get most recent model
    cnn = get_model(DATASET, MODEL_TYPE, MODEL_NOTE)
    # set image shape
    if plane == "transverse":
        rows, cols, channels = (208, 176, 1)
    elif plane == "coronal":
        rows, cols, channels = (176, 176, 1)
    else:
        print("Incompatable plane")
        return
    batch_size = 12
    # get test dataset
    test_dir = "resources/data/oasis/{0}/{1}/test".format(depth, plane)
    test_data = get_dataset(test_dir, rows, cols, batch_size, label_mode="binary", shuffle=False)
    # get real labels and predictions
    y = np.asarray([y for _, y in test_data]).flatten()
    preds = cnn.predict(test_data).flatten().round()
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
    cnn.evaluate(test_data)

    print("\n", subjects_correct)
    print("\n", subjects_score)
    print(f"\nSubject accuracy: {100*subject_accuracy:.2f}%\n")
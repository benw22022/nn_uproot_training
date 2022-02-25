"""

Training Script
________________________________________________
Script to run the neural network training
"""
import os
import glob
# import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tqdm
# from source.datagenerator import DataGenerator
from source.complex_datagenerator import DataGenerator
# from source.dataloader import DataLoader
from config.config import config_dict, models_dict
from config.features import make_feature_handler
import subprocess


def grab_files(directory, glob_exprs):
    files = []
    for expr in glob_exprs: 
        files.append([glob.glob(os.path.join(directory, expr, "*.root"))])
        
    return files

def train():

    files = grab_files("../NTuples/", ["*Gammatautau*", "*JZ1*", "*JZ2*", "*JZ3*", "*JZ4*", "*JZ5*", "*JZ6*", "*JZ7*", "*JZ8*"])
    
    files_dict = {"Gammatautau": files[0], 
                  "JZ1": files[1],
                  "JZ2": files[2],
                  "JZ3": files[3],
                  "JZ4": files[4],
                  "JZ5": files[5],
                  "JZ6": files[6],
                  "JZ7": files[7],
                  "JZ8": files[8],
                  }

    # training_batch_generator = DataGenerator(files, make_feature_handler(), batch_size=512)
    training_batch_generator = DataGenerator(files_dict, make_feature_handler(), batch_size=100000)
    # training_batch_generator = DataLoader(files[0], make_feature_handler(), batch_size=100000)
    # training_batch_generator2 = DataLoader(files[1], make_feature_handler(), batch_size=100000)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Initialize Model
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # model_config = config_dict
    # model = models_dict["DSNN_2Step"](model_config)

    # # Configure callbacks
    # early_stopping = tf.keras.callbacks.EarlyStopping(
    #     monitor="val_loss", min_delta=0.0001,
    #     patience=20, verbose=0, restore_best_weights=True)

    # model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='weights-{epoch:02d}.h5', monitor='val_loss', verbose=0, save_best_only=False,
    #                                    save_weights_only=True)                                               

    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=3, min_lr=1e-9)
    
    # callbacks = [early_stopping, model_checkpoint, reduce_lr]

    # # Compile and summarise model
    # model.summary()
    # opt = tf.keras.optimizers.Adam()
    # model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=[tf.keras.metrics.CategoricalAccuracy()], )
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Train Model
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    # process_pid = os.getpid()
    # os.system("rm mem.log")

    # with open("mem.log", "w") as logfile:
    
    #     subprocess.Popen([f"top -d 1 -b | grep {process_pid}"], shell=True, stdout=logfile)

    while True:
        for i in tqdm.tqdm(range(0, len(training_batch_generator))):
            x = training_batch_generator[i]


    # history = model.fit(training_batch_generator, epochs=200, class_weight=class_weight, callbacks=callbacks,
    #                     # validation_data=validation_batch_generator, validation_freq=1, 
    #                     verbose=1, steps_per_epoch=len(training_batch_generator),
    #                     use_multiprocessing=False, workers=1)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Make Plots 
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # Loss History
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='train')
    ax.plot(history.history['val_loss'], label='val')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(os.path.join("plots", "loss_history.png"))
    

    # Accuracy history
    fig, ax = plt.subplots()
    ax.plot(history.history['categorical_accuracy'], label='train')
    ax.plot(history.history['val_categorical_accuracy'], label='val')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Categorical Accuracy')
    ax.legend()
    plt.savefig(os.path.join("plots", "accuracy_history.png"))

    # Return best validation loss and accuracy
    best_val_loss_epoch = np.argmin(history.history["val_loss"])
    best_val_loss = history.history["val_loss"][best_val_loss_epoch]
    best_val_acc = history.history["val_categorical_accuracy"][best_val_loss_epoch]

    logger.log(f"Best Epoch: {best_val_loss_epoch + 1} -- Val Loss = {best_val_loss} -- Val Acc = {best_val_acc}")

    return best_val_loss, best_val_acc

if __name__ == "__main__":#
    
    train()
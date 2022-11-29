import tensorflow as tf
import keras
import json
import argparse

from keras import layers, models
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from keras.callbacks import ModelCheckpoint
from utils.configure import Config
from utils.utils import *


# build model
def conv1D_scr(input_shape, cfg):
    model = keras.Sequential()

    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(cfg.num_class, activation='softmax'))

    opt = tf.optimizers.Adam(cfg.lr)
    model.compile(optimizer=opt, loss=cfg.loss, metrics='acc')

    print(model.summary())

    return model


# build Multi-head CNN
def Multi_head_conv1D_scr(input_shape, cfg):
    # head 1
    inputs1 = layers.Input(shape=input_shape)
    conv1 = layers.Conv1D(filters=32, kernel_size=3, activation='relu')(inputs1)
    drop1 = layers.Dropout(0.5)(conv1)
    pool1 = layers.MaxPooling1D(pool_size=2)(drop1)
    flat1 = layers.Flatten()(pool1)

    # head 2
    inputs2 = layers.Input(shape=input_shape)
    conv2 = layers.Conv1D(filters=32, kernel_size=5, activation='relu')(inputs2)
    drop2 = layers.Dropout(0.5)(conv2)
    pool2 = layers.MaxPooling1D(pool_size=2)(drop2)
    flat2 = layers.Flatten()(pool2)

    # head 3
    inputs3 = layers.Input(shape=input_shape)
    conv3 = layers.Conv1D(filters=32, kernel_size=11, activation='relu')(inputs3)
    drop3 = layers.Dropout(0.5)(conv3)
    pool3 = layers.MaxPooling1D(pool_size=2)(drop3)
    flat3 = layers.Flatten()(pool3)

    # concatenate three heads
    merged = layers.concatenate([flat1, flat2, flat3])
    dense1 = layers.Dense(100, activation='relu')(merged)
    outputs = layers.Dense(cfg.num_class, activation='softmax')(dense1)
    model = models.Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

    # optimize
    opt = tf.optimizers.Adam(cfg.lr)
    model.compile(optimizer=opt, loss=cfg.loss, metrics='acc')

    print(model.summary())

    return model


if __name__ == "__main__":
    cfg = Config()

    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--is_org_data_only_process", default='Yes', type=str, required=True,
                        help="Select original data including process data (and) task data")
    parser.add_argument("--is_flt", default='Yes', type=str, required=True,
                        help="Select the filtered data")
    args = parser.parse_args()

    # split data source
    if (args.is_org_data_only_process == 'Yes') and (args.is_flt == 'No'):
        X_train, X_test, y_train, y_test = load_org_data_only_process(cfg.org_aursad_cln_path, expand_flag=True)
    elif (args.is_org_data_only_process == 'Yes') and (args.is_flt == 'Yes'):
        X_train, X_test, y_train, y_test = load_org_data_only_process(cfg.org_aursad_flt_path, expand_flag=True)
    elif (args.is_org_data_only_process == 'No') and (args.is_flt == 'No'):
        X_train, X_test, y_train, y_test = load_org_data_process_task(cfg.org_aursad_cln_path, expand_flag=True)
    elif (args.is_org_data_only_process == 'No') and (args.is_flt == 'Yes'):
        X_train, X_test, y_train, y_test = load_org_data_process_task(cfg.org_aursad_flt_path, expand_flag=True)
    
    # set the path for the results including the model, loss, acc, f1, recall, and precision
    model_path, loss_img, acc_img, precision, recall, f1 = cfg.model_parameters_set_process_task("Conv1D_org_data", args.is_org_data_only_process, args.is_flt)

    # print the shape of training data sample
    print(X_train.shape[1:])
    # save the best model
    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]


    # # construct Conv1D_org_data
    # model = conv1D_scr(X_train.shape[1:], cfg)
    # # # training model
    # history = model.fit(X_train, y_train, epochs=cfg.epochs, batch_size=cfg.batch_size,
    #                     validation_data=(X_test, y_test), callbacks=callbacks_list, verbose=1)

    # construct multi-head Conv1D_org_data
    model = Multi_head_conv1D_scr(X_train.shape[1:], cfg)
    history = model.fit([X_train, X_train, X_train], y_train, epochs=cfg.epochs, batch_size=cfg.batch_size,
                        validation_data=([X_test, X_test, X_test], y_test), callbacks=callbacks_list, verbose=1)


    # save model
    model.save(model_path)

    # plot the acc and loss
    plot_loss_acc(history, loss_img, acc_img)

    # get f1, precision and recall scores
    model = keras.models.load_model(model_path)

    # # Conv1D_org_data
    # y_pred1 = model.predict(X_test)

    # Multi-head Conv1D_org_data
    y_pred1 = model.predict([X_test, X_test, X_test])

    y_pred = np.argmax(y_pred1, axis=1)

    # save f1, precision, and recall scores
    scores = {precision: precision_score(y_test, y_pred, average="macro"),
              recall: recall_score(y_test, y_pred, average="macro"),
              f1: f1_score(y_test, y_pred, average="macro")}
    with open(cfg.scores_file_path, 'a') as outfile:
        json.dump(scores, outfile)
    outfile.close()

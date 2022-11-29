import keras
import json
import argparse

from keras import layers, utils, models
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.callbacks import ModelCheckpoint

from utils.configure import Config
from utils.utils import *


# build model
def LSTM_src(X_train, y_train, X_test, y_test, cfg, features, outputs, callbacks_list):
    features, outputs = features, outputs
    input_shape = (X_train.shape[1], features)
   
    # create model
    model = models.Sequential()
    model.add(layers.LSTM(400, input_shape=input_shape))
    model.add(layers.Dropout(cfg.dropout))
    model.add(layers.Dense(400, activation='relu'))
    model.add(layers.Dense(outputs, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # Training and evaluation
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=cfg.epochs,
                        batch_size=cfg.batch_size, callbacks=[callbacks_list],
                        verbose=cfg.verbose)

    print(model.summary())

    return model, history


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
        X_train, X_test, y_train, y_test = load_org_data_process_task(cfg.org_aursad_cln_path, expand_flag=True)
    elif (args.is_org_data_only_process == 'Yes') and (args.is_flt == 'Yes'):
        X_train, X_test, y_train, y_test = load_org_data_process_task(cfg.org_aursad_flt_path, expand_flag=True)
    # else:
    #     X_train, X_test, y_train, y_test = load_org_data_process_and_task(cfg.org_aursad_path, expand_flag=True)
    # set up parameters

    model_path, loss_img, acc_img, precision, recall, f1 = cfg.model_parameters_set_process_task("LSTM_org_data", args.is_org_data_only_process, args.is_flt)

    # callbacks = [keras.callbacks.EarlyStopping(patience=cfg.patience, restore_best_weights=True)]

    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]

    #features, class
    features, outputs = X_train.shape[2], cfg.num_class
    # # reshape training data
    # X_train = X_train.reshape((X_train.shape[0], cfg.steps, 1, cfg.length, X_train.shape[2]))
    # X_test = X_test.reshape((X_test.shape[0], cfg.steps, 1, cfg.length, X_train.shape[2]))
     # one-hot encoder
    y_train, y_test = utils.np_utils.to_categorical(y_train, num_classes=4), utils.np_utils.to_categorical(y_test, num_classes=4)

    # construct Conv2D
    model, history = LSTM_src(X_train, y_train, X_test, y_test, cfg, features, outputs, callbacks_list)

    # save model
    model.save(model_path)

    # plot the acc and loss
    plot_loss_acc(history, loss_img, acc_img)

    # get f1, precision and recall scores
    model = keras.models.load_model(model_path)

    y_pred1 = model.predict(X_test)
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred1, axis=1)

    # save f1, precision, and recall scores
    scores = {precision: precision_score(y_test, y_pred, average="macro"),
              recall: recall_score(y_test, y_pred, average="macro"),
              f1: f1_score(y_test, y_pred, average="macro")}
    with open(cfg.scores_file_path, 'a') as outfile:
        json.dump(scores, outfile)
    outfile.close()

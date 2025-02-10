import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import random
import crossvalidation
from sklearn.metrics import log_loss
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
import tensorflow as tf
import datetime
from tensorflow.keras.losses import BinaryCrossentropy

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import log_loss


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

def set_seed(seed):
    np.random.seed(seed)
   # tf.random.set_seed(seed)
    random.seed(seed)



def neural_model(df, train_cols, learning_rate=0.001, batch_size=128, epochs=300, seed=42):

    set_seed(seed)
   # print(train_cols)
    #df = df[train_cols]

    df.dropna(inplace=True)
    df.dropna(how='all', inplace=True)
    df.drop_duplicates(subset=['date'],inplace=True)

    train_datasets, test_datasets= crossvalidation.run_split_process(df)
    train_dataset =train_datasets[-1] 
    test_dataset = test_datasets[-1]

    ########feature and target seperation
    feature_cols =train_cols
    target_cols = 'label'
    n_components = len(train_cols)

    X_train =train_dataset[feature_cols]
    #lookback=30

   # for i in range(lookback, len(X_train)):

    
    y_train =train_dataset[target_cols]

    X_test = test_dataset[feature_cols]
    y_test = test_dataset[target_cols]
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)




    ####### Class weighting
    y_train_int = y_train + 1 
    
    class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_int),
    y=y_train_int
    )
# Convert to dictionary format:
    class_weights_dict = dict(enumerate(class_weights))

    ############ Fix One Hot Encoding 
    y_train = y_train + 1  # Now classes become 0, 1, 2
    y_test = y_test + 1

    y_train = to_categorical(y_train, num_classes=3)
    y_test = to_categorical(y_test, num_classes=3)
    print('categegory check ##########',y_train)
#    pca = PCA(n_components=n_components)
 #   X_train = pca.fit_transform(X_train)
  #  X_test = pca.transform(X_test)


    

    
    ###########PCA
    
    #pca = PCA(n_components=n_components)
    #X_train = pca.fit_transform(X_train)
    #X_test = pca.transform(X_test)
    
    ###########SCALER


    ########## Model Structure
    # Adjust final layer: 3 outputs for one-hot encoding
    
    model = models.Sequential()
    model.add(layers.Dense(32, activation='swish', input_shape=(n_components,), 
                        kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(16, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    # Final layer with 3 outputs
    model.add(layers.Dense(2, activation='softmax'))

    # Compile using categorical crossentropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(class_id=0),tf.keras.metrics.Precision(class_id=1),tf.keras.metrics.Precision(class_id=2), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )



    ##### Early Stopping and learning rate
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.001, patience=15, min_lr=1e-10)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping, lr_scheduler,tensorboard_callback],class_weight=class_weights_dict)
    
    test_loss, test_acc, test_precision_dwn,test_precision_n,test_precision_up, test_recall, test_auc = model.evaluate(X_test, y_test)
    
   
    
    print(f'testloss:{test_loss}')
    print(f'Test accuracy: {test_acc}')
    #print(f'F1 Score:{f1}')
    print(f'Test precision dwn: {test_precision_dwn}')
    print(f'Test precision neutral: {test_precision_n}')
    print(f'Test precision up: {test_precision_up}')
    print(f'Test recall: {test_recall}')
    print(f'Test AUC: {test_auc}')
    

   # precision_class_0 = tf.keras.metrics.Precision(class_id=0)
   # precision_class_1 = tf.keras.metrics.Precision(class_id=1)
   # precision_class_2 = tf.keras.metrics.Precision(class_id=2)


    #print(f'Precision down:{precision_class_0}')
    #print(f'Precision neutral: {precision_class_1}')
    #print(f'Precision up: {precision_class_2}')


    y_pred_proba = model.predict(X_test)
    log_loss_value = log_loss(y_test, y_pred_proba)
    print(f'Log Loss: {log_loss_value}')

    proba_df = pd.DataFrame(y_pred_proba, columns=['Proba_Class_0', 'Proba_Class_1', 'Proba_Class_2'])

    test_results = test_dataset.reset_index(drop=True)
    test_results = pd.concat([test_results, proba_df], axis=1)
    test_results['True_Label'] = np.argmax(y_test, axis=1)

    test_results = test_results[['label','Proba_Class_0', 'Proba_Class_1', 'Proba_Class_2','formatted_time']]
    '''
    test_results['Close'] = startprice.reset_index(drop=True)
    test_results['touch_price'] = endprice.reset_index(drop=True)
    test_results['Date'] = Dates.reset_index(drop=True)
    test_results['endbarrier_unix'] = enddate.reset_index(drop=True)
    test_results['upper_barrier'] = upperbarrier.reset_index(drop=True)
    test_results['lower_barrier'] = lowerbarrier.reset_index(drop=True)
    test_results['touch_time_unix'] = touchtime.reset_index(drop=True)
    '''
    test_results.to_csv(f'test_result_1.csv')
    print(test_results)





    return model, y_pred_proba, y_test, proba_df


def create_sequences(data, labels, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])  # Take the last `lookback` rows
        y.append(labels[i])  # Target is the current row
    return np.array(X), np.array(y)

def LSTM_model(df, train_cols, learning_rate=0.001, batch_size=128, epochs=300, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Preprocessing
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['date'], inplace=True)

    # Split data into train and test datasets
    train_datasets, test_datasets = crossvalidation.run_split_process(df)
    train_dataset = train_datasets[-1]
    test_dataset = test_datasets[-1]

    # Separate features and target
    feature_cols = train_cols
    target_col = 'label'
    n_components = len(train_cols)

    X_train = train_dataset[feature_cols]
    y_train = train_dataset[target_col]
    X_test = test_dataset[feature_cols]
    y_test = test_dataset[target_col]

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Class weighting
    y_train_int = y_train + 1
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_int),
        y=y_train_int
    )
    class_weights_dict = dict(enumerate(class_weights))

    # One-hot encode targets
    y_train = to_categorical(y_train + 1, num_classes=3)
    y_test = to_categorical(y_test + 1, num_classes=3)

    # Create LSTM sequences
    lookback = 30
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, lookback)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, lookback)

    # Define LSTM model
    input_shape = (lookback, n_components)
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(3, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(class_id=0),
                 tf.keras.metrics.Precision(class_id=1),
                 tf.keras.metrics.Precision(class_id=2),
                 tf.keras.metrics.Recall(),
                 tf.keras.metrics.AUC()]
    )

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.001, patience=15, min_lr=1e-10)

    # Train model
    model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, lr_scheduler, tensorboard_callback],
        class_weight=class_weights_dict
    )

    # Evaluate model
    test_loss, test_acc, test_precision_dwn, test_precision_n, test_precision_up, test_recall, test_auc = model.evaluate(X_test_seq, y_test_seq)

    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_acc}')
    print(f'Test Precision Down: {test_precision_dwn}')
    print(f'Test Precision Neutral: {test_precision_n}')
    print(f'Test Precision Up: {test_precision_up}')
    print(f'Test Recall: {test_recall}')
    print(f'Test AUC: {test_auc}')

    # Make predictions
    y_pred_proba = model.predict(X_test_seq)
    log_loss_value = log_loss(y_test_seq, y_pred_proba)
    print(f'Log Loss: {log_loss_value}')

    # Prepare results DataFrame
    proba_df = pd.DataFrame(y_pred_proba, columns=['Proba_Class_0', 'Proba_Class_1', 'Proba_Class_2'])
    test_results = test_dataset.reset_index(drop=True)
    test_results = pd.concat([test_results.iloc[lookback:].reset_index(drop=True), proba_df], axis=1)
    test_results['True_Label'] = np.argmax(y_test_seq, axis=1)

    test_results = test_results[['label', 'Proba_Class_0', 'Proba_Class_1', 'Proba_Class_2', 'date']]
    test_results.to_csv(f'test_result_1.csv', index=False)
    print(test_results)

    return model, y_pred_proba, y_test_seq, proba_df



def neural_model2(df, train_cols, learning_rate=0.001, batch_size=128, epochs=300, seed=42):

    set_seed(seed)
   # print(train_cols)
    #df = df[train_cols]

    df.dropna(inplace=True)
    df.dropna(how='all', inplace=True)
    df.drop_duplicates(subset=['date'],inplace=True)

    train_datasets, test_datasets= crossvalidation.run_split_process(df)
    train_dataset =train_datasets[-1] 
    test_dataset = test_datasets[-1]

    ########feature and target seperation
    feature_cols =train_cols
    target_cols = 'label'
    n_components = len(train_cols)

    X_train =train_dataset[feature_cols]
    #lookback=30

   # for i in range(lookback, len(X_train)):

    
    y_train =train_dataset[target_cols]

    X_test = test_dataset[feature_cols]
    y_test = test_dataset[target_cols]
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)




    ####### Class weighting
    y_train_int = y_train + 1 
    
    class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_int),
    y=y_train_int
    )
# Convert to dictionary format:
    class_weights_dict = dict(enumerate(class_weights))

    ############ Fix One Hot Encoding 
    #y_train = y_train + 1  # Now classes become 0, 1, 2
    #y_test = y_test + 1

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(n_components,), kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(2, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.25, callbacks=[early_stopping, lr_scheduler])
    
    test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')
    print(f'Test precision: {test_precision}')
    print(f'Test recall: {test_recall}')
    print(f'Test AUC: {test_auc}')

    y_pred_proba = model.predict(X_test)
    log_loss_value = log_loss(y_test, y_pred_proba)
    print(f'Log Loss: {log_loss_value}')

    proba_df = pd.DataFrame(y_pred_proba, columns=['Proba_Class_0', 'Proba_Class_1'])


    test_results = test_dataset.reset_index(drop=True)
    test_results = pd.concat([test_results, proba_df], axis=1)
    test_results['True_Label'] = np.argmax(y_test, axis=1)

    test_results = test_results[['label','Proba_Class_0', 'Proba_Class_1', 'Proba_Class_2','formatted_time']]
    '''
    test_results['Close'] = startprice.reset_index(drop=True)
    test_results['touch_price'] = endprice.reset_index(drop=True)
    test_results['Date'] = Dates.reset_index(drop=True)
    test_results['endbarrier_unix'] = enddate.reset_index(drop=True)
    test_results['upper_barrier'] = upperbarrier.reset_index(drop=True)
    test_results['lower_barrier'] = lowerbarrier.reset_index(drop=True)
    test_results['touch_time_unix'] = touchtime.reset_index(drop=True)
    '''
    test_results.to_csv(f'test_result_1.csv')
    print(test_results)



   
  

   # joblib.dump(hmm_model, f'../deploy/models/EURUSD/{model_num}_hmm_model.pkl')

    return model, y_pred_proba, y_test, proba_df


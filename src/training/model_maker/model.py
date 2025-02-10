
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import log_loss
#import crossvalidation
import joblib
import random
#from hmmlearn.hmm import GaussianHMM



def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)





def neural_model(df, asset, lookback, n_components, training_cols, model_num, learning_rate=0.001, batch_size=128, epochs=300, seed=42):
    # Set seed at the start of the function to ensure consistency across runs
    set_seed(seed)
    
    if asset is not None:
        asset = asset

    startlookback = lookback * 10

    df.dropna(inplace=True)
    df.dropna(how='all', inplace=True)

    df = df[startlookback:]
    df['endbarrier_unix'] = pd.to_datetime(df['endbarrier_unix'], unit='s')
    prices = df[['Close', 'touch_price', 'Date', 'endbarrier_unix', 'upper_barrier', 'lower_barrier', 'pct','touch_time_unix']]
    df = df[training_cols]
    
 
    train_datasets, test_datasets = crossvalidation.run_split_process(df)
    feature_cols = df.drop('label', axis=1).columns
    target_col = 'label'

    n_components = 17
    scaler = StandardScaler()

    train_idx = train_datasets[-3]
    test_idx = test_datasets[-3]
    
    print('train index', train_idx)
    print('test idx', test_idx)

    train_data = df.iloc[train_idx]
    test_data = df.iloc[test_idx]
    
    pct_change = prices['pct'].iloc[test_idx]
    startprice = prices['Close'].iloc[test_idx]
    endprice = prices['touch_price'].iloc[test_idx]
    Dates = prices['Date'].iloc[test_idx]
    enddate = prices['endbarrier_unix'].iloc[test_idx]
    upperbarrier = prices['upper_barrier'].iloc[test_idx]
    lowerbarrier = prices['lower_barrier'].iloc[test_idx]
    touchtime = prices['touch_time_unix'].iloc[test_idx]

    # Get the feature columns (including HMM hidden states or probabilities)
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]

    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
    
    # Define your neural network model (same as before)
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

    test_results = test_data.reset_index(drop=True)
    test_results = pd.concat([test_results, proba_df], axis=1)
    test_results['True_Label'] = np.argmax(y_test, axis=1)

    test_results['Close'] = startprice.reset_index(drop=True)
    test_results['touch_price'] = endprice.reset_index(drop=True)
    test_results['Date'] = Dates.reset_index(drop=True)
    test_results['endbarrier_unix'] = enddate.reset_index(drop=True)
    test_results['upper_barrier'] = upperbarrier.reset_index(drop=True)
    test_results['lower_barrier'] = lowerbarrier.reset_index(drop=True)
    test_results['touch_time_unix'] = touchtime.reset_index(drop=True)

    test_results.to_csv(f'test_result_{model_num}.csv')
    print(test_results)

    joblib.dump(scaler, f'../deploy/models/EURUSD/{model_num}_scaler.pkl')
    model.save(f'../deploy/models/EURUSD/{model_num}.h5')
    # Save the PCA
    joblib.dump(pca, f'../deploy/models/EURUSD/{model_num}_pca.pkl')

   # joblib.dump(hmm_model, f'../deploy/models/EURUSD/{model_num}_hmm_model.pkl')

    return model, y_pred_proba, y_test, test_results


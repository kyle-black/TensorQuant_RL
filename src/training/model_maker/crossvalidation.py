from sklearn.model_selection import TimeSeriesSplit

def time_series_split(data, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, test_index in tscv.split(data):
        yield train_index, test_index


def purged_walk_forward_split_with_embargo(data, initial_train_size, test_size, gap=5, embargo_size=3):
    """
    ... [previous docstring content] ...
    """
    start_train = 0
    start_test = initial_train_size
    
    data_length = len(data)  # It's efficient to calculate the length once beforehand.

    while start_test + test_size + embargo_size <= data_length:
        train_indices = list(range(start_train, start_test))
        test_indices = list(range(start_test + gap, start_test + gap + test_size))
        
        # Clipping indices to ensure they are within valid range.
        train_indices = [index for index in train_indices if index < data_length]
        test_indices = [index for index in test_indices if index < data_length]

        # Check if there are enough indices to proceed. If not, you might want to break the loop or handle it accordingly.
        if not train_indices or not test_indices:
            print("Not enough data points for a valid train/test split at this step.")
            # You can choose to break, continue, or handle this situation in another appropriate manner.
            break

        yield train_indices, test_indices
        
        start_test += test_size + embargo_size

        # If the new start point for the test set is out of bounds, you might want to break the loop here too.
        if start_test + test_size + embargo_size > data_length:
            print("Reached the end of data; can't create more splits.")
            break

'''
def purged_walk_forward_split_with_embargo(data, initial_train_size, test_size, gap=5, embargo_size=3):
    """
    Parameters:
    data: your time series data
    initial_train_size: initial size of the training set
    test_size: size of the test set for each fold
    gap: number of timestamps to "purge" or leave out between train and test sets
    embargo_size: number of timestamps after test set to leave out before next train set begins
    """
    start_train = 0
    start_test = initial_train_size
    
    while start_test + test_size + embargo_size <= len(data):
        train_indices = list(range(start_train, start_test))
        
        # Ensure there's a gap and then an embargo after the test set
        test_indices = list(range(start_test + gap, start_test + gap + test_size))
        
        yield train_indices, test_indices
        
        start_test += test_size + embargo_size

'''
def time_decay_weights(data, decay_factor=0.95):
    """
    Parameters:
    data: your time series data
    decay_factor: decay factor between 0 and 1
    """
    n = len(data)
    return [decay_factor ** (n - i) for i in range(n)]



def run_split_process(data):
    
    data = data.reset_index(drop=True)  # resetting the index without adding 'index' as a column
    train_datasets = []
    test_datasets = []
    
    for train, test in purged_walk_forward_split_with_embargo(data, initial_train_size=20, test_size=1000, gap=5, embargo_size=3):
        train_data = data.iloc[train]
        test_data = data.iloc[test]

        train_datasets.append(train_data)
        test_datasets.append(test_data)

        # If you want to use time-decay weights
        weights = time_decay_weights(train_data, decay_factor=0.95)


    
    return train_datasets, test_datasets

'''
def run_split_process(data_list):
    
    all_train_datasets = []
    all_test_datasets = []

    #return data_list
    
    
    for data in data_list:
        data = data.reset_index(drop=True)  # resetting the index without adding 'index' as a column
        train_datasets = []
        test_datasets = []

        for train, test in purged_walk_forward_split_with_embargo(data, initial_train_size=20, test_size=10, gap=5, embargo_size=3):
            train_data = data.iloc[train]
            test_data = data.iloc[test]

            train_datasets.append(train_data)
            test_datasets.append(test_data)

        all_train_datasets.append(train_datasets)
        all_test_datasets.append(test_datasets)

    return all_train_datasets, all_test_datasets
'''
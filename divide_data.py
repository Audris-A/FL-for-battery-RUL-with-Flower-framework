# Divide only the training data set equally between all client setups for each configuration

import numpy as np

train_x = np.load("all_processed_data/full_train_x.npy")
train_y = np.load("all_processed_data/full_train_y.npy",)
# test_x = np.load("all_processed_data/full_test_x.npy")
# test_y = np.load("all_processed_data/full_test_y.npy")
# val_x = np.load("all_processed_data/full_val_x.npy")
# val_y = np.load("all_processed_data/full_val_y.npy")

client_limit = 12
for setup_it in range(2, client_limit+1):
    print("creating setup=", setup_it, "data")
    folder_path = "test_setup_data/" + str(setup_it) + "_client_setup/"
    
    split_train_x = np.array_split(train_x, setup_it)
    split_train_y = np.array_split(train_y, setup_it)
    # split_test_x = np.array_split(test_x, setup_it)
    # split_test_y = np.array_split(test_y, setup_it)
    # split_val_x = np.array_split(val_x, setup_it)
    # split_val_y = np.array_split(val_y, setup_it)
    
    # print(train_x.shape)
    # print(train_y.shape)
    # print(len(split_train_x))
    # print(len(split_train_y))
    # print(split_train_x[0].shape)
    # print(split_train_y[0].shape)
    # print("==============")

    for client_it in range(setup_it):
        print("creating client=", client_it, "data")

        np.save(folder_path + "x_data/client_" + str(client_it) + "_x", split_train_x[client_it], False, False)
        np.save(folder_path + "y_data/client_" + str(client_it) + "_y", split_train_y[client_it], False, False)

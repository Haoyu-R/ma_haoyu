import os

if __name__ == "__main__":

    path = r'..\preprocessed_data\processed_data\csv\second\0_cv_ego.csv'
    # Walk through every ego data


    # Depend on number of features and length of each example
    window_size = 500

    columns_name = ['speed', 'acc_x', 'acc_y']

    trained_model = load_model(r'NN_data\model_all_2.h5')

    # Following visualize the label in on sample
    y_test = trained_model.predict(np.expand_dims(X[num, :, :], axis=0))

    X = np.concatenate([x for x in X_list], axis=0)
    Y = np.concatenate([y for y in Y_list], axis=0)

    X, mean, std = normalization(X)
    # Reshape X and Y to fit the input of NN
    X = np.reshape(X, (int(X.shape[0]/window_size), window_size, len(columns_name)))
    Y = np.reshape(Y, (int(Y.shape[0]/new_y_length), new_y_length, class_num))
    # Shuffle the X and Y
    p = np.random.permutation(X.shape[0])
    X = X[p, :, :]
    Y = Y[p, :, :]

    mean_std = np.reshape(np.concatenate((mean, std), axis=0), (2, len(columns_name)))

    np.save('{}\\X_without_steering_ang.npy'.format(path), X)
    np.save('{}\\Y_without_steering_ang.npy'.format(path), Y)
    np.save('{}\\mean_std_without_steering_ang.npy'.format(path), mean_std)
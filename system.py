from typing import List

import numpy as np

N_DIMENSIONS = 10


def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    This dummy implementation simply returns the empty square label ('.')
    for every input feature vector in the test data.

    Note, this produces a surprisingly high score because most squares are empty.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    # Implementing k-nn for the classifer, with k = 6 
    K = 6
    all_labels =[]
    for test_data in test:
        #Computing the Euclidean Distance
        distance = np.linalg.norm (train - test_data, axis = 1)

        #Finding indices of k - nearest neighbours
        nn = np.argsort(distance)[:K]

        #Getting corresponding labels and their counts
        labels = train_labels[nn]
        unique_label, count = np.unique(labels, return_counts=True)

        #Index of the label with the maximum count
        max_count_index = np.argmax(count)

        #Appending the predicted label for test feature vector
        all_labels.append(unique_label[max_count_index])

    return all_labels


# The functions below must all be provided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    #Returns the reduced dimensionality matrix after conducting PCA
    #Checks whether the mean and eigenvectors are available in the model 
    if "mean_value" in model :
        mean_value = np.mat(model["mean_value"])
        eigenvectors = np.mat(model["selected_eigenvectors"])
        mean_center_data = data - mean_value
        reduced_dimensionality_Mat = mean_center_data * eigenvectors

    #If mean and eigenvectors are not available in the model, uses PCA for dimensionality reduction 
    else:
        mean_value = np.mean(data, axis = 0)
        mean_center_data = data - mean_value
        
        #Calculating the covariance matrix
        covariance_matrix = np.cov(mean_center_data, rowvar=False)

        #Get the eigenvalues and eigenvectors from the covariance matrix and sort the eigenvectors in descending order
        eigenvalues, eigenvectors = np.linalg.eig(np.mat(covariance_matrix))
        sorted_eigenvalues_index = np.argsort(eigenvalues)[:-(N_DIMENSIONS + 1): -1]

        #Selecting the eigenvectors and performing dimensionality reduction
        selected_eigenvectors = eigenvectors[:,sorted_eigenvalues_index]
        reduced_dimensionality_Mat = mean_center_data * selected_eigenvectors

        #Stores the computed mean values and selected eigenvectors to the model for future use
        model["mean_value"] = mean_value.tolist()
        model["selected_eigenvectors"] = selected_eigenvectors.tolist()

    return reduced_dimensionality_Mat


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    # The design of this is entirely up to you.
    # Note, if you are using an instance based approach, e.g. a nearest neighbour,
    # then the model will need to store the dimensionally-reduced training data and labels.
    model = {}
    model["labels_train"] = labels_train.tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()

    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy code below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    return classify_squares(fvectors_test, model)

# Import python libraries 
import numpy as np
from typing import List, Tuple
from collections import Counter
import time, platform, psutil # measure system performance/information
import matplotlib.pyplot  as plt # for plotting results
import re # Regular expression library
from ucimlrepo import fetch_ucirepo # for fetching UCI datasets directly

def print_machine_info(): # This function prints system info like OS, CPU, and RAM
    print("\n--- Machine Information ---")
    print(f"Operating System: {platform.system()} {platform.release()}") 
    print(f"Processor: {platform.processor()}")
    print(f"CPU Count: {psutil.cpu_count(logical=False)}")
    print(f"Total RAM: {round(psutil.virtual_memory().total / (1024**3), 2)} GB")
    print("---------------------------\n")

# Function to calculate euclidean distance between two data points
# Each point is a NumPy array of feature values.
def distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.sqrt(np.sum((point1 - point2)**2)) # euclidean distance formula


#   Function to predict the label of a single test point using the K-Nearest Neighbors algorithm. We use K=1 
#   train_data: the dataset we are learning from
#   train_labels: the correct labels for our dataset
#   test_point: the point we want to classify
#   k_neighbors: how many neighbors to consider for voting, k=1

def predict_label(train_data: np.ndarray, train_labels: np.ndarray, test_point: np.ndarray, k_neighbors: int) -> int:
    distances = []

    # Calculate distance from the test_point to every training example
    for i in range(len(train_data)):
        dist = distance(test_point, train_data[i]) # get distance between test_point and training point
        distances.append((dist, train_labels[i]))  # store both the distance and its label

     # Sort the list of distances in increasing order
    distances.sort(key=lambda x: x[0]) # sort by the distance, first element of the tuple

    # Pick the labels of the k nearest neighbors
    neighbors = distances[:k_neighbors]  # select top-k closest points
    neighbor_labels = [neighbor[1] for neighbor in neighbors] # extract labels of selected points

    # Find the most common label among the neighbors
    label_counts = Counter(neighbor_labels) # count number of times each label appears
    predicted_label = label_counts.most_common(1)[0][0] # pick the label with the highest count

    # Return prediction
    return predicted_label

# Function to evaluates how well our model performs using Leave-One-Out Cross Validation
# It loops through the dataset, treating one point as the test set and the rest as training data each time
def leave_one_out_evaluation(data: np.ndarray, labels: np.ndarray, features_to_use: List[int], k: int) -> float:
    num_correct = 0 # number of correct predictions
    num_total = len(data) # total size of examples

    # We use every data point once as a test point
    for i in range(num_total):
        # Use only selected features for this evaluation
        test_point = data[i, features_to_use]  # hold out one point for testing
        train_data = np.delete(data, i, axis=0)[:, features_to_use]   # remove the test point from training data
        train_labels = np.delete(labels, i) # remove the label of the selected point for testing
        predicted = predict_label(train_data, train_labels, test_point, k) # call the predict label function to make a prediction

        if predicted == labels[i]: # check if prediction is correct, increment the count
            num_correct += 1

    accuracy = num_correct / num_total # calculate overall accuracy
    return accuracy


# This function is to plot our results using the matplotlib library

def plot_accuracy(accuracy_history: dict, title: str):
    # Maintain a list of accuracies and feature sets
    feature_sets = [list(f) for f in accuracy_history.keys()]
    accuracies = list(accuracy_history.values())

    labels = []
    # Create feature set labels to add to the X-axis
    for features in feature_sets:
        if not features:
            labels.append("{}")
        else:
            labels.append("{" + ",".join(map(str, [f + 1 for f in features])) + "}")

    # Define the axis labels and plot the bar graph
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, [acc * 100 for acc in accuracies], color='orange')
    for bar, acc in zip(bars, accuracies):
       yval = acc * 100
       plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', fontsize=9)
    plt.xlabel("Current Feature Set")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.xticks(rotation=45, ha="right", fontsize=6)
    plt.yticks(range(10, 110, 10))
    plt.tight_layout()
    plt.show()

# This function performs Forward Feature Selection using KNN with Leave-One-Out evaluation
# It starts with no features and keeps adding the best feature one at a time
def forward_feature_selection(data: np.ndarray, labels: np.ndarray, k: int) -> Tuple[List[int], float]:

    num_features = data.shape[1] # Total number of features in the dataset
    selected_features = []       # Keeps track of the features we've selected so far
    best_accuracy = 0.0          # Track the best accuracy seen so far
    best_feature_set = []        # Track the best feature set giving best_accuracy
    all_possible_features = list(range(num_features)) # Indexes of all features
    accuracy_history = {}                             # Stores accuracy for each feature subset for later plotting

    print("\n--------- Starting forward selection algorithm ----------\n")

    start_time = time.time() # start time

    accuracy_history[tuple()] = leave_one_out_evaluation(data, labels, [], k) # Accuracy with no features, random guessing
    print(f"Using feature(s) {{}} gives accuracy {accuracy_history[tuple()]:.1%}")

    # Run the loop until we have included all features in the set
    while len(selected_features) < num_features:
        best_current_accuracy = -1.0 # Best accuracy for this iteration
        feature_to_add = -1          # Feature to add next in the set

        # Try all remaining (not-yet-selected) features
        remaining_features = [f for f in all_possible_features if f not in selected_features]

        if not remaining_features:
            break

        # Try adding each remaining feature and compute accuracy
        for feature_index in remaining_features:
            current_features = selected_features + [feature_index] # Add feature at [feature_index]
            accuracy = leave_one_out_evaluation(data, labels, current_features, k) # Calculate accuracy
            print(f"Using feature(s) {{{', '.join(map(str, [f + 1 for f in current_features]))}}} gives accuracy {accuracy:.1%}")

            # Pick feature tentatively if this gives a better accuracy
            if accuracy > best_current_accuracy:
                best_current_accuracy = accuracy
                feature_to_add = feature_index

        # # If we found a good feature to add in this round
        if feature_to_add != -1:
            # Add that feature to the selected list
            selected_features.append(feature_to_add)

            # If the accuracy with this new feature set is better than any we've seen so far
            if best_current_accuracy > best_accuracy:
                best_accuracy = best_current_accuracy # Update the overall best accuracy
                best_feature_set = list(selected_features) # Update the overall best feature set

            accuracy_history[tuple(selected_features)] = best_current_accuracy 

            # Iteration complete. Print which feature set was the best in this round and its accuracy
            print(f"Feature set {{{', '.join(map(str, [f + 1 for f in selected_features]))}}} is best for this iteration with {best_current_accuracy:.1%}\n")
        else:
            break
    
    end_time = time.time()  # Stop timing
    run_time = end_time - start_time # Calculate run time for the search (end-start)

    # Return the results
    print("\n------------ Forward selection complete ---------------\n")
    return best_feature_set, best_accuracy, run_time, accuracy_history


# This function performs Backward Feature Elimination using KNN and Leave-One-Out evaluation
# It starts with all features and keeps removing the least useful one until we have only one left
def backward_feature_elimination(data: np.ndarray, labels: np.ndarray, k: int) -> Tuple[List[int], float]:

    num_features = data.shape[1]
    selected_features = list(range(num_features)) # Start with all features selected
    best_accuracy = leave_one_out_evaluation(data, labels, selected_features, k) # Initial accuracy with all features
    best_feature_set = list(selected_features) # Initial best feature set, includes all features
    accuracy_history = {} 

    print("\n------------ Starting backward elimination algorithm -------------\n")

    start_time = time.time() # start time

    # Record the initial accuracy with all features
    accuracy_history[tuple(selected_features)] = best_accuracy
    print(f"Initial accuracy with all {num_features} features. Feature set {{{', '.join(map(str, [f + 1 for f in selected_features]))}}} is {best_accuracy:.1%}\n")

    # Repeat until only one feature remains
    while len(selected_features) > 1:
        best_current_accuracy = -1.0
        feature_to_remove_index = -1

        # Removing each feature one at a time and see how accuracy changes
        for i in range(len(selected_features)):
            remaining_features = selected_features[:i] + selected_features[i+1:] # Remove feature i
            accuracy = leave_one_out_evaluation(data, labels, remaining_features, k) # Calculate accuracy
            print(f"Using feature(s) {{{', '.join(map(str, [f + 1 for f in remaining_features]))}}} gives accuracy {accuracy:.1%}")

            # Remove feature tentatively if this gives a better accuracy
            if accuracy > best_current_accuracy:
                best_current_accuracy = accuracy
                feature_to_remove_index = i

        # If removing a feature gave the best accuracy, we remove that feature
        if feature_to_remove_index != -1:
            removed_feature = selected_features.pop(feature_to_remove_index) # remove from list
            if best_current_accuracy > best_accuracy:
                best_accuracy = best_current_accuracy      # Update best accuracy 
                best_feature_set = list(selected_features) # Update best feature set
            accuracy_history[tuple(selected_features)] = best_current_accuracy

            if(selected_features): # Iteration complete. Print best feature set and accuracy for this iteration
                print(f"\nRemoved feature: {removed_feature + 1}.\nFeature set {{{', '.join(map(str, [f + 1 for f in selected_features]))}}} is best for this iteration with {best_current_accuracy:.1%}\n")
            else:
                print(f"\nRemoved feature: {removed_feature + 1}.\nFeature set {{{', '.join(map(str, [f + 1 for f in selected_features]))}}} gives default accuracy rate of {best_current_accuracy:.1%}\n")

        else:
            break

    end_time = time.time()  # Stop timing
    run_time = end_time - start_time # Calculate run time for the search

    print("\n------------ Backward elimination complete ------------\n")

    # Return results
    return best_feature_set, best_accuracy, run_time, accuracy_history

# Main driver code
if __name__ == "__main__":

    print("\n------- This is Kopal's Feature Selection Algorithm -------- ")

    print_machine_info() # Print system information

    # Enter the dataset name. Enter name of the test datasets, or leave blank to use the default dataset

    data_file = input("Enter the file name to test the algorithm. Leave blank to evaluate on the default UCI ML Wine Dataset: ") 
    # CS205_small_Data__22.txt, CS205_large_Data__1.txt

    # Process/Load the dataset based on the file selected
    if re.match(r'CS205.*', data_file):
        print(f"\nTesting the algorithm on sample dataset: {data_file}\n")
        dataset = np.loadtxt(data_file, delimiter=None)
        labels = dataset[:, 0].astype(int)
        features = dataset[:, 1:]
    else:
        print(f"\nEvaluating the algorithm on dataset: Wine\n")
        wine = fetch_ucirepo(id=109) 
        # data (as numpy arrays)
        features = wine.data.features.to_numpy()
        labels = wine.data.targets.to_numpy().flatten().astype(int)

    # Print dataset information
    print(f"The dataset has {features.shape[1]} features and {len(labels)} examples")

    # Choose forward/backward feature selection algorithm
    print("\nChoose one of the below search algorithms to run: ")
    print("1) Forward Selection - start with no features and add features")
    print("2) Backward Elimination - start with all features and remove features")

    choice = input("Enter your choice (1 or 2): ")

    # Run forward feature selection and print the results returned after completion, k =1
    if choice == '1':
        best_features, accuracy, run_time, accuracy_history = forward_feature_selection(features, labels, k=1)
        print("\n------- Forward Selection Best Feature Set: --------\n")
        print(f"Best features found: {{{', '.join(map(str, [f + 1 for f in best_features]))}}}")
        print(f"Accuracy: {accuracy:.1%}")
        print(f"Run Time: {run_time/60:.1f} minutes\n")
        plot_accuracy(accuracy_history, "Accuracy of Feature Subsets (Forward Selection)") # Plot the graphs
    
    # Run backward feature elimination and print the results returned after completion, k =1
    elif choice == '2':
        best_features, accuracy, run_time, accuracy_history = backward_feature_elimination(features, labels, k=1)
        print("\n------- Backward Elimination Best Feature Set: -------\n")
        print(f"Best features found: {{{', '.join(map(str, [f + 1 for f in best_features]))}}}")
        print(f"Accuracy: {accuracy:.1%}")
        print(f"Run Time: {run_time/60:.1f} minutes\n")
        plot_accuracy(accuracy_history, "Accuracy of Feature Subsets (Backward Elimination)") # Plot the graphs

    else:
      print(f"\n-------- Invalid Choice of Algorithm -------")
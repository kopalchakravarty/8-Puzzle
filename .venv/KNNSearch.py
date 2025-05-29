import numpy as np
from typing import List, Tuple
from itertools import combinations
from collections import Counter
import time, platform, psutil
import matplotlib.pyplot  as plt
import re
from ucimlrepo import fetch_ucirepo

def print_machine_info(): # Machine Information
    print("\n--- Machine Information ---")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    print(f"CPU Count: {psutil.cpu_count(logical=False)}")
    print(f"Total RAM: {round(psutil.virtual_memory().total / (1024**3), 2)} GB")
    print("---------------------------\n")

def distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.sqrt(np.sum((point1 - point2)**2))

def predict_label(train_data: np.ndarray, train_labels: np.ndarray, test_point: np.ndarray, k_neighbors: int) -> int:
    distances = []

    for i in range(len(train_data)):
        dist = distance(test_point, train_data[i])
        distances.append((dist, train_labels[i]))

    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k_neighbors]
    neighbor_labels = [neighbor[1] for neighbor in neighbors]

    # Find the most common label among the neighbors
    label_counts = Counter(neighbor_labels)
    predicted_label = label_counts.most_common(1)[0][0]
    return predicted_label

def leave_one_out_evaluation(data: np.ndarray, labels: np.ndarray, features_to_use: List[int], k: int) -> float:
    num_correct = 0
    num_total = len(data)

    for i in range(num_total):
        test_point = data[i, features_to_use]
        train_data = np.delete(data, i, axis=0)[:, features_to_use]
        train_labels = np.delete(labels, i)
        predicted = predict_label(train_data, train_labels, test_point, k)

        if predicted == labels[i]:
            num_correct += 1

    accuracy = num_correct / num_total
    return accuracy


def plot_accuracy(accuracy_history: dict, title: str):
    feature_sets = [list(f) for f in accuracy_history.keys()]
    accuracies = list(accuracy_history.values())

    labels = []
    for features in feature_sets:
        if not features:
            labels.append("{}")
        else:
            labels.append("{" + ",".join(map(str, [f + 1 for f in features])) + "}")

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, [acc * 100 for acc in accuracies], color='orange')
    for bar, acc in zip(bars, accuracies):
        yval = acc * 100
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', fontsize=9)
    plt.xlabel("Current Feature Set")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(range(10, 110, 10))
    plt.tight_layout()
    plt.show()

    
def forward_feature_selection(data: np.ndarray, labels: np.ndarray, k: int) -> Tuple[List[int], float]:
    num_features = data.shape[1]
    selected_features = []
    best_accuracy = 0.0
    best_feature_set = []
    all_possible_features = list(range(num_features))
    accuracy_history = {}

    print("\n--------- Starting forward selection algorithm ----------\n")

    start_time = time.time() # start time

    accuracy_history[tuple()] = leave_one_out_evaluation(data, labels, [], k) # Accuracy with no features
    print(f"Using feature(s) {{}} gives accuracy {accuracy_history[tuple()]:.1%}")

    while len(selected_features) < num_features:
        best_current_accuracy = -1.0
        feature_to_add = -1

        remaining_features = [f for f in all_possible_features if f not in selected_features]

        if not remaining_features:
            break

        for feature_index in remaining_features:
            current_features = selected_features + [feature_index]
            accuracy = leave_one_out_evaluation(data, labels, current_features, k)
            print(f"Using feature(s) {{{', '.join(map(str, [f + 1 for f in current_features]))}}} gives accuracy {accuracy:.1%}")

            if accuracy > best_current_accuracy:
                best_current_accuracy = accuracy
                feature_to_add = feature_index

        if feature_to_add != -1:
            selected_features.append(feature_to_add)
            if best_current_accuracy > best_accuracy:
                best_accuracy = best_current_accuracy
                best_feature_set = list(selected_features)
            accuracy_history[tuple(selected_features)] = best_current_accuracy
            print(f"Feature set {{{', '.join(map(str, [f + 1 for f in selected_features]))}}} is best for this iteration with {best_current_accuracy:.1%}\n")
        else:
            break
    
    end_time = time.time()  # Stop timing
    run_time = end_time - start_time

    print("\n------------ Forward selection complete ---------------\n")
    return best_feature_set, best_accuracy, run_time, accuracy_history

def backward_feature_elimination(data: np.ndarray, labels: np.ndarray, k: int) -> Tuple[List[int], float]:

    num_features = data.shape[1]
    selected_features = list(range(num_features))
    best_accuracy = leave_one_out_evaluation(data, labels, selected_features, k)
    best_feature_set = list(selected_features)
    accuracy_history = {}

    print("\n------------ Starting backward elimination algorithm -------------\n")

    start_time = time.time() # start time

    accuracy_history[tuple(selected_features)] = best_accuracy
    print(f"Initial accuracy with all {num_features} features. Feature set {{{', '.join(map(str, [f + 1 for f in selected_features]))}}} is {best_accuracy:.1%}\n")

    while len(selected_features) > 1:
        best_current_accuracy = -1.0
        feature_to_remove_index = -1

        for i in range(len(selected_features)):
            remaining_features = selected_features[:i] + selected_features[i+1:]
            accuracy = leave_one_out_evaluation(data, labels, remaining_features, k)
            print(f"Using feature(s) {{{', '.join(map(str, [f + 1 for f in remaining_features]))}}} gives accuracy {accuracy:.1%}")

            if accuracy > best_current_accuracy:
                best_current_accuracy = accuracy
                feature_to_remove_index = i

        if feature_to_remove_index != -1:
            removed_feature = selected_features.pop(feature_to_remove_index)
            if best_current_accuracy > best_accuracy:
                best_accuracy = best_current_accuracy
                best_feature_set = list(selected_features)
            accuracy_history[tuple(selected_features)] = best_current_accuracy
            print(f"\nRemoved feature: {removed_feature + 1}.\nFeature set {{{', '.join(map(str, [f + 1 for f in selected_features]))}}} is best for this iteration with {best_current_accuracy:.1%}\n")
        else:
            break

    end_time = time.time()  # Stop timing
    run_time = end_time - start_time

    print("\n------------ Backward elimination complete ------------\n")
    return best_feature_set, best_accuracy, run_time, accuracy_history

if __name__ == "__main__":

    print("\n------- This is Kopal's Feature Selection Algorithm -------- ")

    print_machine_info()

    data_file = input("Enter the file name to test the algorithm. Leave blank to evaluate on the default UCI ML Wine Dataset: ") 
    # CS205_small_Data__22.txt, CS205_large_Data__1.txt

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

    #k = 1
    print(f"The dataset has {features.shape[1]} features and {len(labels)} examples")

    print("\nChoose one of the below search algorithms to run: ")
    print("1) Forward Selection - start with no features and add features")
    print("2) Backward Elimination - start with all features and remove features")

    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        best_features, accuracy, run_time, accuracy_history = forward_feature_selection(features, labels, k=1)
        print("\n------- Forward Selection Best Feature Set: --------\n")
        print(f"Best features found: {{{', '.join(map(str, [f + 1 for f in best_features]))}}}")
        print(f"Accuracy: {accuracy:.1%}")
        print(f"Run Time: {run_time/60:.1f} minutes\n")
        plot_accuracy(accuracy_history, "Accuracy of Feature Subsets (Forward Selection)")

    elif choice == '2':
        best_features, accuracy, run_time, accuracy_history = backward_feature_elimination(features, labels, k=1)
        print("\n------- Backward Elimination Best Feature Set: -------\n")
        print(f"Best features found: {{{', '.join(map(str, [f + 1 for f in best_features]))}}}")
        print(f"Accuracy: {accuracy:.1%}")
        print(f"Run Time: {run_time/60:.1f} minutes\n")
        plot_accuracy(accuracy_history, "Accuracy of Feature Subsets (Backward Elimination)")

    else:
      print(f"\n-------- Invalid Choice of Algorithm -------")
import networks as n
import mnist_data as mnist
from matplotlib import pyplot
import os
import numpy as np

def show_wrong_predictions(network, X, y, keep_showing = False):
    data = mnist.denormalise_data(X)
    pyplot.clf()
    count = 0
    for i in range(len(X)):
        (prediction, confidence) = network.predict(X[i])
        if prediction != y[i]:
            ax = pyplot.subplot(330 + 1 + count)
            ax.set_title(f'A:{y[i]}, P:{prediction}, {(confidence*100):.2f}%')
            pyplot.imshow(data[i], cmap=pyplot.get_cmap('gray'))
            count += 1
        if count >= 9:
            pyplot.tight_layout()
            pyplot.autoscale()
            pyplot.show()
            if keep_showing:
                pyplot.clf()
                count = 0
            else:
                return
    pyplot.show()

def save_wrong_predictions(network, X, y, folder):
    data = mnist.denormalise_data(X)
    script_dir = os.path.dirname(__file__)
    pyplot.clf()
    count = 0
    for i in range(len(X)):
        (prediction, confidence) = network.predict(X[i])
        if prediction != y[i]:
            ax = pyplot.subplot(330 + 1 + count)
            ax.set_title(f'A:{y[i]}, P:{prediction}, {(confidence*100):.2f}%')
            pyplot.imshow(data[i], cmap=pyplot.get_cmap('gray'))
            count += 1
        if count >= 9:
            pyplot.tight_layout()
            pyplot.autoscale()
            dir = os.path.join(script_dir, folder+f'/{i}')
            pyplot.savefig(dir)
            pyplot.clf()
            count = 0
    dir = os.path.join(script_dir, folder+f'/final')
    pyplot.savefig(dir)
    pyplot.clf()

def confusion_matrix(network, X, y, num_classes=10):
    """
    Returns a confusion matrix of shape (num_classes, num_classes)
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for xi, yi in zip(X, y):
        pred = network.predict(xi)[0]
        true = np.argmax(yi)
        cm[true, pred] += 1

    return cm

def print_confusion_matrix(cm):
    print("Confusion Matrix:")
    print("Rows = True label, Columns = Predicted label\n")

    header = "    " + " ".join(f"{i:4d}" for i in range(cm.shape[0]))
    print(header)

    for i, row in enumerate(cm):
        print(f"{i:2d}: " + " ".join(f"{v:4d}" for v in row))

def per_class_accuracy(cm):
    accuracies = {}
    for i in range(len(cm)):
        total = np.sum(cm[i])
        correct = cm[i, i]
        accuracies[i] = correct / total if total > 0 else 0.0
    return accuracies

def most_common_confusions(cm, top_k=10):
    errors = []

    for true_label in range(len(cm)):
        for pred_label in range(len(cm)):
            if true_label != pred_label and cm[true_label, pred_label] > 0:
                errors.append((
                    cm[true_label, pred_label],
                    true_label,
                    pred_label
                ))

    errors.sort(reverse=True)
    return errors[:top_k]

def average_confusion_confidence(network, X,y):
    sum = 0
    count = 0
    for xi, true in zip(X,y):
        (predicted, confidence) = network.predict(xi)
        if predicted != true:
            sum += confidence
            count += 1
    return sum/count

def average_true_confidence(network, X,y):
    sum = 0
    count = 0
    for xi, true in zip(X,y):
        (predicted, confidence) = network.predict(xi)
        if predicted == true:
            sum += confidence
            count += 1
    return sum/count

file = 'ReLu_Softmax_3.json'
dataset_size = 10000 #MAX=10000
print('Initialising network...', end='\r')
network = n.N_Network([128,64], 28 * 28, 10, [n.relu, n.relu, n.softmax], [n.relu_prime, n.relu_prime])
print('Network initialised              ')

print('Normalising test data...', end='\r')
(test_data, test_labels) = mnist.normalised_test_data()
print('Data normalised          ')

print(f'Using dataset size: {dataset_size}')
(dataset_X, dataset_y) = (test_data[:dataset_size], test_labels[:dataset_size])

test_outputs = mnist.expected_outputs(dataset_y)
print('Dataset formatting complete')

import os.path
if os.path.exists(file):
    print(f'Loading network from {file}...', end='\r')
    if file[-4:] == '.txt':
        network.from_string(n.read_from_file(file))
    else:
        network.load_json(file)
    print(f'Network loaded from {file}                                     ')
else:
    print('No network found')

print('='*100)
print('Calculating cost...', end='\r')
print(f'Network average cost: {network.average_cost(dataset_X, test_outputs)[0]:.4f}           ')
print('Calculating accuracy...', end='\r')
print(f'Network accuracy: {network.accuracy(dataset_X, dataset_y) * 100:.3f}%       ')
print()
print('Calculating confidence when correct...', end='\r')
print(f'Confidence when correct: {average_true_confidence(network,dataset_X,dataset_y) * 100:.3f}%              ')
print('Calculating confusion confidence...', end='\r')
print(f'Confidence when wrong: {average_confusion_confidence(network,dataset_X,dataset_y) * 100:.3f}%              ')
print('Calulating confusion matrix...', end='\r')
cm = confusion_matrix(network, dataset_X, test_outputs)
print('                                 ')
print_confusion_matrix(cm)
print('\nPer digit accuracy:')
acc = per_class_accuracy(cm)
for digit, a in acc.items():
    print(f"Digit {digit}: {a*100:.3f}%                     ")
print('\nMost common confusions:')
errors = most_common_confusions(cm)
for count, true, pred in errors:
    print(f"{true} → {pred}: {count} times")
print('='*100)

#if input('Show examples of wrong predictions?') == 'y':
#    if input('Show all?') == 'y':
#        show_wrong_predictions(network, dataset_X, dataset_y, True)
#    else:
#        show_wrong_predictions(network, dataset_X, dataset_y)

#if input('Save images of wrong predictions?') == 'y':
#    save_wrong_predictions(network, dataset_X, dataset_y, 'wrong_predictions')
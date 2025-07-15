from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def plot_confusion(y_true, y_pred, labels, title, cmap):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap=cmap)
    plt.title(title)
    plt.show()

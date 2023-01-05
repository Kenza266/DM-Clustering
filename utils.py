from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score

def report(actual_labels, predicted_labels, X=None):
    classes = list(set(actual_labels + predicted_labels))
    epsilon = 1e-6

    tps = {}
    tns = {}
    fps = {}
    fns = {}

    for c in classes:
        tps[c] = 0
        tns[c] = 0
        fps[c] = 0
        fns[c] = 0

    for actual, predicted in zip(actual_labels, predicted_labels):
        for c in classes:
            if c == actual and c == predicted:
                tps[c] += 1
            elif c == actual and c != predicted:
                fns[c] += 1
            elif c != actual and c == predicted:
                fps[c] += 1
            elif c != actual and c != predicted:
                tns[c] += 1

    precision = {}
    recall = {}
    f1 = {}

    for c in classes:
        precision[c] = tps[c] / (tps[c] + fps[c])
        recall[c] = tps[c] / (tps[c] + fns[c])
        f1[c] = 2 * (precision[c] * recall[c]) / (precision[c] + recall[c] + epsilon)

    accuracy = sum(tps.values()) / (sum(tps.values()) + sum(fps.values()) + sum(fns.values()))

    out = "Classification Report\n"
    for c in classes:
        out += "Class {}\n".format(c)
        out += "Precision: {:.3f}\n".format(precision[c])
        out += "Recall: {:.3f}\n".format(recall[c])
        out += "F1-Score: {:.3f}\n\n".format(f1[c])
    out += "Accuracy: {:.3f}\n\n".format(accuracy)
    
    '''out += "Silhouette score: {:.3f}\n".format(silhouette_score(X, predicted_labels))
    out += "Adjusted rand score: {:.3f}\n\n".format(adjusted_rand_score(actual_labels, predicted_labels))'''
    return out 
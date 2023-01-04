def report(actual_labels, predicted_labels):
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

    print("Classification Report")
    for c in classes:
        print("Class {}".format(c))
        print("Precision: {:.3f}".format(precision[c]))
        print("Recall: {:.3f}".format(recall[c]))
        print("F1-Score: {:.3f}".format(f1[c]))
        print()
    print("Accuracy: {:.3f}".format(accuracy))
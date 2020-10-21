def metrics(true_positive, total_truth, predicted_positive, correct_total, total):
    n_way = len(
        true_positive)  # Retrieve n_way from the length of the variables. All 3 inputs should be the same length
    f1_flag = 0  # Flag for invalid F1 score
    precision = list(0. for i in range(n_way))
    recall = list(0. for i in range(n_way))
    class_f1 = list(0. for i in range(n_way))

    # Find class accuracy, precision and recall
    for j in range(n_way):
        if predicted_positive[j] != 0 and true_positive[j] != 0:  # Check if F1 score is valid
            precision[j] = true_positive[j] / predicted_positive[j]
            recall[j] = true_positive[j] / total_truth[j]  # Recall is the same as per class accuracy
            class_f1[j] = 2 * precision[j] * recall[j] / (precision[j] + recall[j])
        else:
            f1_flag = 1

    # Find Accuracy, Macro Accuracy and Macro F1 Score
    macro_acc_sum = 0
    f1_sum = 0
    for k in range(n_way):
        macro_acc_sum += recall[k]
        if f1_flag == 0:  # Check for invalid f1 score
            f1_sum += class_f1[k]

    accuracy = correct_total / total
    macro_accuracy = macro_acc_sum / n_way
    f1_score = f1_sum / n_way
    return accuracy, macro_accuracy, f1_score, class_f1

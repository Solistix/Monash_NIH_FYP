import torch.optim as optim
from biobertology import get_tokenizer

import sys
sys.path.append('..')
from shared.models import *
from shared.datasets import *


def train(model, train_loader, criterion, device, optimizer, freeze=False):
    # freeze accepts a list and represents the layers not to freeze
    model.train()

    # Freeze all layers except those indicated
    if freeze:
        for name, param in model.named_parameters():
            if name not in freeze:
                param.requires_grad = False

    train_loss = 0
    for step, (data_image, data_text, data_attention, data_labels) in enumerate(train_loader):
        image_inputs, labels = data_image.to(device), data_labels.to(device)
        text_inputs, attention_inputs = data_text.to(device), data_attention.to(device)
        optimizer.zero_grad()
        pred = model(image_inputs, text_inputs, attention_inputs)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()  # Running training loss

    return train_loss / (step + 1)


def test(model, test_loader, criterion, device, n_way):
    # An F1 Score of 0 indicates that it is invalid
    model.eval()
    true_positive = list(0. for i in range(n_way))  # Number of correctly predicted samples per class
    total_truth = list(0. for i in range(n_way))  # Number of ground truths per class
    predicted_positive = list(0. for i in range(n_way))  # Number of predicted samples per class
    precision = list(0. for i in range(n_way))
    recall = list(0. for i in range(n_way))
    class_f1 = list(0. for i in range(n_way))
    val_loss = 0
    correct_total = 0  # Total correctly predicted samples
    total = 0  # Total samples
    f1_flag = 0  # Flag for invalid F1 score
    with torch.no_grad():
        for step, (data_image, data_text, data_attention, data_labels) in enumerate(test_loader):
            image_inputs, labels = data_image.to(device), data_labels.to(device)
            text_inputs, attention_inputs = data_text.to(device), data_attention.to(device)
            pred = model(image_inputs, text_inputs, attention_inputs)
            loss = criterion(pred, labels)
            val_loss += loss.item()  # Running validation loss
            _, predicted = torch.max(pred, 1)
            correct = (predicted == labels).squeeze()  # Samples that are correctly predicted
            correct_total += (predicted == labels).sum().item()
            total += labels.size(0)

            for i in range(len(predicted)):
                label = labels[i]
                true_positive[label] += correct[i].item()
                total_truth[label] += 1
                predicted_positive[predicted[i].item()] += 1  # True Positive + False Positive

        # Find class accuracy, precision and recall
        for j in range(n_way):
            if (predicted_positive[j] != 0 and true_positive[j] != 0):  # Check if F1 score is valid
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

    return val_loss / (step + 1), accuracy, macro_accuracy, f1_score, class_f1


def main():
    # Variables
    num_epochs = 100
    num_workers = 12
    bs = 8
    n_way = 3
    k_shot = 20
    lr = 1e-5
    save_models = True  # Whether to save the trained models (Occurs every epoch)

    root_image = '../../../../scratch/rl80/mimic-cxr-jpg-2.0.0.physionet.org/files'
    root_text = '../../../../scratch/rl80/mimic-cxr-2.0.0.physionet.org'
    path_biobert = './'
    path_pretrained = '../results/basic/basic_36.pth'  # Pretrained image model
    path_splits = f'../splits/{k_shot}_shot.csv'
    path_results = f'../../results/{k_shot}shot_multi_modal.csv'  # Full path to save the CSV results
    path_models = f'../../models/multi_modal/{k_shot}_shot'  # Folder path to save the trained models to
    freeze = ['concat_linear.weight', 'concat_linear.bias']

    if not os.path.exists(path_models):
        os.makedirs(path_models)

    # Check for GPU device
    torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load in model
    model = MultiModalNet(n_way, path_biobert).to(device)
    pretrained_dict = torch.load(path_pretrained)

    # Convert image model to work with the multi modal model
    multi_dict = {}
    del pretrained_dict['linear.weight']  # Pretrained model is for 10-way, remove last layer for 3-way
    del pretrained_dict['linear.bias']
    for key, value in pretrained_dict.items():
        multi_dict[f'baseline.{key}'] = pretrained_dict[
            key]  # The model has 'baseline.' in front of every image model key
    model_dict = model.state_dict()
    model_dict.update(multi_dict)
    model.load_state_dict(model_dict)

    # Check if the layers to be unfrozen are in the model
    if type(freeze) != bool:
        check = all(item in model_dict for item in freeze)
        if check:
            print(f'The layers that will not be frozen are: {freeze}')
        else:
            raise Exception('Not all elements to stay unfrozen are in the model')

    # Load in dataset
    tokenizer = get_tokenizer()
    train_dataset = MimicCxrMulti(root_image, root_text, path_splits, tokenizer, mode='novel_train')
    test_dataset = MimicCxrMulti(root_image, root_text, path_splits, tokenizer, mode='novel_validate')

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)

    # Create Dataframe to export results to CSV
    df_results = pd.DataFrame(columns=['Epoch', 'Training Loss', 'Validation Loss', 'Accuracy', 'Macro Accuracy',
                                       'Macro-F1 Score'] + [str(x) + ' F1' for x in range(n_way)])

    # Training
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, device, optimizer)
        val_loss, acc, m_acc, macro_f1, class_f1 = test(model, test_loader, criterion, device, n_way)

        if (save_models):
            torch.save(model.state_dict(), os.path.join(path_models, f'multi_modal_{epoch + 1}.pth'))  # Save the model

        # Append and report results
        df_results.loc[epoch] = [epoch + 1, train_loss, val_loss, acc, m_acc, macro_f1] + class_f1

        print(
            f'[{epoch + 1}] t_loss: {train_loss:.5f} v_loss: {val_loss:.5f} val_acc: {acc:.5f} '
            f'val_m_acc: {m_acc:.5f} f1: {macro_f1:.5f}')

    df_results.to_csv(path_results, index=False)  # Export results to a CSV file


if __name__ == '__main__':
    main()

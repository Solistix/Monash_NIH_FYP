from basic_train import *


def main():
    # Set Training Parameters
    num_workers = 12
    bs = 64
    n_way = 3
    path_pretrained = '../results/basic/basic_36.pth'
    save_models = True  # Whether to save the trained models (Occurs every epoch)
    k_shot = 20  # Must have the generated split to match it

    path_splits = f'../splits/{k_shot}_shot.csv'  # Location of preprocessed splits
    path_results = f'../../results/{k_shot}shot_nc.csv'  # Full path to save the CSV results
    path_models = f'../../models/nc/{k_shot}_shot'  # Folder path to save the trained models to

    torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # Load in model
    model = CosineSimilarityNet(n_way).to(device)
    pretrained_dict = torch.load(path_pretrained)
    del pretrained_dict['linear.weight']  # Remove the last layer
    del pretrained_dict['linear.bias']
    # del pretrained_dict['cos_sim.weight']
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # Load in data
    train_dataset = MimicCxrJpg(root='../../../../scratch/rl80/mimic-cxr-jpg-2.0.0.physionet.org/files/',
                                csv_path=path_splits, mode='novel_train', resize=224)
    test_dataset = MimicCxrJpg(root='../../../../scratch/rl80/mimic-cxr-jpg-2.0.0.physionet.org/files/',
                               csv_path=path_splits, mode='novel_validate', resize=224)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)

    # Create Dataframe to export results to CSV
    df_results = pd.DataFrame(columns=['Validation Loss', 'Accuracy', 'Macro Accuracy',
                                       'Macro-F1 Score'] + [str(x) + ' F1' for x in range(n_way)])

    # Find Average Features
    label_features = [torch.FloatTensor([]).to(device) for i in range(n_way)]
    model.eval()
    fc_weight = torch.FloatTensor([]).to(device)  # Initialise weight for the nearest centroid, last layer weight
    with torch.no_grad():
        for step, (data_inputs, data_labels) in enumerate(train_loader):
            # Get Features
            inputs, labels = data_inputs.to(device), data_labels.to(device)
            _, features = model(inputs, extract_features=True)

            # Sort features into labels
            for i in range(features.size(0)):
                label = labels[i]
                label_features[label] = torch.cat((label_features[label], features[i][None]))

        # Create weight for the last layer
        for j in range(n_way):
            feature_avg = torch.mean(label_features[j], 0)
            fc_weight = torch.cat((fc_weight, feature_avg[None]), 0)

        # Apply weight to the model
        nc_dict = model.state_dict()
        nc_dict['cos_sim.weight'] = fc_weight
        model.load_state_dict(nc_dict)

    # Testing
    val_loss, acc, m_acc, macro_f1, class_f1 = test(model, test_loader, criterion, device, n_way)

    if (save_models):
        torch.save(model.state_dict(), os.path.join(path_models, f'nc.pth'))  # Save the model

    # Append and report results
    df_results.loc[0] = [val_loss, acc, m_acc, macro_f1] + class_f1
    print(f'v_loss: {val_loss:.5f} val_acc: {acc:.5f} val_m_acc: {m_acc:.5f} f1: {macro_f1:.5f}')

    df_results.to_csv(path_results, index=False)  # Export results to a CSV file


if __name__ == '__main__':
    main()

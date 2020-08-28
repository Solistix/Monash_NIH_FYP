from basic_train import *


def main():
    # Set Training Parameters
    num_epochs = 100
    num_workers = 12
    bs = 64
    n_way = 3
    path_pretrained = '../results/basic_cosine/basic_cosine_58.pth'
    save_models = True  # Whether to save the trained models (Occurs every epoch)
    freeze = ['cos_sim.weight']  # Freeze all layers except linear layers
    k_shot = 20  # Must have the generated split to match it

    path_splits = f'../splits/{k_shot}_shot.csv'  # Location of preprocessed splits
    path_results = f'../../results/{k_shot}shot_cosine.csv'  # Full path to save the CSV results
    path_models = f'../../models/cosine_similarity/{k_shot}_shot'  # Folder path to save the trained models to

    torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load in model
    model = CosineSimilarityNet(n_way).to(device)
    pretrained_dict = torch.load(path_pretrained)
    del pretrained_dict['cos_sim.weight']  # Remove the last layer
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Check if the layers to be unfrozen are in the model
    if type(freeze) != bool:
        check = all(item in model_dict for item in freeze)
        if check:
            print(f'The layers that will not be frozen are: {freeze}')
        else:
            raise Exception('Not all elements to stay unfrozen are in the model')

    # Load in data
    train_dataset = MimicCxrJpg(root='../../../../scratch/rl80/mimic-cxr-jpg-2.0.0.physionet.org/files/',
                                csv_path=path_splits, mode='novel_train', resize=224)
    test_dataset = MimicCxrJpg(root='../../../../scratch/rl80/mimic-cxr-jpg-2.0.0.physionet.org/files/',
                               csv_path=path_splits, mode='novel_validate', resize=224)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)

    # Create Dataframe to export results to CSV
    df_results = pd.DataFrame(columns=['Epoch', 'Training Loss', 'Validation Loss', 'Accuracy', 'Macro Accuracy',
                                       'Macro-F1 Score'] + [str(x) + ' F1' for x in range(n_way)])

    # Training Loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, device, optimizer, freeze=freeze)
        val_loss, acc, m_acc, macro_f1, class_f1 = test(model, test_loader, criterion, device, n_way)

        if (save_models):
            torch.save(model.state_dict(), os.path.join(path_models, f'cosine_{epoch + 1}.pth'))  # Save the model

        # Append and report results
        df_results.loc[epoch] = [epoch + 1, train_loss, val_loss, acc, m_acc, macro_f1] + class_f1
        print(
            f'[{epoch + 1}] t_loss: {train_loss:.5f} v_loss: {val_loss:.5f} val_acc: {acc:.5f} '
            f'val_m_acc: {m_acc:.5f} f1: {macro_f1:.5f}')

    df_results.to_csv(path_results, index=False)  # Export results to a CSV file


if __name__ == '__main__':
    main()

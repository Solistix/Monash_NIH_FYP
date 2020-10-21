from baseline_train import *


def main():
    # Set Training Parameters
    n_way = 3
    k_shot = 20
    k_query = 16
    num_episodes = 100
    num_epochs = 100
    num_workers = 12
    bs = 4
    lr = 1e-4
    root = '../../../../scratch/rl80/mimic-cxr-jpg-2.0.0.physionet.org/files'
    path_splits = '../splits/splits.csv'  # Location of preprocessed splits
    path_results = '../../results'  # Folder to save the CSV results
    path_pretrained = '../results/basic_cosine/basic_cosine_55.pth'
    freeze = ['cos_sim.weight']   # Freeze all layers except linear layers

    torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # Load in data
    dataset = NovelMimicCxrJpg(root, path_splits, n_way, k_shot, k_query, num_episodes)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers)

    # Create Dataframe to export results to CSV
    df_results = pd.DataFrame(columns=['Epoch', 'Training Loss', 'Validation Loss', 'Accuracy', 'Macro Accuracy',
                                       'Macro-F1 Score'] + [str(x) + ' F1' for x in range(n_way)])

    # Iterate through batched episodes. One episode is one experiment
    for step, (support_imgs, support_labels, query_imgs, query_labels) in enumerate(loader):
        # Convert Tensors to appropriate device
        batch_support_x, batch_support_y = support_imgs.to(device), support_labels.to(device)
        batch_query_x, batch_query_y = query_imgs.to(device), query_labels.to(device)

        # [num_batch, training_sz, channels, height, width] = support_x.size()
        # num_batch = num of episodes
        # training_sz = size of support or query set
        num_batch = batch_support_x.size(0) # Number of episodes in the batch

        # Break down the batch of episodes into single episodes
        for i in range(num_batch):
            # Load in model and reset weights every episode/experiment
            model = CosineSimilarityNet(n_way).to(device)
            pretrained_dict = torch.load(path_pretrained)
            del pretrained_dict['cos_sim.weight']  # Remove the last layer
            model_dict = model.state_dict()
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            # Reset optimizer with model parameters
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Break down the sets into individual episodes
            support_x, support_y = batch_support_x[i], batch_support_y[i]
            query_x, query_y = batch_query_x[i], batch_query_y[i]

            # Variables for best epoch per experiment
            best_score = 0
            df_best = pd.DataFrame(columns=['Epoch', 'Training Loss', 'Validation Loss', 'Accuracy', 'Macro Accuracy',
                                       'Macro-F1 Score'] + [str(x) + ' F1' for x in range(n_way)]) # Track best epoch
            # Training and testing for specified epochs
            for epoch in range(num_epochs):
                # Training
                train_loss = train(support_x, support_y, model, criterion, device, optimizer, freeze=freeze)

                # Testing
                val_loss, acc, m_acc, macro_f1, class_f1 = test(query_x, query_y, model, criterion, device, n_way)

                # Find best epoch
                score = 0.5*acc + 0.5*macro_f1
                if score > best_score:
                    best_score = score
                    df_best.loc[0] = [epoch + 1, train_loss, val_loss, acc, m_acc, macro_f1] + class_f1

            # Print the best results per experiment
            print(
                f'[{int(df_best.iloc[0,0])}] t_loss: {df_best.iloc[0,1]} v_loss: {df_best.iloc[0,2]} '
                f'val_acc: {df_best.iloc[0,3]} f1: {df_best.iloc[0,5]}')

            # Record the best epoch to be saved into a CSV
            df_results = df_results.append(df_best.loc[0], ignore_index=True)

    # Create results folder if it does not exist
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    # Export results to a CSV file
    df_results.to_csv(os.path.join(path_results, f'{k_shot}shot_cosine.csv'), index=False)


if __name__ == '__main__':
    main()

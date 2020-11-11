if __name__ == '__main__':
    from biobertology import get_biobert, get_tokenizer

    path = '../results'  # Where to download the pre-trained BioBERT weights to
    biobert = get_biobert(model_dir=path, download=True)  # Download weights
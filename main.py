import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import sys

from utilities import Utilities
from transformer import TransformerEncoder, TransformerDecoder, AlibiDecoder
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers

eval_interval = 50  # How often to evaluate train and test perplexity during training
max_iters = 500  # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set

## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15  # epochs for classifier training


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files:
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts


def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])),
                                               "constant", 0)
    labels = torch.stack(labels)
    return padded_sequences, labels


def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses = []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y)  # your model should be computing the cross entropy loss
        losses.append(loss.item())
        # total_loss += loss.item()
        if len(losses) >= eval_iters: break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity


def compute_perplexity2(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses = []
    loss_fn = nn.CrossEntropyLoss()
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)

        pred, _ = decoderLMmodel(X)
        pred = pred.view(-1, decoderLMmodel.vocab_size)
        Y = Y.view(-1)
        loss = loss_fn(pred, Y)
        losses.append(loss.item())
        # total_loss += loss.item()
        if len(losses) >= eval_iters: break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity


def main():
    global n_hidden, n_output
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts))  # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)

    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch)

    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    argument = sys.argv[1]

    if argument == "Part1":
        # for the classification  task, you will train for a fixed number of epochs like this:

        encoder_model = TransformerEncoder(vocab_size=tokenizer.vocab_size, n_embd=n_embd, n_head=n_head,
                                           n_hidden=n_hidden, n_layer=n_layer, block_size=block_size, n_output=3)
        encoder_model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(encoder_model.parameters(), lr=learning_rate)

        all_train_accuracy = []
        all_test_accuracy = []
        all_train_loss = []
        all_test_loss = []

        num_param = sum(p.numel() for p in encoder_model.parameters() if p.requires_grad)
        print("Number of paratemters: ", num_param)

        for epoch in range(epochs_CLS):
            size = len(train_CLS_dataset)
            num_batches = len(train_CLS_loader)
            encoder_model.train()
            train_loss, correct = 0, 0

            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)

                pred, _ = encoder_model(xb)
                loss = loss_fn(pred, yb)
                train_loss += loss.item()
                correct += (pred.argmax(1) == yb).type(torch.float).sum().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            average_train_loss = train_loss / num_batches
            train_accuracy = correct / size

            print(f"epoch: {epoch}, train loss: {average_train_loss}, train accuracy: {train_accuracy}")

            # Testing
            size = len(test_CLS_dataset)
            num_batches = len(test_CLS_loader)
            encoder_model.eval()
            test_loss, correct = 0, 0

            for xb, yb in test_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)

                pred, _ = encoder_model(xb)
                loss = loss_fn(pred, yb)
                test_loss += loss.item()
                correct += (pred.argmax(1) == yb).type(torch.float).sum().item()

            average_test_loss = test_loss / num_batches
            test_accuracy = correct / size

            print(f"epoch: {epoch}, test loss: {average_test_loss}, test accuracy: {test_accuracy}")

            all_train_accuracy.append(train_accuracy)
            all_test_accuracy.append(test_accuracy)
            all_train_loss.append(average_train_loss)
            all_test_loss.append(average_test_loss)

        # Create a figure with two subplots
        plt.figure(figsize=(12, 5))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(all_train_accuracy, label='Train Accuracy', marker='o')
        plt.plot(all_test_accuracy, label='Test Accuracy', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Train and Test Accuracy')
        plt.legend()

        for i, v in enumerate(all_train_accuracy):
            plt.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8)

        for i, v in enumerate(all_test_accuracy):
            plt.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8)

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(all_train_loss, label='Train Loss', marker='o')
        plt.plot(all_test_loss, label='Test Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Test Loss')
        plt.legend()

        for i, v in enumerate(all_train_loss):
            plt.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8)

        for i, v in enumerate(all_test_loss):
            plt.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8)

        # Adjust layout and show the plots
        plt.show()
        plt.savefig("Part1.png")

        helper = Utilities(tokenizer, encoder_model)
        helper.sanity_check("In fact, I will be right there with you, as a citizen, for all my remaining days.",
                            block_size, "Encoder")

    if argument == "Part2":
        # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:

        n_hidden = 100
        n_output = tokenizer.vocab_size
        decoder_model = TransformerDecoder(vocab_size=tokenizer.vocab_size, n_embd=n_embd, n_head=n_head,
                                           n_hidden=n_hidden, n_layer=n_layer, block_size=block_size,
                                           n_output=n_output)
        decoder_model.to(device)

        num_param = sum(p.numel() for p in decoder_model.parameters() if p.requires_grad)
        print("Number of paratemters: ", num_param)

        loss_fn = nn.CrossEntropyLoss()
        lr = 5e-4
        optimizer = torch.optim.Adam(decoder_model.parameters(), lr=lr)

        inputfile = "speechesdataset/test_LM_hbush.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtestText = f.read()
        test_LM_dataset = LanguageModelingDataset(tokenizer, lmtestText, block_size)
        test_hbush_loader = DataLoader(test_LM_dataset, batch_size=batch_size, shuffle=True)

        inputfile = "speechesdataset/test_LM_obama.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtestText = f.read()
        test_LM_dataset = LanguageModelingDataset(tokenizer, lmtestText, block_size)
        test_obama_loader = DataLoader(test_LM_dataset, batch_size=batch_size, shuffle=True)

        inputfile = "speechesdataset/test_LM_wbush.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtestText = f.read()
        test_LM_dataset = LanguageModelingDataset(tokenizer, lmtestText, block_size)
        test_wbush_loader = DataLoader(test_LM_dataset, batch_size=batch_size, shuffle=True)

        train_list = []
        Hbush_list = []
        Obama_list = []
        Wbush_list = []

        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break

            if (i + 1) % eval_interval == 0:
                perplexity = compute_perplexity2(decoder_model, train_LM_loader)
                print(f"Training perplexity after {i + 1} iterations: {perplexity}")
                train_list.append(perplexity)

                perplexity = compute_perplexity2(decoder_model, test_hbush_loader)
                print(f"Hbush after {i + 1} iterations: {perplexity}")
                Hbush_list.append(perplexity)
                perplexity = compute_perplexity2(decoder_model, test_obama_loader)
                print(f"Obama after {i + 1} iterations: {perplexity}")
                Obama_list.append(perplexity)
                perplexity = compute_perplexity2(decoder_model, test_wbush_loader)
                print(f"TWbush after {i + 1} iterations: {perplexity}")
                Wbush_list.append(perplexity)



            xb, yb = xb.to(device), yb.to(device)

            decoder_model.train()

            pred, _ = decoder_model(xb)
            pred = pred.view(-1, tokenizer.vocab_size)
            yb = yb.view(-1)

            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        x_values = list(range(50, 550, 50))

        # Create a figure with two subplots side by side
        plt.figure(figsize=(12, 5))

        # Plot train perplexity
        plt.subplot(1, 2, 1)
        plt.plot(x_values, train_list, label='Train Perplexity', marker='o', color='b')
        plt.xlabel('Iteration')
        plt.ylabel('Perplexity')
        plt.title('Train Perplexity')
        plt.legend()

        for i, v in enumerate(train_list):
            plt.text(x_values[i], v, str(int(v)), ha='center', va='bottom', fontsize=8)

        # Plot test perplexities for Hbush, Obama, and Wbush
        plt.subplot(1, 2, 2)
        plt.plot(x_values, Hbush_list, label='H. Bush Perplexity', marker='o', color='r')
        plt.plot(x_values, Obama_list, label='Obama Perplexity', marker='o', color='g')
        plt.plot(x_values, Wbush_list, label='W. Bush Perplexity', marker='o', color='c')
        plt.xlabel('Iteration')
        plt.ylabel('Perplexity')
        plt.title('Test Perplexity (H. Bush, Obama, W. Bush)')
        plt.legend()

        for i, v in enumerate(Hbush_list):
            plt.text(x_values[i], v, str(int(v)), ha='center', va='bottom', fontsize=8, color='r')
        for i, v in enumerate(Obama_list):
            plt.text(x_values[i], v, str(int(v)), ha='center', va='bottom', fontsize=8, color='g')
        for i, v in enumerate(Wbush_list):
            plt.text(x_values[i], v, str(int(v)), ha='center', va='bottom', fontsize=8, color='c')

        # Adjust layout and show the plots
        plt.tight_layout()
        plt.show()
        plt.savefig("Part2.png")

        helper = Utilities(tokenizer, decoder_model)
        helper.sanity_check("In fact, I will be right there with you, as a citizen, for all my remaining days.",
                            block_size, "Decoder")

    if argument == "Part3":
        # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:

        n_hidden = 100
        n_output = tokenizer.vocab_size
        decoder_model = AlibiDecoder(vocab_size=tokenizer.vocab_size, n_embd=n_embd, n_head=n_head,
                                     n_hidden=n_hidden, n_layer=n_layer, block_size=block_size,
                                     n_output=n_output)
        decoder_model.to(device)

        num_param = sum(p.numel() for p in decoder_model.parameters() if p.requires_grad)
        print("Number of paratemters: ", num_param)


        loss_fn = nn.CrossEntropyLoss()
        lr = 5e-4
        optimizer = torch.optim.Adam(decoder_model.parameters(), lr=lr)

        inputfile = "speechesdataset/test_LM_hbush.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtestText = f.read()
        test_LM_dataset = LanguageModelingDataset(tokenizer, lmtestText, block_size)
        test_hbush_loader = DataLoader(test_LM_dataset, batch_size=batch_size, shuffle=True)

        inputfile = "speechesdataset/test_LM_obama.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtestText = f.read()
        test_LM_dataset = LanguageModelingDataset(tokenizer, lmtestText, block_size)
        test_obama_loader = DataLoader(test_LM_dataset, batch_size=batch_size, shuffle=True)

        inputfile = "speechesdataset/test_LM_wbush.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtestText = f.read()
        test_LM_dataset = LanguageModelingDataset(tokenizer, lmtestText, block_size)
        test_wbush_loader = DataLoader(test_LM_dataset, batch_size=batch_size, shuffle=True)

        train_list = []
        Hbush_list = []
        Obama_list = []
        Wbush_list = []

        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break

            if (i + 1) % eval_interval == 0:
                perplexity = compute_perplexity2(decoder_model, train_LM_loader)
                print(f"Training perplexity after {i + 1} iterations: {perplexity}")
                train_list.append(perplexity)

                perplexity = compute_perplexity2(decoder_model, test_hbush_loader)
                print(f"Hbush after {i + 1} iterations: {perplexity}")
                Hbush_list.append(perplexity)
                perplexity = compute_perplexity2(decoder_model, test_obama_loader)
                print(f"Obama after {i + 1} iterations: {perplexity}")
                Obama_list.append(perplexity)
                perplexity = compute_perplexity2(decoder_model, test_wbush_loader)
                print(f"TWbush after {i + 1} iterations: {perplexity}")
                Wbush_list.append(perplexity)

            xb, yb = xb.to(device), yb.to(device)

            decoder_model.train()

            pred, _ = decoder_model(xb)
            pred = pred.view(-1, tokenizer.vocab_size)
            yb = yb.view(-1)

            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        x_values = list(range(50, 550, 50))

        # Create a figure with two subplots side by side
        plt.figure(figsize=(12, 5))

        # Plot train perplexity
        plt.subplot(1, 2, 1)
        plt.plot(x_values, train_list, label='Train Perplexity', marker='o', color='b')
        plt.xlabel('Iteration')
        plt.ylabel('Perplexity')
        plt.title('Train Perplexity')
        plt.legend()

        for i, v in enumerate(train_list):
            plt.text(x_values[i], v, str(int(v)), ha='center', va='bottom', fontsize=8)

        # Plot test perplexities for Hbush, Obama, and Wbush
        plt.subplot(1, 2, 2)
        plt.plot(x_values, Hbush_list, label='H. Bush Perplexity', marker='o', color='r')
        plt.plot(x_values, Obama_list, label='Obama Perplexity', marker='o', color='g')
        plt.plot(x_values, Wbush_list, label='W. Bush Perplexity', marker='o', color='c')
        plt.xlabel('Iteration')
        plt.ylabel('Perplexity')
        plt.title('Test Perplexity (H. Bush, Obama, W. Bush)')
        plt.legend()

        for i, v in enumerate(Hbush_list):
            plt.text(x_values[i], v, str(int(v)), ha='center', va='bottom', fontsize=8, color='r')
        for i, v in enumerate(Obama_list):
            plt.text(x_values[i], v, str(int(v)), ha='center', va='bottom', fontsize=8, color='g')
        for i, v in enumerate(Wbush_list):
            plt.text(x_values[i], v, str(int(v)), ha='center', va='bottom', fontsize=8, color='c')

        # Adjust layout and show the plots
        plt.tight_layout()
        plt.show()
        plt.savefig("Part3.png")

        helper = Utilities(tokenizer, decoder_model)
        helper.sanity_check("In fact, I will be right there with you, as a citizen, for all my remaining days.",
                            block_size, "AlibiDecoder")


if __name__ == "__main__":
    main()

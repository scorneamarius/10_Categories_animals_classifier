import argparse
import utils
import os
import pandas as pd
import models
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch



def get_accuracy(outputs, labels):
    m = outputs.shape[0]
    correct_ans = torch.argmax(outputs)
    return float(torch.sum(correct_ans == labels).item()) / float(m)

def train(args):

    device = "cpu"
    if args.cuda == 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print(f"Training will be made on {device}")


    dataset_train_csv = os.path.join('dataset', 'train.csv')
    dataset_test_csv = os.path.join('dataset', 'test.csv')
    dataset_val_csv = os.path.join('dataset', 'val.csv')

    dataset_train = utils.get_dataset(dataset_train_csv)
    dataset_test = utils.get_dataset(dataset_test_csv)
    dataset_val = utils.get_dataset(dataset_val_csv)

    train_dataloader = utils.get_dataLoader(dataset_train, args.batch_size_train)
    test_dataloader = utils.get_dataLoader(dataset_test, args.batch_size_test)
    val_dataloader = utils.get_dataLoader(dataset_val, args.batch_size_val)

    model = models.ResNet50(10)
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in tqdm(range(args.epochs)):

        for batch_id_train, (inputs, labels) in enumerate(train_dataloader): 

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_train_acc = get_accuracy(outputs, labels)

            print(f"Epoch {epoch} | Batch {batch_id_train} train_accuracy = {batch_train_acc}")

        batch_val_accs = []
        iteration_number = 0

        for batch_id_val, (inputs_val, labels_val) in enumerate(val_dataloader):
            inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
            outputs_val = model(inputs_val)
            batch_val_acc = get_accuracy(outputs_val, labels_val)
            batch_val_accs.append(batch_val_acc)
            iteration_number += 1

            print(f"Epoch {epoch} | batch {batch_id_val} | val_accuracy = {batch_val_acc}")

        val_accuracy = sum(batch_val_accs) / iteration_number
        print(f"Epoch {epoch} | validation accuracy | TOTAL = {val_accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train model script")
    parser.add_argument("--batch_size_train", default = 4, type = int, help = "size of batch size for train")
    parser.add_argument("--batch_size_test", default = 4, type = int, help = "size of batch size for test")
    parser.add_argument("--batch_size_val", default = 4, type = int, help = "size of batch size for validation")
    parser.add_argument("--epochs", default = 10, type = int, help = "number of epochs for training dataset")
    parser.add_argument("--saveCheckPoints", default = 1, type = int, help = "save checkpoint for better validation accuracy", choices = [0,1])
    parser.add_argument("--cuda", default = 1, type = int, help = "Cuda ->0/1 -> OFF/ON")
    args = parser.parse_args()

    train(args)
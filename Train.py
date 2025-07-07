import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

from data_processing import spiltDataset_bert, dataPreprocess_bert_from_csv
from Model_MMA import Model


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (x1, x2, x3, ip, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training", unit="batch")):
        x1, x2, x3, ip, y = x1.to(device), x2.to(device), x3.to(device), ip.to(device), y.to(device)

        y_pred = model([x1, x2, x3], ip)
        loss = F.cross_entropy(y_pred, y.view(-1).long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(x1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def validation(model, device, test_loader, save_dir):
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x1, x2, x3, ip, y in tqdm(test_loader, desc="Validation", unit="batch"):
            x1, x2, x3, ip, y = x1.to(device), x2.to(device), x3.to(device), ip.to(device), y.to(device)
            y_ = model([x1, x2, x3], ip)
            test_loss += F.cross_entropy(y_, y.view(-1).long()).item()
            pred = y_.argmax(dim=-1)
            y_true.extend(y.view(-1).cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['benign', 'malicious'], yticklabels=['benign', 'malicious'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))

    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    return accuracy, precision, recall, f1


def run_training(data_limit, device):
    print(f"\n====== Training with {data_limit} samples ======")
    input_ids, input_types, input_masks, ip_embeds, label = [], [], [], [], []

    # 生成临时 CSV 文件
    full_path = "Data/aa/train.csv"
    tmp_path = f"Data/aa/train_{data_limit}.csv"

    # 截取前 N 条并保存为新文件
    df = pd.read_csv(full_path)
    df.iloc[:data_limit].to_csv(tmp_path, index=False)

    dataPreprocess_bert_from_csv(
        filename=tmp_path,
        input_ids=input_ids,
        input_types=input_types,
        input_masks=input_masks,
        ip_embeds=ip_embeds,
        label=label
    )

    # 划分训练验证集
    input_ids_train, input_types_train, input_masks_train, ip_embeds_train, y_train, \
    input_ids_val, input_types_val, input_masks_val, ip_embeds_val, y_val = spiltDataset_bert(
        input_ids, input_types, input_masks, ip_embeds, label
    )

    # 构建 Dataloader
    BATCH_SIZE = 16
    train_data = TensorDataset(
        torch.tensor(input_ids_train),
        torch.tensor(input_types_train),
        torch.tensor(input_masks_train),
        torch.tensor(ip_embeds_train, dtype=torch.float32),
        torch.tensor(y_train)
    )
    val_data = TensorDataset(
        torch.tensor(input_ids_val),
        torch.tensor(input_types_val),
        torch.tensor(input_masks_val),
        torch.tensor(ip_embeds_val, dtype=torch.float32),
        torch.tensor(y_val)
    )

    train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=BATCH_SIZE)


    model = Model(ip_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    NUM_EPOCHS = 10
    best_acc = 0.0
    save_dir = f'results/aa/{data_limit//10000}w'
    model_path = os.path.join(save_dir, 'model.pth')
    # checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')

    for epoch in range(1, NUM_EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)
        acc, precision, recall, f1 = validation(model, device, val_loader, save_dir)

        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)

        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'best_acc': best_acc
        # }, checkpoint_path)

        print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))

    # 保存结果
    with open(os.path.join(save_dir, 'result.txt'), 'w') as f:
        f.write(f'Best Accuracy: {best_acc:.4f}\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write(f'F1 Score: {f1:.4f}\n')


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for data_limit in [100000, 200000, 300000, 400000, 500000]:
        run_training(data_limit, DEVICE)


if __name__ == '__main__':
    main()

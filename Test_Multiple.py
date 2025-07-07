import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
from data_processing import dataPreprocess_bert_from_csv_multi
from Model_MMA_multi import Model


def test_multiclass(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", unit="batch"):
            x1, x2, x3, ip, y = [x.to(device, non_blocking=True) for x in batch]

            with torch.cuda.amp.autocast():  # 自动混合精度
                logits = model([x1, x2, x3], ip)
                loss = F.cross_entropy(logits, y.view(-1).long())
                test_loss += loss.item()

            pred = logits.argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    os.makedirs('results/multi/dd', exist_ok=True)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['mal', 'benign', 'phishing'],
                yticklabels=['mal', 'benign', 'phishing'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('results/multi/dd/confusion_matrix.png')
    plt.close()

    # ROC Curve and AUC
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = y_true_bin.shape[1]

    # 计算每个类别的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], np.array(y_probs)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), np.array(y_probs).ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 计算Macro-average ROC
    # 首先聚合所有假阳性率
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # 然后对每个类别的真阳性率进行插值
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # 最后平均
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average ROC curve (AUC = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='Macro-average ROC curve (AUC = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (AUC = {1:0.4f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC')
    plt.legend(loc="lower right")
    plt.savefig('results/multi/dd/roc_curve_multiclass.png')
    plt.close()

    # 保存测试结果
    np.savetxt('results/multi/dd/test_results.txt',
               np.column_stack((y_true, y_pred, np.array(y_probs))),
               fmt='%1.6f', delimiter='\t',
               header='True label\tPredicted label\tMal Prob\tBenign Prob\tPhishing Prob')

    # 保存标签、one-hot编码标签、概率和ROC数据
    np.savez(
        f"results/multi/dd/roc_data_transurl.npz",
        y_true=np.array(y_true),  # 原始标签 (n_samples,)
        y_true_onehot=y_true_bin,  # one-hot编码 (n_samples, n_classes)
        y_probs=np.array(y_probs),  # 概率矩阵 (n_samples, n_classes)
        fpr=fpr,  # 包含micro和macro的FPR字典
        tpr=tpr,  # 包含micro和macro的TPR字典
        roc_auc=roc_auc  # 各类别及Micro/Macro的AUC字典
    )

    class_names = {0: 'benign', 1: 'malicious', 2: 'phishing'}
    class_metrics = {}

    for class_id, class_name in class_names.items():
        # 计算每个类别的二分类指标（将当前类别视为正类，其他为负类）
        y_true_class = (np.array(y_true) == class_id).astype(int)
        y_pred_class = (np.array(y_pred) == class_id).astype(int)

        # Accuracy
        accuracy_class = accuracy_score(y_true_class, y_pred_class)

        # Precision, Recall, F1
        precision_class = precision_score(y_true_class, y_pred_class, zero_division=0)
        recall_class = recall_score(y_true_class, y_pred_class, zero_division=0)
        f1_class = f1_score(y_true_class, y_pred_class, zero_division=0)

        # AUC
        fpr_class, tpr_class, _ = roc_curve(y_true_class, np.array(y_probs)[:, class_id])
        auc_class = auc(fpr_class, tpr_class)

        class_metrics[class_id] = {
            'name': class_name,
            'accuracy': accuracy_class,
            'precision': precision_class,
            'recall': recall_class,
            'f1': f1_class,
            'auc': auc_class
        }

    print(
        'Test set: Avg loss: {:.4f}, Acc: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%, Macro AUC: {:.2f}%'.format(
            test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100, roc_auc["macro"] * 100))

    return {
        'global_metrics': {
            'test_loss': test_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'macro_auc': roc_auc["macro"],
            'micro_auc': roc_auc["micro"]
        },
        'class_metrics': class_metrics
    }


def main():
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    input_ids = []  # input char ids
    input_types = []  # segment ids
    input_masks = []  # attention mask
    label = []  # Labels
    ip_embeds = []  # IP Embeddings

    dataPreprocess_bert_from_csv_multi(
        filename="Data/dd/test.csv",
        input_ids=input_ids,
        input_types=input_types,
        input_masks=input_masks,
        ip_embeds=ip_embeds,
        label=label
    )

    # Load data into efficient DataLoaders
    BATCH_SIZE = 16
    test_data = TensorDataset(torch.tensor(input_ids).to(DEVICE),
                              torch.tensor(input_types).to(DEVICE),
                              torch.tensor(input_masks).to(DEVICE),
                              torch.tensor(ip_embeds, dtype=torch.float32).to(DEVICE),
                              torch.tensor(label).to(DEVICE))
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

    # Load the pre-trained model
    model = Model(ip_dim=128)
    print("Loading model weights from results/dd/model.pth ...")
    state_dict = torch.load("results/multi/dd/model.pth", map_location=DEVICE)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"[Warning] Missing keys in model:")
        for k in missing_keys:
            print(f"  - {k}")
    if unexpected_keys:
        print(f"[Warning] Unexpected keys in checkpoint:")
        for k in unexpected_keys:
            print(f"  - {k}")
    if missing_keys or unexpected_keys:
        print("\n[Note] State dict partially loaded. Consider retraining if needed.")

    model.to(DEVICE)

    # Test the model
    results = test_multiclass(model, DEVICE, test_loader)

    with open("results/multi/dd/final_results.txt", "w") as f:
        # 写入全局指标
        f.write("=== Global Metrics ===\n")
        f.write(f"Test Loss: {results['global_metrics']['test_loss']:.4f}\n")
        f.write(f"Accuracy: {results['global_metrics']['accuracy']:.4f}\n")
        f.write(f"Precision: {results['global_metrics']['precision']:.4f}\n")
        f.write(f"Recall: {results['global_metrics']['recall']:.4f}\n")
        f.write(f"F1 Score: {results['global_metrics']['f1']:.4f}\n")
        f.write(f"Macro AUC: {results['global_metrics']['macro_auc']:.4f}\n")
        f.write(f"Micro AUC: {results['global_metrics']['micro_auc']:.4f}\n\n")

        # 写入每个类别的指标
        f.write("=== Per-Class Metrics ===\n")
        for class_id, metrics in results['class_metrics'].items():
            f.write(f"\nClass {class_id} ({metrics['name']}):\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1-score: {metrics['f1']:.4f}\n")
            f.write(f"  AUC: {metrics['auc']:.4f}\n")


if __name__ == '__main__':
    main()
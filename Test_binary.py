import os
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_processing import dataPreprocess_bert_from_csv
from Model_MMA import Model

FPR_THRESHOLDS = [0.0001, 0.001, 0.01, 0.1]

def compute_tpr_at_fpr(fpr, tpr, thresholds):
    result = {}
    for target_fpr in thresholds:
        indices = np.where(fpr <= target_fpr)[0]
        if len(indices) == 0:
            result[target_fpr] = 0.0
        else:
            result[target_fpr] = float(tpr[indices[-1]])
    return result

def test_binary(model, device, test_loader, save_dir):
    model.eval()
    test_loss = 0.0
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", unit="batch"):
            x1, x2, x3, ip, y = [x.to(device, non_blocking=True) for x in batch]
            logits = model([x1, x2, x3], ip)
            loss = F.cross_entropy(logits, y.view(-1).long())
            test_loss += loss.item()

            pred = logits.argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_probs.extend(torch.softmax(logits, dim=1).cpu().numpy()[:, 1])

    test_loss /= len(test_loader)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ROC 曲线 + AUC + TPR@FPR
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    tpr_at_fpr = compute_tpr_at_fpr(fpr, tpr, FPR_THRESHOLDS)

    # 输出核心指标
    print(
        f'[{save_dir}] Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}')
    for k in FPR_THRESHOLDS:
        print(f'FPR: {k:.4g}, TPR: {tpr_at_fpr[k]:.6f}')

    # 保存 TPR@FPR 到当前模型目录
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "tpr_at_fpr.txt"), "w") as f:
        for k in FPR_THRESHOLDS:
            f.write(f"FPR {k:.4g}: TPR {tpr_at_fpr[k]:.6f}\n")

    # 保存结果
    results = {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'tpr_at_fpr': tpr_at_fpr
    }
    return results

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 加载固定测试集
    input_ids, input_types, input_masks, ip_embeds, label = [], [], [], [], []
    dataPreprocess_bert_from_csv(
        filename="Data/aa/test.csv",
        input_ids=input_ids,
        input_types=input_types,
        input_masks=input_masks,
        ip_embeds=ip_embeds,
        label=label
    )

    test_data = TensorDataset(
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(input_types, dtype=torch.long),
        torch.tensor(input_masks, dtype=torch.long),
        torch.tensor(ip_embeds, dtype=torch.float32),
        torch.tensor(label, dtype=torch.long)
    )

    test_loader = DataLoader(
        test_data,
        sampler=SequentialSampler(test_data),
        batch_size=16,
        pin_memory=True,
        num_workers=os.cpu_count()
    )

    # 记录不同训练数据集大小下的性能指标
    dataset_sizes = [10, 20, 30, 40, 50]
    results = {
        'precision': [],
        'recall': [],
        'f1': [],
        'accuracy': [],
        'auc': []
    }

    for size in dataset_sizes:
        model_dir = f'results/aa/{size}w'
        model_path = os.path.join(model_dir, 'model.pth')
        print(f'\n===> Testing model from {model_path}')

        model = Model(ip_dim=128)
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
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

        result = test_binary(model, DEVICE, test_loader, model_dir)

        results['precision'].append(result['precision'])
        results['recall'].append(result['recall'])
        results['f1'].append(result['f1'])
        results['accuracy'].append(result['accuracy'])
        results['auc'].append(result['auc'])

    # 绘制图表
    plt.figure(figsize=(8, 6))
    plt.plot(dataset_sizes, results['precision'], 'x-', label='Precision')
    plt.plot(dataset_sizes, results['recall'], '*-', label='Recall')
    plt.plot(dataset_sizes, results['f1'], 's-', label='F1')
    plt.plot(dataset_sizes, results['accuracy'], '+-', label='Accuracy')
    plt.plot(dataset_sizes, results['auc'], 'o-', label='AUC')
    plt.xlabel('Training dataset size')
    plt.ylabel('Score')
    plt.title('Model Evaluation Results')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_chart.png')
    plt.show()

    # 保存所有结果到 summary.txt
    os.makedirs("results/aa", exist_ok=True)
    summary_path = os.path.join("results/aa", "test_metrics_summary.txt")
    with open(summary_path, "w") as f:
        for idx, size in enumerate(dataset_sizes):
            f.write(f"=== Model trained on {size} samples ===\n")
            f.write(f"Precision: {results['precision'][idx]:.4f}\n")
            f.write(f"Recall:    {results['recall'][idx]:.4f}\n")
            f.write(f"F1 Score:  {results['f1'][idx]:.4f}\n")
            f.write(f"Accuracy:  {results['accuracy'][idx]:.4f}\n")
            f.write(f"AUC:       {results['auc'][idx]:.4f}\n")

            # 加载对应模型下的 tpr_at_fpr.txt 内容
            tpr_file = os.path.join(f"results/aa/{size}w", "tpr_at_fpr.txt")
            if os.path.exists(tpr_file):
                with open(tpr_file, "r") as tf:
                    f.write(tf.read())
            f.write("\n")

if __name__ == '__main__':
    main()

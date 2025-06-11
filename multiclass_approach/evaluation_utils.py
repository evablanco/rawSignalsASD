import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report


# Subject (sample) states (classes)
CALM, PRE_ATTACK, ATTACK = 0, 1, 2


def report_model(model, dataloader, device, class_names=None):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).long()
            outputs = model(batch_features)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted")
    print("=== Classification Report ===")
    print(classification_report(all_targets, all_preds, target_names=class_names if class_names else None))
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")


# default: f1-score en pre-attack class. class_ids=[0, 1, 2, 3] (calm, pre-attack, attack, post-attack)
def evaluate_f1_selected_classes(model, dataloader, device, class_ids=[PRE_ATTACK]):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).long()
            outputs = model(batch_features)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
    f1_selected = f1_score(all_targets, all_preds, labels=class_ids, average="macro", zero_division=0)
    return f1_selected


def evaluate_f1_score_model(model, dataloader, device, average="macro"):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).long()
            outputs = model(batch_features)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
    f1 = f1_score(all_targets, all_preds, average=average)
    return f1


def evaluate_auc_pre_attack(model, dataloader, device):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).long()
            outputs = model(batch_features)
            probs = F.softmax(outputs, dim=1)
            pre_probs = probs[:, 1]
            all_probs.extend(pre_probs.cpu().numpy())
            all_targets.extend((batch_labels == 1).cpu().numpy())
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.0
        print('roc_auc_score() ValueError...')
    return auc


def evaluate_macro_auc(model, dataloader, device, num_classes=4):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).long()
            outputs = model(batch_features)
            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
    try:
        auc = roc_auc_score(
            y_true=all_targets,
            y_score=all_probs,
            multi_class="ovr",  # one-vs-rest
            average="macro"
        )
    except ValueError:
        print('roc_auc_score() ValueError...')
        auc = 0.0  # si falta alguna clase...
    return auc


def compute_f1_per_class(model, dataloader, device, class_names=None):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).long()
            outputs = model(batch_features)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
    f1s = f1_score(all_targets, all_preds, average=None)
    class_ids = range(len(f1s))
    if class_names is None:
        return {f"Class_{i}": round(f1, 4) for i, f1 in zip(class_ids, f1s)}
    else:
        return {name: round(f1, 4) for name, f1 in zip(class_names, f1s)}


def model_evaluation(model, dataloader, device):
    print("*** Test Evaluation")
    report_model(model, dataloader, device, class_names=["Calm", "Pre-attack", "Attack"])
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).long()
            outputs = model(batch_features)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
    auc_macro = evaluate_macro_auc(model, dataloader, device)
    f1_macro = f1_score(all_targets, all_preds, average="macro")
    f1_weighted = f1_score(all_targets, all_preds, average="weighted")
    acc = accuracy_score(all_targets, all_preds)
    f1_per_class = compute_f1_per_class(model, dataloader, device,
                                        class_names=["Calm", "Pre-attack", "Attack"])
    f1_macro = round(f1_macro, 4)
    f1_weighted = round(f1_weighted, 4)
    auc_macro = round(auc_macro, 4)
    acc = round(acc, 4)
    f1_per_class = f1_per_class
    return f1_macro, f1_weighted, auc_macro, acc, f1_per_class



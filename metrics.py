def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    acc = float(corr) / float(tensor_size)

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT > threshold

    # TP : True Positive
    # FN : False Negative
    TP = torch.logical_and(SR, GT)
    FN = torch.logical_and((SR == 0), (GT == 1))

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT > threshold

    # TN : True Negative
    # FP : False Positive
    TN = torch.logical_and((SR == 0), (GT == 0))
    FP = torch.logical_and((SR == 1), (GT == 0))

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT > threshold

    # TP : True Positive
    # FP : False Positive
    TP = torch.logical_and(SR, GT)
    P = SR

    PC = float(torch.sum(TP)) / (float(torch.sum(P)) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT > threshold
    Inter = torch.sum(torch.logical_and(SR, GT))
    Union = torch.sum(torch.logical_or(SR, GT))

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT > threshold
    Inter = torch.sum(torch.logical_and(SR, GT)).item()
    Union = torch.sum(torch.logical_or(SR, GT)).item()
    DC = float(2 * Inter) / (float(Inter + Union) + 1e-6)

    return DC
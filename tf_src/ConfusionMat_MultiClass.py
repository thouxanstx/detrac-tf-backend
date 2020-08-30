import numpy as np

# This is the confusion matrix used for multiple classes
# Here we get all the false positives, true positives, false negatives and true negatives
# and assess the accuracy, sensitivity and specificity based on them.
def ConfusionMat_MultiClass(cmat, numClasses):
    ACC_Class = np.zeros(numClasses)
    SN_Class = np.zeros(numClasses)
    SP_Class = np.zeros(numClasses)

    for C in range(numClasses):
        TP, TN, FP, FN = 0, 0, 0, 0

        TP += cmat[C][C]

        for i in range(numClasses):
            if i == C:
                for j in range(numClasses):
                    if j != i:
                        FN += cmat[i][j]
                        FP += cmat[j][i]
            else:
                for j in range(numClasses):
                    if j != C:
                        TN += cmat[i][j]

        ACC_Class[C] = (TP + TN) / (TP + TN + FP + FN)
        SN_Class[C] = TP / (TP + FN)
        SP_Class[C] = TN / (TN + FP)

    return np.mean(ACC_Class), np.mean(SN_Class), np.mean(SP_Class)
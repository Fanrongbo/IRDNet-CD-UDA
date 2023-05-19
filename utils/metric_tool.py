"""
Copied and modified from
https://github.com/justchenhao/BIT_CD
"""
import numpy as np


###################       metrics      ###################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
            self.numT = 0
            self.a = 0
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def clear(self):
        self.initialized = False


###################      cm metrics      ###################
class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""
    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class
        self.numT=0
        self.a=0
    def update_cm(self, pr, gt, weight=1):
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score
    def confuseM(self,pr,presoft, gt ):

        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)

        current_score = cm2F1(val)
        valsoftmax = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=presoft)
        current_scoresoftmax = cm2F1(valsoftmax)
        self.numT=self.numT+val
        self.a=self.a+1
        # core_dict = {'accT': current_score['acc'], 'chgT': current_score['chgAcc'], 'unchgT': current_score['unchgAcc'],
        #              'mF1T':current_score['fm1'],'accs':current_scoresoftmax['acc'],'uchg':current_scoresoftmax['unchgAcc'],
        #              'chg':current_scoresoftmax['chgAcc'],'mf1s':current_scoresoftmax['fm1'],'s-cmf1':current_scoresoftmax['fm1']-current_score['fm1'],
        #              's-cchg':current_scoresoftmax['chgAcc']-current_score['chgAcc']}
        core_dict = {'accT': current_score['acc'], 'chgT': current_score['chgAcc'], 'unchgT': current_score['unchgAcc'],
                     'mF1T': current_score['fm1']}
        return core_dict
    def confuseMold(self,pr, gt ):

        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)

        current_score = cm2F1(val)
        # valsoftmax = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=presoft)
        # current_scoresoftmax = cm2F1(valsoftmax)
        self.numT=self.numT+val
        self.a=self.a+1
        core_dict = {'accT': current_score['acc'], 'chgT': current_score['chgAcc'], 'unchgT': current_score['unchgAcc'],
                     'mF1T':current_score['fm1']}
        return core_dict
    def get_scoresT(self):
        scores_dict = cm2F1(self.numT)
        core_dict = {'accT': scores_dict['acc'], 'chgT': scores_dict['chgAcc'],
                     'unchgT': scores_dict['unchgAcc'],'mF1T': scores_dict['fm1']}
        message='T:'
        for k, v in core_dict.items():
            message += '%s: %.3f ' % (k, v * 100)
        print(message)
        return message,core_dict
    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict



def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x+1e-6)**-1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    fn = hist.sum(axis=1) - np.diag(hist)
    fp = hist.sum(axis=0) - np.diag(hist)
    tn = hist.sum() - (fp + fn + tp)

    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)
    chgAcc = tp[1] / (tp[1] + fn[1]+1)
    unchgAcc = tn[1] / (tn[1] + fp[1]+1)

    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    score_dict = {'acc': acc, 'chgAcc': chgAcc, 'unchgAcc': unchgAcc,
                  'tp': int(tp[1]), 'fn': int(fn[1]), 'fp': int(fp[1]), 'tn': int(tn[1]),'fm1':mean_F1}
    return score_dict


def cm2score(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    fn = hist.sum(axis=1) - np.diag(hist)
    fp = hist.sum(axis=0) - np.diag(hist)
    tn = hist.sum() - (fp + fn + tp)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    chgAcc=tp[1]/(tp[1]+fn[1]+1)
    unchgAcc = tn[1] / (tn[1] + fp[1]+1)
    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2*recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    # ---------------------------------------------------------------------- #
    # 2. Frequency weighted Accuracy & Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    #
    cls_iou = dict(zip(['iou_'+str(i) for i in range(n_class)], iu))

    cls_precision = dict(zip(['precision_'+str(i) for i in range(n_class)], precision))
    cls_recall = dict(zip(['recall_'+str(i) for i in range(n_class)], recall))
    cls_F1 = dict(zip(['F1_'+str(i) for i in range(n_class)], F1))

    score_dict = {'acc': acc, 'miou': mean_iu, 'mf1':mean_F1, 'chgAcc': chgAcc, 'unchgAcc': unchgAcc,
                  'tp': int(tp[1]), 'fn': int(fn[1]), 'fp': int(fp[1]), 'tn': int(tn[1])}
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    return score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    def __fast_hist(label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes**2).reshape(num_classes, num_classes)
        return hist
    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix


def get_mIoU(num_classes, label_gts, label_preds):
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    score_dict = cm2score(confusion_matrix)
    return score_dict['miou']

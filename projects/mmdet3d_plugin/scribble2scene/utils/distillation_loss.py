import torch
import torch.nn as nn
import torch.nn.functional as F


# The proposed range-guided offline-to-online distillation (RGO2D)
def range_guided_distill_logit_loss(YS, YT, Label3d, Label_weight):
    # split semantic occupancy at short and middle ranges
    student_short = YS[:, :, 0:64:, 96:160, :]
    student_middle = YS[:, :, 0:128, 64:192, :]

    teacher_short = YT[:, :, 0:64:, 96:160, :]
    teacher_middle = YT[:, :, 0:128, 64:192, :]

    label_short = Label3d[:, 0:64:, 96:160, :]
    label_middle = Label3d[:, 0:128, 64:192, :]

    weight_short = Label_weight[:, 0:64:, 96:160, :]
    weight_middle = Label_weight[:, 0:128:, 64:192, :]

    # compute distillation losses at different ranges
    loss_short = distill_logit_loss(student_short, teacher_short, label_short, weight_short)
    loss_middle = distill_logit_loss(student_middle, teacher_middle, label_middle, weight_middle)
    loss_long = distill_logit_loss(YS, YT, Label3d, Label_weight)

    return loss_short, loss_middle, loss_long


def distill_logit_loss(YS, YT, Label3d, Label_weight):
    eps = 1e-8
    loss_global, loss_local = 0., 0.
    for y_s, y_t, label, weight in zip(YS, YT, Label3d, Label_weight):
        label = label.flatten(0)
        weight = weight.flatten(0)
        y_s, y_t = y_s.flatten(1).permute(1, 0)[weight], y_t.flatten(1).permute(1, 0)[weight]
        label = label[weight]

        # compute global semantic logits (GSL)
        unique_label = label.unique()
        unique_label = unique_label[unique_label != 0]
        mask = label[:, None] == unique_label[None, :]
        y_t_gsl = (y_t[:, None, :] * mask[:, :, None]).sum(0) / (mask.sum(0)[:, None] + eps)
        y_s_gsl = (y_s[:, None, :] * mask[:, :, None]).sum(0) / (mask.sum(0)[:, None] + eps)

        # compute inter- and intra-relation losses with global semantic logits (GSL)
        p_s = F.softmax(y_s_gsl, dim=1)
        p_t = F.softmax(y_t_gsl, dim=1)
        inter_loss = inter_class_relation(p_s, p_t)
        intra_loss = intra_class_relation(p_s, p_t)
        loss_global = loss_global + (inter_loss*2.625 + intra_loss*0.375) * 1.0
        # compute local semantic affinity (lsa) and corresponding local losses
        y_t_lsa = torch.cosine_similarity(y_t[:, None] + eps, y_t_gsl[None, :] + eps, dim=-1)
        y_s_lsa = torch.cosine_similarity(y_s[:, None] + eps, y_s_gsl[None, :] + eps, dim=-1)
        mseloss = nn.MSELoss()
        loss_local = loss_local + mseloss(y_t_lsa + eps, y_s_lsa + eps)
    loss = (loss_global + loss_local) / Label3d.shape[0]

    return loss


def cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)


def pearson_correlation(x, y, eps=1e-8):
    return cosine_similarity(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


# feature-level distillation with mse loss
def distill_feat(feat_s, feat_t):
    loss = torch.nn.functional.mse_loss(feat_s, feat_t, reduction='none') * 0.6
    return loss.mean()


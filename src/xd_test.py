import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import CLIPVAD
from utils.dataset import XDDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.xd_detectionMAP import getDetectionMAP as dmAP
import xd_option


def snippet_to_frame_scores(scores, loop_length=16):
    scores = np.asarray(scores, dtype=np.float32)
    return np.repeat(scores, loop_length)


def compute_binary_metrics(gt, scores):
    gt = np.asarray(gt, dtype=np.float32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    assert len(gt) == len(scores), (
        f"Prediction/GT length mismatch: pred={len(scores)}, gt={len(gt)}"
    )
    auc = float(roc_auc_score(gt, scores))
    ap = float(average_precision_score(gt, scores))
    return auc, ap


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
    model.to(device)
    model.eval()

    element_logits2_stack = []
    coarse_scores_all = []
    alignment_scores_all = []
    abnormal_video_scores = []
    abnormal_video_gt = []
    num_abnormal_videos = 0
    gt_offset = 0

    with torch.no_grad():
        for item in testdataloader:
            visual = item[0].squeeze(0)
            length = int(item[2])
            video_name = item[3][0]
            video_label = str(item[4][0])

            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])

            # C-branch: binary normal/abnormal anomaly score.
            coarse_scores = torch.sigmoid(logits1[0:len_cur].squeeze(-1)).detach().cpu().numpy()
            # A-branch: anomaly score derived from vision-language class probability.
            alignment_scores = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1)).detach().cpu().numpy()

            coarse_frame_scores = snippet_to_frame_scores(coarse_scores)
            alignment_frame_scores = snippet_to_frame_scores(alignment_scores)
            gt_video = gt[gt_offset:gt_offset + len(coarse_frame_scores)]
            if len(gt_video) != len(coarse_frame_scores):
                raise ValueError(
                    f"Prediction/GT length mismatch for video {video_name}: "
                    f"pred={len(coarse_frame_scores)}, gt={len(gt_video)}"
                )

            coarse_scores_all.extend(coarse_frame_scores.tolist())
            alignment_scores_all.extend(alignment_frame_scores.tolist())
            gt_offset += len(coarse_frame_scores)

            if video_label != 'A':
                num_abnormal_videos += 1
                abnormal_video_scores.extend(coarse_frame_scores.tolist())
                abnormal_video_gt.extend(gt_video.tolist())

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

    if gt_offset != len(gt):
        raise ValueError(
            f"Prediction/GT length mismatch at end: pred={gt_offset}, gt={len(gt)}"
        )

    coarse_auc, coarse_ap = compute_binary_metrics(gt, coarse_scores_all)
    alignment_auc, alignment_ap = compute_binary_metrics(gt, alignment_scores_all)

    if len(abnormal_video_gt) != len(abnormal_video_scores):
        raise ValueError(
            f"Abnormal-video score/GT length mismatch: scores={len(abnormal_video_scores)}, gt={len(abnormal_video_gt)}"
        )

    xd_ano_auc = None
    xd_ano_ap = None
    if len(abnormal_video_gt) > 0:
        abnormal_video_gt_arr = np.asarray(abnormal_video_gt, dtype=np.float32)
        abnormal_video_scores_arr = np.asarray(abnormal_video_scores, dtype=np.float32)
        unique_labels = np.unique(abnormal_video_gt_arr)
        if len(unique_labels) < 2:
            raise ValueError(
                f"Abnormal-video subset must contain both normal and abnormal frames before Ano-AUC; "
                f"found labels={unique_labels.tolist()} (videos={num_abnormal_videos})"
            )
        xd_ano_auc = float(roc_auc_score(abnormal_video_gt_arr, abnormal_video_scores_arr))
        xd_ano_ap = float(average_precision_score(abnormal_video_gt_arr, abnormal_video_scores_arr))

    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    average_map = float(np.mean(dmap))

    print("============================================================")
    print("XD-Violence Evaluation")
    print("============================================================")
    print("[Paper / checkpoint metric]")
    print(f"AP                             : {alignment_ap * 100:.2f}%")
    print("\n[Fine-grained anomaly detection]")
    for idx, threshold in enumerate(iou):
        print(f"mAP@{threshold:.1f}                        : {dmap[idx]:.2f}%")
    print(f"Average mAP                     : {average_map:.2f}%")

    print("\n[Additional research metrics]")
    if xd_ano_auc is not None:
        print(f"Ano-AUC (abnormal videos only)  : {xd_ano_auc * 100:.2f}%")
    if xd_ano_ap is not None:
        print(f"Ano-AP  (abnormal videos only)  : {xd_ano_ap * 100:.2f}%")

    print("\n[Auxiliary branch diagnostics]")
    print(f"C-branch AUC                   : {coarse_auc * 100:.2f}%")
    print(f"C-branch AP                    : {coarse_ap * 100:.2f}%")
    print(f"A-branch AUC                   : {alignment_auc * 100:.2f}%")
    print(f"A-branch AP                    : {alignment_ap * 100:.2f}%")
    print("============================================================")
    print(f"XD abnormal-only evaluation: abnormal_videos={num_abnormal_videos}, frames={len(abnormal_video_scores)}, abnormal_frames={sum(int(v) for v in abnormal_video_gt)}, normal_frames={sum(1 - int(v) for v in abnormal_video_gt)}")
    print("============================================================")

    return coarse_auc, alignment_ap, average_map

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
    model_param = torch.load(args.model_path)
    model.load_state_dict(model_param)

    test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
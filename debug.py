import os
import random

import numpy as np
import torch
from util.config import cfg
import time

import util.eval as eval
from checkpoint import align_and_update_state_dicts, strip_prefix_if_present
from datasets.scannetv2 import BENCHMARK_SEMANTIC_LABELS

from model.geoformer.geoformer_fs import GeoFormerFS
from datasets.scannetv2_fs_inst import FSInstDataset
from lib.pointgroup_ops.functions import pointgroup_ops
from util.log import create_logger
from util.utils_3d import load_ids, matrix_non_max_suppression


def init():
    os.makedirs(cfg.exp_path, exist_ok=True)

    global logger
    logger = create_logger(task="test")
    logger.info(cfg)

    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)

def do_test(model, dataset):
    result_root = 'test_results'
    os.makedirs(result_root, exist_ok=True)

    model.eval()
    net_device = next(model.parameters()).device

    logger.info(">>>>>>>>>>>>>>>> Start Inference >>>>>>>>>>>>>>>>")
    dataloader = dataset.testLoader()

    num_test_scenes = len(dataloader)

    zero_instance_types = {}

    with torch.no_grad():

        gt_file_arr = []
        test_scene_name_arr = []
        pred_info_arr = [[] for idx in range(cfg.run_num)]

        start_time = time.time()
        for i, batch_input in enumerate(dataloader):
            nclusters = [0] * cfg.run_num
            clusters = [[] for idx in range(cfg.run_num)]
            cluster_scores = [[] for idx in range(cfg.run_num)]
            cluster_semantic_id = [[] for idx in range(cfg.run_num)]
            is_valid, list_support_dicts, query_dict, scene_infos = batch_input
            if not is_valid:
                continue

            test_scene_name = scene_infos["query_scene"]
            active_label = scene_infos["active_label"]

            test_scene_name_arr.append(test_scene_name)
            gt_file_name = os.path.join(cfg.data_root, cfg.dataset, "val_gt", test_scene_name + ".txt")
            gt_file_arr.append(gt_file_name)

            N = query_dict["feats"].shape[0]    # query scene点的数量

            for key in query_dict:
                if torch.is_tensor(query_dict[key]):
                    query_dict[key] = query_dict[key].to(net_device)

            for j, (label, l_support_dicts) in enumerate(zip(active_label, list_support_dicts)):
                for k in range(cfg.run_num):  # NOTE number of runs
                    support_dict = l_support_dicts[k]
                    remember = False    # 全部的support_dict都要利用

                    for key in support_dict:
                        if torch.is_tensor(support_dict[key]):
                            support_dict[key] = support_dict[key].to(net_device)

                    outputs = model(
                        support_dict,
                        query_dict,
                        training=False,
                        remember=remember,
                        support_embeddings=None,
                    )

                    if outputs["no_fg"]:
                        continue
                    if outputs["proposal_scores"] is None:
                        continue
                    scores_pred, proposals_pred = outputs["proposal_scores"]
                    if isinstance(scores_pred, list):
                        continue

                    benchmark_label = BENCHMARK_SEMANTIC_LABELS[label]
                    cluster_semantic = torch.ones((proposals_pred.shape[0])).cuda() * benchmark_label

                    clusters[k].append(proposals_pred)
                    cluster_scores[k].append(scores_pred)
                    cluster_semantic_id[k].append(cluster_semantic)

                    # torch.cuda.empty_cache()

            for k in range(cfg.run_num):
                if len(clusters[k]) == 0:
                    pred_info_arr[k].append(None)
                    continue
                clusters[k] = torch.cat(clusters[k], axis=0)
                cluster_scores[k] = torch.cat(cluster_scores[k], axis=0)
                cluster_semantic_id[k] = torch.cat(cluster_semantic_id[k], axis=0)

                # nms
                if cluster_scores[k].shape[0] == 0:
                    pick_idxs_cluster = np.empty(0)
                else:
                    pick_idxs_cluster = matrix_non_max_suppression(
                        clusters[k].float(),
                        cluster_scores[k],
                        cluster_semantic_id[k],
                        final_score_thresh=0.5
                    )

                clusters[k] = clusters[k][pick_idxs_cluster].cpu().numpy()
                cluster_scores[k] = cluster_scores[k][pick_idxs_cluster].cpu().numpy()
                cluster_semantic_id[k] = cluster_semantic_id[k][pick_idxs_cluster].cpu().numpy()
                nclusters[k] = clusters[k].shape[0]

                if cfg.eval:
                    pred_info = {}
                    pred_info["conf"] = cluster_scores[k]
                    pred_info["label_id"] = cluster_semantic_id[k]
                    pred_info["mask"] = clusters[k]
                    pred_info_arr[k].append(pred_info)

            overlap_time = time.time() - start_time

            logger.info(
                f"Test scene {i+1}/{num_test_scenes}: {test_scene_name} | Elapsed time: {int(overlap_time)}s | Remaining time: {int(overlap_time * float(num_test_scenes-(i+1))/(i+1))}s"
            )
            logger.info(f"Num points: {N} | Num instances of {cfg.run_num} runs: {nclusters}")
            if sum(nclusters) == 0:
                print(active_label)
                for label in active_label:
                    if zero_instance_types.get(label) is None:
                        zero_instance_types[label] = 0
                    zero_instance_types[label] += 1
                logger.info("===================================================")
                logger.info(zero_instance_types)
                logger.info("===================================================")
            # if i == 1:
            #     break

        torch.save(pred_info_arr, os.path.join(result_root, 'pred_info_arr.pth'))

        # evaluation
        if cfg.eval:
            logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
            run_dict = {}

            matches_dir = os.path.join(result_root, 'matches')
            os.makedirs(matches_dir, exist_ok=True)

            for k in range(cfg.run_num):
                matches = {}
                for i in range(len(pred_info_arr[k])):  # 就是总共的测试集长度
                    pred_info = pred_info_arr[k][i]     # 第k轮种针对第i个query scene生成的所有prediction（可能对应多种label）
                    if pred_info is None:
                        continue

                    gt_file_name = gt_file_arr[i]
                    test_scene_name = test_scene_name_arr[i]
                    gt_ids = load_ids(gt_file_name)

                    gt2pred, pred2gt = eval.assign_instances_for_scan(test_scene_name, pred_info, gt_ids)
                    matches[test_scene_name] = {}
                    matches[test_scene_name]["gt"] = gt2pred
                    matches[test_scene_name]["pred"] = pred2gt

                ap_scores = eval.evaluate_matches(matches)
                avgs = eval.compute_averages(ap_scores)
                eval.print_results(avgs, logger)
                run_dict = eval.accumulate_averages_over_runs(run_dict, avgs)

                torch.save(matches, os.path.join(matches_dir, 'matches_{}.pth'.format(k)))

            run_dict = eval.compute_averages_over_runs(run_dict)
            eval.print_results(run_dict, logger)


if __name__ == "__main__":
    init()

    # model
    logger.info("=> creating model ...")

    model = GeoFormerFS()
    model = model.cuda()

    logger.info("# parameters (model): {}".format(sum([x.nelement() for x in model.parameters()])))

    checkpoint_fn = cfg.resume
    if os.path.isfile(checkpoint_fn):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_fn))
        state = torch.load(checkpoint_fn)
        model_state_dict = model.state_dict()
        loaded_state_dict = strip_prefix_if_present(state["state_dict"], prefix="module.")
        align_and_update_state_dicts(model_state_dict, loaded_state_dict)
        model.load_state_dict(model_state_dict)

        logger.info("=> loaded checkpoint '{}')".format(checkpoint_fn))
    else:
        raise RuntimeError

    dataset = FSInstDataset(split_set="val")

    # evaluate
    do_test(
        model,
        dataset,
    )
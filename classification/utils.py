import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import logging
import numpy as np
from dataset.imagenet_subsets import IMAGENET_D_MAPPING
from tqdm import tqdm
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def split_results_by_domain(domain_dict, data, predictions):
    """
    Separate the labels and predictions by domain
    :param domain_dict: dictionary, where the keys are the domain names and the values are lists with pairs [[label1, prediction1], ...]
    :param data: list containing [images, labels, domains, ...]
    :param predictions: tensor containing the predictions of the model
    :return: updated result dict
    """

    labels, domains = data[1], data[2]
    assert predictions.shape[0] == labels.shape[0], "The batch size of predictions and labels does not match!"

    for i in range(labels.shape[0]):
        if domains[i] in domain_dict.keys():
            domain_dict[domains[i]].append([labels[i].item(), predictions[i].item()])
        else:
            domain_dict[domains[i]] = [[labels[i].item(), predictions[i].item()]]

    return domain_dict


def eval_domain_dict(domain_dict, domain_seq=None):
    """
    Print detailed results for each domain. This is useful for settings where the domains are mixed
    :param domain_dict: dictionary containing the labels and predictions for each domain
    :param domain_seq: if specified and the domains are contained in the domain dict, the results will be printed in this order
    """
    correct = []
    num_samples = []
    avg_error_domains = []
    dom_names = domain_seq if all([dname in domain_seq for dname in domain_dict.keys()]) else domain_dict.keys()
    logger.info(f"Splitting up the results by domain...")
    for key in dom_names:
        content = np.array(domain_dict[key])
        correct.append((content[:, 0] == content[:, 1]).sum())
        num_samples.append(content.shape[0])
        accuracy = correct[-1] / num_samples[-1]
        error = 1 - accuracy
        avg_error_domains.append(error)
        logger.info(f"{key:<20} error: {error:.2%}")
    logger.info(f"Average error across all domains: {sum(avg_error_domains) / len(avg_error_domains):.2%}")
    # The error across all samples differs if each domain contains different amounts of samples
    logger.info(f"Error over all samples: {1 - sum(correct) / sum(num_samples):.2%}")


def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 domain_name: str,
                 setting: str,
                 domain_dict: dict,
                 device: torch.device = None,
                 ):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    correct = 0.
    mom_pre = 0.1  # model.cfg.OUR.ALPHA  # 在每个域一开始都会重置为0.1
    beta_pre = 0.1
    e = 1.0
    count = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            imgs, labels = data[0], data[1]

            """ ---------------------------------------------------------------------------------------------------------------------------------- """
            if model.cfg["LOG_DEST"].split("_")[0] in {"rottamodify"}:
            #     """ 为BN层设置动量（线性衰减）"""
            #     mom_new = mom_pre * model.cfg.OUR.DECAY_FACTOR  # 0.1 * 0.94
            #     for m in model.modules():
            #         if m.__class__.__name__ in ("RobustBN2d","RobustBN1d"):
            #             m.momentum = mom_new + model.cfg.OUR.LOWERBOunD  # mom_new + 0.005(防止mom_new衰减为0)
            #     mom_pre = mom_new

                """ 为BN层设置动量（通过第一个卷积层后的统计量与源域统计量的KL散度来决定动量值）"""
                if model.dataset_name in {"cifar10_c"}:
                    """cifar100"""
                    embedding_extractor = torch.nn.Sequential(*list(model.model_ema.children())[:1])  # cifar10
                    embedding = embedding_extractor(data[0].to(device))
                    b_var, b_mean = torch.var_mean(embedding, dim=[0, 2, 3], unbiased=False, keepdim=False)
                    source_mean, source_var = list(list(model.model_ema.children())[1].layer[0].children())[0].source_mean, \
                                              list(list(model.model_ema.children())[1].layer[0].children())[0].source_var  # cifar10

                elif model.dataset_name in {"cifar100_c","cifar10_c"}:
                    """cifar100"""
                    embedding_extractor = torch.nn.Sequential(*list(model.model_ema.children())[:1])  # cifar
                    embedding = embedding_extractor(data[0].to(device))
                    b_var, b_mean = torch.var_mean(embedding, dim=[0, 2, 3], unbiased=False, keepdim=False)
                    source_mean, source_var = list(model.model_ema.children())[1].source_mean, \
                                              list(model.model_ema.children())[1].source_var  # cifar
                    # embedding_extractor = torch.nn.Sequential(*list(model.model.children())[:1])  # cifar
                    # embedding = embedding_extractor(data[0].to(device))
                    # b_var, b_mean = torch.var_mean(embedding, dim=[0, 2, 3], unbiased=False, keepdim=False)
                    # source_mean, source_var = list(model.model.children())[1].source_mean, \
                    #                           list(model.model.children())[1].source_var  # cifar


                elif model.dataset_name in {"imagenet_c", "imagenet_r"}:
                    """imagenet"""
                    embedding_extractor = torch.nn.Sequential(*list(model.model_ema.model.children())[:1])  # imagenet
                    # embedding_extractor = torch.nn.Sequential(*list(model.model_ema.children())[:1]+list(model.model_ema.model.children())[:1])
                    embedding = embedding_extractor(data[0].to(device))
                    b_var, b_mean = torch.var_mean(embedding, dim=[0, 2, 3], unbiased=False, keepdim=False)
                    source_mean, source_var = list(model.model_ema.model.children())[1].source_mean, \
                                              list(model.model_ema.model.children())[1].source_var  # imagenet

                elif model.dataset_name in {"domainnet126"}:
                    embedding_extractor = torch.nn.Sequential(*list(model.model_ema.encoder[1][0].children())[:1])  # domainnet126
                    # embedding_extractor = torch.nn.Sequential(*list(model.model_ema.children())[:1]+list(model.model_ema.model.children())[:1])
                    embedding = embedding_extractor(data[0].to(device))
                    b_var, b_mean = torch.var_mean(embedding, dim=[0, 2, 3], unbiased=False, keepdim=False)
                    source_mean, source_var = list(model.model_ema.encoder[1][0].children())[1].source_mean, \
                                              list(model.model_ema.encoder[1][0].children())[1].source_var  # domainnet126
                else:
                    raise ValueError('Unknown dataset!!!')

                kl_distance_mean = 0.5 * F.kl_div(b_mean.softmax(dim=-1).log(), source_mean.softmax(dim=-1), reduction="sum") + \
                                   0.5 * F.kl_div(source_mean.softmax(dim=-1).log(), b_mean.softmax(dim=-1), reduction="sum")

                kl_distance_var = 0.5 * F.kl_div(b_var.softmax(dim=-1).log(), source_var.softmax(dim=-1), reduction="sum") + \
                                  0.5 * F.kl_div(source_var.softmax(dim=-1).log(), b_var.softmax(dim=-1), reduction="sum")

                kl_distance = kl_distance_mean + kl_distance_var

                # beta_t = 1 - torch.exp(-0.01 * kl_distance)

                if model.dataset_name in {"cifar10_c", "cifar100_c"}:
                    beta_t = torch.abs(kl_distance * 20)  # cifar类数据集

                elif model.dataset_name in {"imagenet_c", "imagenet_r", "domainnet126"}:
                    beta_t = torch.abs(kl_distance / 100)  # imagenet类数据集
                else:
                    raise ValueError('Unknown dataset!!!')

                beta_ema = 0.8 * beta_pre + 0.2 * beta_t



                """衰减"""
                # if i / len(data_loader) > 0.7:
                #         beta_ema *= e
                #         e = e * 0.94


                # if abs(beta_t - beta_pre) < 0.018:
                #     count += 1
                #     if count > 10:
                #         beta_ema *= e
                #         e = e * 0.94
                # else:
                #     count = 0
                #     e = 1.0

                for m in model.modules():
                    if m.__class__.__name__ in ("RobustBN2d", "RobustBN1d"):
                        """KL"""
                        # pass
                        m.momentum = beta_ema
                        """衰减 + KL"""
                        # if beta_ema + mom_new + model.cfg.OUR.LOWERBOunD <= 1:
                        #     m.momentum = beta_ema + mom_new + model.cfg.OUR.LOWERBOunD
                        # else:
                        #     m.momentum = beta_ema
                beta_pre = beta_ema
                # mom_pre = mom_new  # 衰减


            """ 为BN层设置动量（通过第一个卷积层后的统计量与源域统计量的巴士距离 / 欧式距离来决定动量值）"""
            # embedding_extractor = torch.nn.Sequential(*list(model.model_ema.children())[:1])
            # embedding = embedding_extractor(data[0].to(device))
            # b_var, b_mean = torch.var_mean(embedding, dim=[0, 2, 3], unbiased=False, keepdim=False)
            # source_mean, source_var = list(model.model_ema.children())[1].source_mean, list(model.model_ema.children())[1].source_var
            #
            # # distance = torch.sqrt(torch.sum(torch.square(source_mean-b_mean)))  # 欧氏距离
            # # distance = torch.log(torch.sum(torch.sqrt(torch.abs(source_mean * b_mean))))  # 巴士距离
            #
            # beta_t = 1 - torch.exp(-0.1 * distance)
            # beta_ema = 0.8 * beta_pre + 0.2 * beta_t
            #
            # for m in model.modules():
            #     if m.__class__.__name__ in ("RobustBN2d","RobustBN1d"):
            #         m.momentum = beta_ema
            # beta_pre = beta_t

            """eatamodify"""
            # embedding_extractor = torch.nn.Sequential(*list(model.model.model.children())[:1])  # imagenet
            # embedding = embedding_extractor(data[0].to(device))
            # b_var, b_mean = torch.var_mean(embedding, dim=[0, 2, 3], unbiased=False, keepdim=False)
            # source_mean, source_var = list(model.model.model.children())[1].source_mean, \
            #                           list(model.model.model.children())[1].source_var  # imagenet
            #
            # kl_distance_mean = F.kl_div(b_mean.softmax(dim=-1).log(), source_mean.softmax(dim=-1),
            #                             reduction="sum") + 0.5 * F.kl_div(
            #     source_mean.softmax(dim=-1).log(), b_mean.softmax(dim=-1), reduction="sum")
            #
            # kl_distance_var = F.kl_div(b_var.softmax(dim=-1).log(), source_var.softmax(dim=-1),
            #                            reduction="sum") + 0.5 * F.kl_div(
            #     source_var.softmax(dim=-1).log(), b_var.softmax(dim=-1), reduction="sum")
            #
            # kl_distance = kl_distance_mean + kl_distance_var
            #
            # # beta_t = torch.abs(kl_distance * 10)
            # # beta_t = torch.abs(kl_distance / 100)
            # beta_t = 1 - torch.exp(-10 * kl_distance)
            # beta_ema = 0.8 * beta_pre + 0.2 * beta_t
            #
            # for m in model.modules():
            #     if m.__class__.__name__ in ("RobustBN2d", "RobustBN1d"):
            #         """KL"""
            #         m.momentum = beta_ema
            #         """衰减 + KL"""
            #         # if beta_ema + mom_new + model.cfg.OUR.LOWERBOunD <= 1:
            #         #     m.momentum = beta_ema + mom_new + model.cfg.OUR.LOWERBOunD
            #         # else:
            #         #     m.momentum = beta_ema
            # beta_pre = beta_t
            # # mom_pre = mom_new  # 衰减
            """ ---------------------------------------------------------------------------------------------------------------------------------- """
            if model.cfg["LOG_DEST"].split("_")[0] in {"rottamodify"}:
                output = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(imgs.to(device))
            else:
                output = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(
                    imgs.to(device))
            predictions = output.argmax(1)

            if dataset_name == "imagenet_d" and domain_name != "none":
                mapping_vector = list(IMAGENET_D_MAPPING.values())
                predictions = torch.tensor([mapping_vector[pred] for pred in predictions], device=device)

            correct += (predictions == labels.to(device)).float().sum()

            if "mixed_domains" in setting and len(data) >= 3:
                domain_dict = split_results_by_domain(domain_dict, data, predictions)

    accuracy = correct.item() / len(data_loader.dataset)
    return accuracy, domain_dict


""" ---------------------------------------------------------------------------------------------------------------------------------- """


# 计算协方差
def covariance(features):
    assert len(features.size()) == 2, "TODO: multi-dimensional feature map covariance"
    n = features.shape[0]
    tmp = torch.ones((1, n), device=features.device) @ features
    cov = (features.t() @ features - (tmp.t() @ tmp) / n) / n
    return cov


def coral(cs, ct):
    d = cs.shape[0]
    loss = (cs - ct).pow(2).sum() / (4. * d ** 2)
    return loss


def linear_mmd(ms, mt):
    loss = (ms - mt).pow(2).mean()
    return loss


""" 获取source数据的均值和方差 """


def offline(cfg, feature_extractor=None, classifier=None, source_data_loader=None):
    # class_num = 0
    if cfg.CORRUPTION.DATASET == "cifar10_c":
        class_num = 10
        if os.path.exists('/home/tjut_hanlei/test-time-adaptation-main/classification/offline_cifar10.pth'):
            data = torch.load('/home/tjut_hanlei/test-time-adaptation-main/classification/offline_cifar10.pth')

            return data
    elif cfg.CORRUPTION.DATASET == "cifar100_c":
        class_num = 100
        if os.path.exists('/home/tjut_hanlei/test-time-adaptation-main/classification/offline_cifar100.pth'):
            data = torch.load('/home/tjut_hanlei/test-time-adaptation-main/classification/offline_cifar100.pth')

            return data
    else:
        raise Exception("还未填加提取其它数据集的statistics的情况.")

    feature_extractor.eval()
    feat_stack = [[] for i in range(class_num)]
    var_stack = [[] for i in range(class_num)]
    with torch.no_grad():
        for batch_idx, datas in tqdm(enumerate(source_data_loader), total=len(source_data_loader)):
            inputs, labels = datas[0], datas[1]

            # print(batch_idx)
            feat = feature_extractor(inputs.cuda()).squeeze(-1).squeeze(-1)
            predict_logit = classifier(feat)
            pseudo_label = predict_logit.max(dim=1)[1]

            for label in range(class_num):
                label_mask = label == labels
                feat_stack[label].extend(feat[label_mask, :])
            # for label in pseudo_label.unique():
            #     label_mask = pseudo_label == label
            #     feat_stack[label].extend(feat[label_mask, :])

    source_class_mu = []
    source_class_var = []
    source_class_cov = []
    source_feature_all = []

    for feat in feat_stack:  # 特征
        source_class_mu.append(torch.stack(feat).mean(dim=0))  # 计算 source data 的每个 class 的特征的均值
        source_class_var.append(torch.var(torch.stack(feat), dim=0))
        source_class_cov.append(covariance(torch.stack(feat)))  # 计算协方差
        source_feature_all.extend(feat)

    source_feature_all = torch.stack(source_feature_all)
    source_feature_mean = source_feature_all.mean(dim=0)
    source_feature_var = torch.var(source_feature_all, dim=0)
    source_feature_cov = covariance(source_feature_all)

    if cfg.CORRUPTION.DATASET == "cifar10_c":
        torch.save((source_class_mu, source_class_var, source_class_cov, source_feature_mean, source_feature_var,
                    source_feature_cov),
                   '/home/tjut_hanlei/test-time-adaptation-main/classification/offline_cifar10.pth')
    if cfg.CORRUPTION.DATASET == "cifar100_c":
        torch.save((source_class_mu, source_class_var, source_class_cov, source_feature_mean, source_feature_var,
                    source_feature_cov),
                   '/home/tjut_hanlei/test-time-adaptation-main/classification/offline_cifar100.pth')


"""计算方差和"""


def compute_os_variance(os, th):
    """
    Calculate the area of a rectangle.

    Parameters:
        os : OOD score queue.
        th : Given threshold to separate weak and strong OOD samples.

    Returns:
        float: Weighted variance at the given threshold th.
    """

    thresholded_os = np.zeros(os.shape)
    thresholded_os[os >= th] = 1

    # compute weights
    nb_pixels = os.size
    nb_pixels1 = np.count_nonzero(thresholded_os)
    weight1 = nb_pixels1 / nb_pixels  # OOD score > th的样本个数 / os.size
    weight0 = 1 - weight1  # OOD score < th的样本个数 / os.size

    # if one the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = os[thresholded_os == 1]  # OOD score > th的样本
    val_pixels0 = os[thresholded_os == 0]  # OOD score < th的样本

    # compute variance of these classes
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0

    return weight0 * var0 + weight1 * var1


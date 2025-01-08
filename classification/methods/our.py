"""
Builds upon: https://github.com/BIT-DA/RoTTA
Corresponding paper: https://arxiv.org/pdf/2303.13899.pdf
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from methods.base import TTAMethod
from augmentations.transforms_cotta import get_tta_transforms
# from augmentations.transforms_memo_cifar import aug_cifar
import os, sys

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from torchvision.transforms import transforms

from dataset.data_loading import get_source_loader

sys.path.insert(0, "/home/tjut_hanlei/test-time-adaptation-main/")

from classification.utils import covariance, compute_os_variance


class RoTTAMODIFY(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        """加载source loader"""
        # batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
        # _, self.src_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
        #                                        root_dir=cfg.DATA_DIR, adaptation=cfg.MODEL.ADAPTATION,
        #                                        batch_size=batch_size_src, ckpt_path=cfg.CKPT_PATH,
        #                                        percentage=cfg.SOURCE.PERCENTAGE,
        #                                        workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()))
        # self.src_loader_iter = iter(self.src_loader)
        # self.lambda_ce_src = cfg.RMT.LAMBDA_CE_SRC

        self.memory_size = cfg.ROTTA.MEMORY_SIZE
        self.lambda_t = cfg.ROTTA.LAMBDA_T
        self.lambda_u = cfg.ROTTA.LAMBDA_U
        self.nu = cfg.ROTTA.NU
        self.gama = cfg.ROTTA.gama
        self.T = 0.5


        # data = torch.load('/home/tjut_hanlei/test-time-adaptation-main/classification/offline_cifar100.pth')
        # self.source_class_mu, self.source_class_var, self.source_class_cov, self.source_feature_mean, self.source_feature_var, self.source_feature_cov = \
        # data[0], data[1], data[2], data[3], data[4], data[5],
        # self.protos = torch.load("/home/tjut_hanlei/test-time-adaptation-main/classification/ckpt/prototypes/protos_cifar10_c_Standard.pth")
        # setup the ema model
        self.count = 0
        self.mean_dict = {}
        self.var_dict = {}
        self.mean_perLayer = []
        self.var_perLayer = []
        self.lamda = 10
        print("lambda:",self.lamda)

        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.model, self.model_ema]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

        # create the test-time transformations
        self.transform = get_tta_transforms(self.dataset_name)

    @torch.enable_grad()
    def forward_and_adapt(self, x):

        imgs_test = x[0]
        # with torch.no_grad():
        with torch.enable_grad():
            # self.model.eval()
            # self.model_ema.train()

            self.model.train()
            self.model_ema.train()

            ema_out = self.model_ema(imgs_test)


            # self.model_ema.eval()
            # aug_ema_out = self.model_ema(self.aug_cifar(imgs_test))
            # predict = torch.softmax(ema_out, dim=1)
            # pseudo_label = torch.argmax(predict, dim=1)
            # aug_predict = torch.softmax(aug_ema_out, dim=1)
            # aug_pseudo_label = torch.argmax(aug_predict, dim=1)
            # mask = pseudo_label == aug_pseudo_label
            # mask
            """------------------------------------------------------------------------------------------------------------------"""
            """遍历模型的每个BN层,提取对当前batch数据进行归一化的statistics"""
            # self.count += 1
            # if self.count <= 3:
            #     for name, module in self.model_ema.named_modules():
            #         if isinstance(module, RobustBN2d):
            #             if name not in self.mean_dict:
            #                 self.mean_dict[name] = []
            #                 self.var_dict[name] = []
            #             # 计算当前批次的均值和方差
            #             mean = module.source_mean
            #             var = module.source_var
            #             # 将当前批次的均值和方差添加到列表中
            #             self.mean_dict[name].append(mean)
            #             self.var_dict[name].append(var)
            #
            #
            #     for name in self.mean_dict:
            #         mean_list = torch.stack(self.mean_dict[name])
            #         var_list = torch.stack(self.var_dict[name])
            #         mean = torch.mean(mean_list, dim=0)
            #         var = torch.mean(var_list, dim=0)
            #         self.mean_perLayer.append(mean)
            #         self.var_perLayer.append(var)
            #     if self.count == 3:
            #         torch.save(self.mean_perLayer,"/home/tjut_hanlei/test-time-adaptation-main/classification/ckpt/mean_var/cifar100_c/gaussian_noise_batch{}_mean_per_layer_by_cov1_S.pkl".format(self.count))
            #         torch.save(self.var_perLayer,"/home/tjut_hanlei/test-time-adaptation-main/classification/ckpt/mean_var/cifar100_c/gaussian_noise_batch{}_var_per_layer_by_cov1_S.pkl".format(self.count))
            #     print("完成！")
            """------------------------------------------------------------------------------------------------------------------"""

            """ 筛选低熵样本 """
            predict = torch.softmax(ema_out, dim=1)
            # pseudo_label = torch.argmax(predict, dim=1)
            # confidence, _ = torch.max(predict, dim=1)
            entropy = torch.sum(- predict * torch.log(predict + 1e-6), dim=1)
            # best_threshold = 10
            # mask = entropy < best_threshold
            threshold_range = np.arange(0, 4, 0.01)
            criterias = [compute_os_variance(np.array(entropy.detach().cpu()), th) for th in threshold_range]
            best_threshold = threshold_range[np.argmin(criterias)]
            # best_threshold = 0.4 * math.log(10)
            mask = entropy < best_threshold  # 低熵样本的mask向量




            """------------------------------------------------------------------------------------------------------------------"""
            """student 复制 teacher 的动量"""
            # 获取教师模型和学生模型的所有BN层
            # teacher_bns = [m for m in self.model_ema.modules() if isinstance(m, RobustBN2d)]
            # student_bns = [m for m in self.model.modules() if isinstance(m, RobustBN2d)]
            # # 检查教师模型和学生模型的BN层数量是否相同
            # if len(teacher_bns) != len(student_bns):
            #     raise ValueError(
            #         "The number of BatchNorm2d layers in the teacher model and the student model must be equal.")
            # # 遍历每个BN层，将教师模型的momentum值赋值给学生模型
            # for t_bn, s_bn in zip(teacher_bns, student_bns):
            #     s_bn.momentum = t_bn.momentum
            """------------------------------------------------------------------------------------------------------------------"""

            """ MixMatch算法提高对低高熵样本的预测 """
            # impossible_imgs = imgs_test
            # impossible_imgs_1 = train_transform_1(impossible_imgs)
            # # impossible_imgs_2 = train_transform_2(impossible_imgs)
            # outputs_u = self.model_ema(impossible_imgs_1)
            # # outputs_u2 = self.model_ema(impossible_imgs_2)
            # impossible_imgs_out = (outputs_u + ema_out) / 2  # 求两次的平均值
            # sharpen_out = impossible_imgs_out ** (1 / self.T)
            # sharpen_out = sharpen_out.detach()

            """ 筛选高熵样本中的低熵样本 """
            # threshold_range = np.arange(1, 5, 0.01)
            # # entropy = F.normalize(entropy,dim=-1) * 100
            # # threshold_range = np.arange(0, 20, 0.01)
            # criterias = [compute_os_variance(np.array(entropy[~mask].cpu()), th) for th in threshold_range]
            # best_threshold = threshold_range[np.argmin(criterias)]
            # mask_2 = entropy[~mask] < best_threshold

            stu_sup_out = self.model(self.transform(imgs_test))


            """ 消融 """
            # loss_distill = (softmax_entropy(stu_sup_out, ema_out)).mean()


            """ 加权蒸馏 """
            coeff = (1 / (torch.exp(entropy[mask].clone().detach() - best_threshold)))
            loss_distill = (softmax_entropy(stu_sup_out[mask], ema_out[mask]).mul(coeff)).mean()
            # coeff = (1 / (torch.exp(entropy.clone().detach() - best_threshold))) / 0.1
            # loss_distill = (softmax_entropy(stu_sup_out, ema_out).mul(coeff)).mean()



            """ 加权蒸馏 + Stu的自熵"""
            # predict_S = torch.softmax(stu_sup_out, dim=1)
            # entropy_S = torch.sum(- predict_S * torch.log(predict_S + 1e-6), dim=1)
            # threshold_range_S = np.arange(0, 4, 0.01)
            # criterias_S = [compute_os_variance(np.array(entropy_S.detach().cpu()), th) for th in threshold_range_S]
            # best_threshold_S = threshold_range_S[np.argmin(criterias_S)]
            # mask_S = entropy_S < best_threshold_S
            # coeff_S = 1/torch.exp(entropy_S.clone().detach()-best_threshold_S)
            # loss_distill = (softmax_entropy(stu_sup_out[mask], ema_out[mask]).mul(coeff)).mean() + entropy_S[mask_S].mul(coeff_S[mask_S]).mean()

            # loss_distill = (l2_dis(stu_sup_out[mask], ema_out[mask]).mul(coeff)).mean()
            # TS_entropy = softmax_entropy(stu_sup_out, ema_out)
            # loss_distill = TS_entropy[mask].mul(coeff).mean()



            if loss_distill is not None:
                self.optimizer.zero_grad()
                loss_distill.backward()
                self.optimizer.step()

            """重放训练集的数据"""
            # if self.lambda_ce_src > 0:
            #     # sample source batch
            #     try:
            #         batch = next(self.src_loader_iter)
            #     except StopIteration:
            #         self.src_loader_iter = iter(self.src_loader)
            #         batch = next(self.src_loader_iter)
            #     # train on labeled source data
            #     imgs_src, labels_src = batch[0], batch[1]
            #     self.model.eval()
            #     outputs_src = self.model(imgs_src.cuda())
            #     loss_ce_src = F.cross_entropy(outputs_src, labels_src.cuda().long())
            #     loss_ce_src *= self.lambda_ce_src
            #     loss_ce_src.backward()
            # self.optimizer.step()


            self.update_ema_variables(self.model_ema, self.model, self.nu)






            # return torch.where(mask.unsqueeze(1).expand(-1,100).cuda(), ema_out.cuda(), sharpen_out.cuda()) + \
            #        torch.where(~mask.unsqueeze(1).expand(-1,100).cuda(), stu_sup_out.cuda(), torch.tensor(0).cuda())

            # return ema_out
            # return ema_out + stu_sup_out
            return ema_out + (1-torch.exp(-self.lamda * entropy)).unsqueeze(1).repeat(1, stu_sup_out.size(1)) * stu_sup_out
            # return self.gama * ema_out + stu_sup_out
            # return torch.where(mask.unsqueeze(1).expand(-1, 100).cuda(), ema_out.cuda(), sharpen_out.cuda()) + stu_sup_out

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved self.model/optimizer state")
        self.load_model_and_optimizer()

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model

    def configure_model(self):
        self.model.requires_grad_(False)
        # self.model.requires_grad_(True)
        normlayer_names = []

        for name, sub_module in self.model.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)
            elif isinstance(sub_module, (nn.LayerNorm, nn.GroupNorm)):
                sub_module.requires_grad_(True)

        for name in normlayer_names:
            bn_layer = get_named_submodule(self.model, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = RobustBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = RobustBN2d
            else:
                raise RuntimeError()

            momentum_bn = NewBN(bn_layer, self.cfg.ROTTA.ALPHA)
            momentum_bn.requires_grad_(True)
            set_named_submodule(self.model, name, momentum_bn)
        # self.model[1:2][0].fc.requires_grad_(True)


@torch.jit.script
def softmax_entropy(x, x_ema):
    # return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1) - (x.softmax(1) * x_ema.log_softmax(1)).sum(1)
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def l2_dis(x, x_ema):
    return torch.sqrt(torch.sum(torch.pow(x_ema - x, 2), dim=1))


def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module


def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])

        else:
            setattr(module, names[i], value)


class MomentumBN(nn.Module):
    def __init__(self, bn_layer: nn.BatchNorm2d, momentum):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.momentum = momentum
        self.beta_pre = 0.1
        self.count = 0
        if bn_layer.track_running_stats and bn_layer.running_var is not None and bn_layer.running_mean is not None:
            self.register_buffer("source_mean", deepcopy(bn_layer.running_mean))
            self.register_buffer("source_var", deepcopy(bn_layer.running_var))
            self.register_buffer("target_mean", deepcopy(bn_layer.running_mean))
            self.register_buffer("target_var", deepcopy(bn_layer.running_var))
            self.source_num = bn_layer.num_batches_tracked
        self.weight = deepcopy(bn_layer.weight)
        self.bias = deepcopy(bn_layer.bias)

        # self.register_buffer("target_mean", torch.zeros_like(self.source_mean))
        # self.register_buffer("target_var", torch.ones_like(self.source_var))
        self.eps = bn_layer.eps

        self.current_mu = None
        self.current_sigma = None

    def forward(self, x):
        raise NotImplementedError


class RobustBN1d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=0, unbiased=False, keepdim=False)  # (C,)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(var.detach())
            mean, var = mean.view(1, -1), var.view(1, -1)
        else:
            mean, var = self.source_mean.view(1, -1), self.source_var.view(1, -1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1)
        bias = self.bias.view(1, -1)

        return x * weight + bias


class RobustBN2d(MomentumBN):

    def forward(self, x):

        if self.training:


            """------------------------------------------------------------------------------------------------------------------"""
            """计算每层的域偏移度"""
            # if self.count == 1000:
            #     self.count = 0
            #     self.beta_pre = 0.1

            # kl_distance_mean = 0.5 * F.kl_div(b_mean.softmax(dim=-1).log(), self.source_mean.softmax(dim=-1), reduction="sum") + \
            #                    0.5 * F.kl_div(self.source_mean.softmax(dim=-1).log(), b_mean.softmax(dim=-1), reduction="sum")
            # kl_distance_var = 0.5 * F.kl_div(b_var.softmax(dim=-1).log(), self.source_var.softmax(dim=-1), reduction="sum") + \
            #                   0.5 * F.kl_div(self.source_var.softmax(dim=-1).log(), b_var.softmax(dim=-1), reduction="sum")
            # kl_distance = kl_distance_mean + kl_distance_var
            #
            # beta_t = torch.abs(kl_distance * 10)
            # beat_ema = 0.8 * self.beta_pre + 0.2 * beta_t
            # self.momentum = beat_ema
            # self.beta_pre = beta_t
            """------------------------------------------------------------------------------------------------------------------"""
            b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False, keepdim=False)  # (C,)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(var.detach())
            # self.target_mean, self.target_var = deepcopy(mean.detach()), deepcopy(var.detach())
            mean, var = mean.view(1, -1, 1, 1), var.view(1, -1, 1, 1)


        else:
            mean, var = self.source_mean.view(1, -1, 1, 1), self.source_var.view(1, -1, 1, 1)

            """------------------------------------------------------------------------------------------------------------------"""
            # b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False, keepdim=False)  # (C,)
            # mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            # var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            # self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(var.detach())
            # mean, var = mean.view(1, -1, 1, 1), var.view(1, -1, 1, 1)
            """------------------------------------------------------------------------------------------------------------------"""

        """ 通过计算得到的新的统计量对当前数据归一化 """
        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)

        return x * weight + bias

class SoftLikelihoodRatio(nn.Module):
    def __init__(self, clip=0.99, eps=1e-5):
        super(SoftLikelihoodRatio, self).__init__()
        self.eps = eps
        self.clip = clip

    def __call__(self, logits):
        logits = logits.softmax(1)
        probs = torch.clamp(logits, min=0.0, max=self.clip)
        return - (probs * torch.log((probs / (torch.ones_like(probs) - probs)) + self.eps)).sum(1)

import os
import logging
import random

import numpy as np
import torch
from models.model import get_model
from utils import get_accuracy, eval_domain_dict, offline
from dataset.data_loading import get_test_loader, get_source_loader
from conf import cfg, load_cfg_from_args, get_num_classes, get_domain_sequence, adaptation_method_lookup

from methods.tent import Tent
# from methods.ttaug import TTAug
# from methods.memo import MEMO
from methods.cotta import CoTTA
# from methods.gtta import GTTA
# from methods.adacontrast import AdaContrast
# from methods.rmt import RMT
from methods.eata import EATA
from methods.norm import Norm
# from methods.lame import LAME
from methods.sar import SAR
# from methods.rotta import RoTTA
# from methods.roid import ROID
# from methods.tipi import TIPI
from methods.our import RoTTAMODIFY as OUR
# from methods.rottamodify import RoTTAMODIFY
# from methods.cottaModify import CottaModify
# from methods.eatamodify import EATAMODIFY
# from methods.rmt_revise import RMT_REVISE

logger = logging.getLogger(__name__)
# import sys
# sys.path.insert(0,'/home/tjut_hanlei/test-time-adaptation-main/classification')
# sys.path.insert(0,'/home/tjut_hanlei/test-time-adaptation-main')

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

def evaluate(description):
    load_cfg_from_args(description)
    valid_settings = ["reset_each_shift",  # reset the model state after the adaptation to a domain
                      "continual",  # train on sequence of domain shifts without knowing when a shift occurs
                      "gradual",  # sequence of gradually increasing / decreasing domain shifts
                      "mixed_domains",  # consecutive test samples are likely to originate from different domains
                      "correlated",  # sorted by class label
                      "mixed_domains_correlated",  # mixed domains + sorted by class label
                      "gradual_correlated",  # gradual domain shifts + sorted by class label
                      "reset_each_shift_correlated"
                      ]
    assert cfg.SETTING in valid_settings, f"The setting '{cfg.SETTING}' is not supported! Choose from: {valid_settings}"

    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    base_model = get_model(cfg, num_classes)

    # setup test-time adaptation method
    model = eval(f'{adaptation_method_lookup(cfg.MODEL.ADAPTATION)}')(cfg=cfg, model=base_model,
                                                                      num_classes=num_classes)
    logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION.upper()}")
    logger.info(f"Successfully prepared test-time adaptation SETTING: {cfg.SETTING}")
    logger.info(f"Successfully prepared test-time adaptation dataset: {cfg.CORRUPTION.DATASET}")
    logger.info(f"Successfully prepared test-time adaptation MODEL: {cfg.MODEL['ARCH']}")


    # get the test sequence containing the corruptions or domain names
    if cfg.CORRUPTION.DATASET in {"domainnet126"}:
        # extract the domain sequence for a specific checkpoint.
        dom_names_all = get_domain_sequence(ckpt_path=cfg.CKPT_PATH)
    elif cfg.CORRUPTION.DATASET in {"imagenet_d", "imagenet_d109"} and not cfg.CORRUPTION.TYPE[0]:
        # dom_names_all = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        dom_names_all = ["clipart", "infograph", "painting", "real", "sketch"]
    else:
        dom_names_all = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {dom_names_all}")

    # prevent iterating multiple times over the same data in the mixed_domains setting
    dom_names_loop = ["mixed"] if "mixed_domains" in cfg.SETTING else dom_names_all

    # setup the severities for the gradual setting
    if "gradual" in cfg.SETTING and cfg.CORRUPTION.DATASET in {"cifar10_c", "cifar100_c", "imagenet_c"} and len(
            cfg.CORRUPTION.SEVERITY) == 1:
        severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        logger.info(f"Using the following severity sequence for each domain: {severities}")
    else:
        severities = cfg.CORRUPTION.SEVERITY

    errs = []
    errs_5 = []
    domain_dict = {}

    """ 获取source domain的数据 """
    # source_data_loader = get_source_loader(dataset_name="cifar100",
    #                                        root_dir="/data/HDD1/hanlei/data/cifar-100-python",
    #                                        adaptation=cfg.MODEL.ADAPTATION,
    #                                        batch_size=cfg.TEST.BATCH_SIZE,
    #                                        )[1]
    """ 拆分特征提取器和分类器 """
    # # 去掉最后一层全连接层
    # feature_extractor = torch.nn.Sequential(*list(model.model.children())[:-1])
    # # 获取分类器（最后一层全连接层）
    # classifier = list(model.model.children())[-1]
    # source_class_mu, source_class_var, source_class_cov, source_feature_mean, source_feature_var, source_feature_cov = offline(
    #     cfg, feature_extractor, classifier, source_data_loader)



    # start evaluation
    for i_dom, domain_name in enumerate(dom_names_loop):
        if i_dom == 0 or "reset_each_shift" in cfg.SETTING:
            try:
                model.reset()
                logger.info("resetting model")
            except:
                logger.warning("not resetting model")
        else:
            logger.warning("not resetting model")


        for severity in severities:
            test_data_loader = get_test_loader(setting=cfg.SETTING,
                                               adaptation=cfg.MODEL.ADAPTATION,  # 方法
                                               dataset_name=cfg.CORRUPTION.DATASET,
                                               root_dir=cfg.DATA_DIR,  # /data/HDD1/hanlei/data/
                                               domain_name=domain_name,
                                               severity=severity,
                                               num_examples=cfg.CORRUPTION.NUM_EX,
                                               rng_seed=cfg.RNG_SEED,
                                               domain_names_all=dom_names_all,
                                               alpha_dirichlet=cfg.TEST.ALPHA_DIRICHLET,
                                               batch_size=cfg.TEST.BATCH_SIZE,
                                               shuffle=False,
                                               workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()))

            # evaluate the model
            acc, domain_dict = get_accuracy(model,
                                            data_loader=test_data_loader,
                                            dataset_name=cfg.CORRUPTION.DATASET,
                                            domain_name=domain_name,
                                            setting=cfg.SETTING,
                                            domain_dict=domain_dict,)

            err = 1. - acc
            errs.append(err)
            if severity == 5 and domain_name != "none":
                errs_5.append(err)

            logger.info(
                f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][#samples={len(test_data_loader.dataset)}]: {err:.2%}")

    if len(errs_5) > 0:
        logger.info(f"mean error: {np.mean(errs):.2%}, mean error at 5: {np.mean(errs_5):.2%}")
    else:
        logger.info(f"mean error: {np.mean(errs):.2%}")

    if "mixed_domains" in cfg.SETTING:
        # print detailed results for each domain
        eval_domain_dict(domain_dict, domain_seq=dom_names_all)


if __name__ == '__main__':
    evaluate('"Evaluation.')

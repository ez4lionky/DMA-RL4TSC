from .pipeline import Pipeline
from .oneline import OneLine
from . import config
import os
import json
import shutil
import copy
import tensorflow as tf
from datetime import datetime
import logging


def get_logger(log_dir, debug=False):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    file_name_time = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    file_name = f"{log_dir}/{file_name_time}"

    if not debug:
        fh = logging.FileHandler(file_name + '.log')
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    return logger


def set_GPU():
    # tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # tf.config.experimental.set_visible_devices(gpus[-1], 'GPU')
    # logical_gpus = tf.config.experimental.list_logical_devices('GPU')


def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result


def pipeline_wrapper(dic_agent_conf, dic_traffic_env_conf, dic_path, logger=None):
    ppl = Pipeline(dic_agent_conf=dic_agent_conf,
                   dic_traffic_env_conf=dic_traffic_env_conf,
                   dic_path=dic_path,
                   logger=logger
                   )
    ppl.run(multi_process=False)

    print("pipeline_wrapper end")
    return


def oneline_wrapper(dic_agent_conf, dic_traffic_env_conf, dic_path, logger=None):
    oneline = OneLine(dic_agent_conf=dic_agent_conf,
                      dic_traffic_env_conf=merge(config.dic_traffic_env_conf, dic_traffic_env_conf),
                      dic_path=merge(config.DIC_PATH, dic_path),
                      logger=logger
                      )
    oneline.train()
    return


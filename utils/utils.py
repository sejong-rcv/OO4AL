import logging
import os
import random

def initial_budget_sampling(indices, args):
    random.shuffle(indices)
    labeled_set = indices[:args.initial]
    unlabeled_set = indices[args.initial:]

    return labeled_set, unlabeled_set

def get_logger(args):
    # logger
    logger = logging.getLogger("Acc")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    project_dir = args.save_name
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    # print on console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # print on file
    file_handler = logging.FileHandler(os.path.join(project_dir, 'accuracy.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    for k, v in vars(args).items():
        logger.info("[ARG] arg.{}: {}".format(k, v))
    return logger
import argparse
import logging
import os
import torch.nn as nn
import torch
import yaml
import torch
import datetime
import shutil


if __name__ == '__main__':
    current_datetime = datetime.datetime.now()
    datetime_string = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='classification', help='The main task', required=True)
    parser.add_argument('--mode', type=str, default="train", help="train/test process")
    ###
    opt = parser.parse_args()
from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torch import nn

import os
import sys

from neural_net.Datasets import ClassificationLoader
from neural_net.model_speech_yolo import VGG
from neural_net.train_classification import test


class NeuralNet:

    def set_args(self, model_path, class_num):

        parser = argparse.ArgumentParser(
            description='ConvNets for Speech Commands Recognition')

        parser.add_argument('--arc', default='VGG11',
                            help='network architecture: VGG11, VGG13, VGG16, VGG19')
        parser.add_argument('--epochs', type=int, default=20,
                            metavar='N', help='number of epochs to train')
        parser.add_argument('--cuda', default=True, help='enable CUDA')
        parser.add_argument('--seed', type=int, default=1234,
                            metavar='S', help='random seed')
        parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                            help='num of batches to wait until logging train status')
        parser.add_argument('--class_num', type=int, default=class_num,
                            help='number of classes to classify')
        #   parser.add_argument('--prev_classification_model', type=str, default='pretraining_model/optimizer_adam_lr_0.001_batch_size_32_arc_VGG11_class_num_7.pth',
        #                       help='the location of the prev classification model')

        parser.add_argument('--prev_classification_model', type=str,
                            default=model_path,
                            help='the location of the prev classification model')

        self.args = parser.parse_args()

        self.args.cuda = self.args.cuda and torch.cuda.is_available()
        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.manual_seed(self.args.seed)

    def __init__(self, model_path, num_of_classes):

        self.set_args(model_path, num_of_classes)
        self.create_model()

    def load_model_pre(self, model):
        checkpoint = torch.load(self.args.prev_classification_model, map_location=lambda storage, loc: storage)

      #  model.load_state_dict(checkpoint['net'], strict=False)

        current_dict = model.state_dict()
        saved_values = list(checkpoint['net'].values())
        index = 0
        for key, val in current_dict.items():
            current_dict[key] = saved_values[index]
            index += 1

        model.load_state_dict(current_dict)

        return model, checkpoint['acc'], checkpoint['epoch'], checkpoint['class_num']

    def create_model(self):

        # build model
        if self.args.arc.startswith("VGG"):
            model = VGG(self.args.arc, self.args.class_num)
        else:
            model = VGG("VGG11", self.args.class_num)

        if self.args.cuda:
            print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model).cuda()

        # build model
        if os.path.isfile(self.args.prev_classification_model):  # model exists

            model, check_acc, check_epoch, class_num = self.load_model_pre(model)
            print(f"found trained model, prev valid loss: {check_acc}, after {check_epoch} epochs")
            self.model = model

    def create_new_test(self, path):

        test_dataset = ClassificationLoader(path, window_size=.02, window_stride=.01,
                                            window_type='hamming', normalize=True,
                                            max_len=101)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=32, shuffle=None,
            num_workers=0, pin_memory=self.args.cuda)

        sum_all = test(test_loader, self.model, self.args.cuda)
        return sum_all.item()





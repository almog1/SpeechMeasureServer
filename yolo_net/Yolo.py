import argparse
import os

import torch

from neural_net import Datasets, utils
from neural_net.model_speech_yolo import VGG, SpeechYoloVGGNetExtended, SpeechYoloVGGNet, load_model1


class Yolo:

    def set_args(self):
        parser = argparse.ArgumentParser(description='train yolo model')
        parser.add_argument('--arc', type=str, default='VGG19',
                            help='arch method (LeNet, VGG11, VGG13, VGG16, VGG19)')
        parser.add_argument('--decision_threshold', type=float, default=0.25,
                            help=' object exist threshold')

        args = parser.parse_args()
        args.cuda = torch.cuda.is_available()



        self.args = args

    def load_model_weights(self, net, model):

        current_dict = model.state_dict()
        saved_values = list(net.values())
        index = 0
        for key, val in current_dict.items():
            current_dict[key] = saved_values[index]
            index += 1

        return current_dict


    def load_model(self, save_dir, is_extended):


        speech_net, check_epoch, loss, self.config_dict = load_model1(save_dir, is_extended)

        if self.args.cuda:
            print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
            speech_net = torch.nn.DataParallel(speech_net).cuda()

        print(f"found trained model, prev valid loss: {loss}, after {check_epoch} epochs")
        self.model = speech_net

    def __init__(self, model_path, is_extended=True):

        self.set_args()
        self.load_model(model_path, is_extended)

    def create_new_test(self, path, t):

        test_dataset = Datasets.SpeechYoloDataSet(path, self.config_dict)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=False,
                                                  num_workers=0,
                                                  pin_memory=self.args.cuda, sampler=None)

        sum_all = self.test(test_loader, t)
        return sum_all

    def test(self, loader, t):
        global_total = 0
        with torch.no_grad():
            self.model.eval()

            for data in loader:
                if self.args.cuda:
                    data = data.cuda()

                output = self.model(data)
                current = self.yolo_accuracy(output, t)
                global_total += current

        return global_total

    def yolo_accuracy(self, prediction,t):

        C, B, K = self.config_dict['C'], self.config_dict['B'], self.config_dict['K']
        pred_ws, pred_start, pred_end, pred_conf, pred_class_all_prob = utils.extract_data(prediction, C, B, K)
        pred_classes_prob, pred_classes = torch.max(pred_class_all_prob, 3)
        conf_class_mult, box_index = torch.max((pred_conf * pred_classes_prob), 2)

        #no_object_correct = torch.eq((conf_class_mult < T).float(), 1 - target[:, :, -1]).cpu().sum()
        #no_object_object_wrong = (torch.eq((conf_class_mult < T).float(), target[:, :, -1])).cpu().sum()

        box_indices_array = box_index.cpu().numpy()

        total_classes = 0
        for batch in range(0, box_indices_array.shape[0]):
            for cell in range(0, box_indices_array.shape[1]):
                if (conf_class_mult > t)[batch,cell].item() == 1:
                    total_classes +=1

        return total_classes


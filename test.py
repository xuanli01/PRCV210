#!/usr/bin/python3
#coding=utf-8
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import dataset
from model import Model
import argparse
from tqdm import tqdm
from sod_metrics import Emeasure, Fmeasure, WeightedFmeasure
from sod_metrics import MAE as calMAE


class Test(object):
    def __init__(self, args, test_set):
        ## dataset
        self.test_set = test_set
        self.data_root = args.data_root
        self.modelpath = args.checkpoint
        self.cfg = dataset.Config(data_root = args.data_root,
                                  test_set = test_set, 
                                  snapshot = self.modelpath,
                                  mode='test')
        self.data = dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        self.net = Model(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def predict(self):
        print("start predict....")
        FM = Fmeasure()
        WFM = WeightedFmeasure()
        EM = Emeasure()
        MAE = calMAE()
        with torch.no_grad():
            for image, mask, shape, name in tqdm(iter(self.loader)):
                image = image.cuda().float()
                spv, feature = self.net(image, shape) 
                out = spv[0] 
                pred = (torch.sigmoid(out[0, 0]) * 255).cpu().numpy()
                pred = np.round(pred)
                mask = mask[0].cpu().numpy()
                FM.step(pred=pred, gt=mask)
                WFM.step(pred=pred, gt=mask)
                EM.step(pred=pred, gt=mask)
                MAE.step(pred=pred, gt=mask)
                save_path = 'result/test/' + self.test_set
                os.makedirs(save_path, exist_ok=True)
                cv2.imwrite(save_path + '/' + name[0] + '.png', pred)
            fm = FM.get_results()["fm"]
            wfm = WFM.get_results()["wfm"]
            em = EM.get_results()["em"]
            mae = MAE.get_results()["mae"]
            ret = "%.3f\t%.3f\t%.3f\t%.3f\t" % \
                        (fm["adp"],wfm, mae, em["adp"])
            with open("./result/test/" + self.test_set + ".txt", 'a') as f:
                f.write(ret + self.modelpath + "\n")
            results = self.test_set+"\n"+\
                      "F_beta      : %.3f\n" \
                      "F_Omega     : %.3f\n" \
                      "MAE         : %.3f\n" \
                      "E_gamma     : %.3f\n" \
                      % (fm["adp"],wfm, mae, em["adp"])
            print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='myModel')
    parser.add_argument('-p', '--data_root', help='the root of testing data')
    parser.add_argument('-c', '--checkpoint', help='the path  of the checkpoint')
    args = parser.parse_args()

    #for test_set in ["ECSSD", "PASCAL-S", "DUT-OMRON", "DUTS-TE", "HKU-IS"]:
    for test_set in ["ECSSD", "PASCAL-S", "DUT-OMRON", "DUTS", "HKU-IS"]:
        test = Test(args, test_set)
        test.predict()



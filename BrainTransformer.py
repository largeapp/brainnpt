from sklearn.model_selection import StratifiedKFold
import torch
from static.dataset_path import dataset_path
from static.model import *
from utils.logger import Log
from torch.cuda.amp import autocast, GradScaler
from static.optimizer import Optimizer
from static.dataset import MyDataset,BuildDataloader,collater
#from classify.dataset_path import dataset_path
from utils.utils import GenerateOOV, AverageMeter
import os
import time
import torch.nn as nn
from static.eval import eval_model
import numpy as np
from sklearn.model_selection import train_test_split

logger = Log(__name__).getlog()

class Classification():
    def __init__(self):
        logger.info(f"hello {__name__}")
        logger.info(f"Classification is processing")

    def run(self, args):

        scaler = GradScaler()
        # 数据加载
        path = dataset_path(args.dataset_name)
        dataset = MyDataset(args, path)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        k = 0
        Y = np.zeros(len(dataset))
        features = Y
        dec_lossfn = nn.CrossEntropyLoss()
        #dec_lossfn = focal_loss()
        acc_list = []
        bac_list = []
        sen_list = []
        spec_list = []
        auc_list = []
        f1_list = []
        for train_index, test_index in skf.split(features, Y):
            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device = torch.cuda.current_device()
            model = NGPBrainNetwork(args)
            # 加载模型        
            if args.resume_file:
                ori_model_state_dict = torch.load(args.resume_file)
                model.load_state_dict(ori_model_state_dict, strict=True)
                logger.info(f"successfully load the previous checkpoint from {args.resume_file}")
            model = model.to(device)  # model中的tensor不会转到devcie，只有变量才会转到devcie

            train_set = dataset[train_index]
            dev_set = dataset[test_index]
            collate_fn = collater(path)
            train_loader = BuildDataloader(train_set, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers,collate_fn=collate_fn)
            collate_fn = collater(path)
            dev_loader = BuildDataloader(dev_set, batch_size=args.dev_bs, shuffle=False, num_workers=args.num_workers,collate_fn=collate_fn)
            # 优化器加载
            steps_per_epoch = len(train_loader)
            optimizer_class = Optimizer(args, [model], all_model=model,steps_per_epoch=steps_per_epoch)
            optimizer, scheduler = optimizer_class.get_optimizer()
            min_loss = 1e6
            best_acc_dev =  0
            best_acc_epoch = 0
            best_bac = 0
            best_sen = 0
            best_spec = 0
            best_auc = 0
            best_f1 = 0
            # 创建输出目录
            output_path = f"{args.output}/{args.exp_name}"
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            if args.exp_name == "debug":
                save_prefix = "debug-"
                if args.resume_file:
                    save_prefix += args.resume_file.split("/")[-1]
            else:
                save_prefix = time.strftime("%Y%m%d%H%M%S", time.localtime())
            train_model_output_path = f"{output_path}/{save_prefix}-min_loss.pth"
            dev_model_output_path = f"{output_path}/{save_prefix}-dev.pth"

            for epoch in range(args.epochs):
                if "train" in args.running_type:
                    epoch_losses = AverageMeter()
                    model.train()
                    optimizer.zero_grad()
                    for step, (dynamic_feature,label) in enumerate(train_loader):
                        dynamic_feature = dynamic_feature.to(device)
                        label = label.to(device)
                        with autocast():
                            scores_pred = model(dynamic_feature)
                            generate_loss = dec_lossfn(scores_pred, label)
                            total_loss = generate_loss
                        assert torch.isnan(total_loss).sum() == 0
                        # 正常梯度回传
                        # optimizer.zero_grad()
                        # total_loss.backward()
                        # optimizer.step()
                        # scheduler.step()
                        scaler.scale(total_loss).backward()
                        if ((step + 1) % args.accum_iter == 0) or ((step + 1) == len(train_loader)):
                            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
                            scaler.step(optimizer)
                            scaler.update()
                            scheduler.step()
                            optimizer.zero_grad()

                        batch = scores_pred.size(0)
                        epoch_losses.update(total_loss.item(), batch)
                        if step % 40 == 0:
                            logger.info(f"{epoch}-{step} | avg_loss: {epoch_losses.avg}, cur_loss: {total_loss.item()}")

                if "dev" in args.running_type:  # TO LOOK
                    model.eval()
                    dev_acc,balanced,sen,spec,auc,f1 = eval_model(model, dev_loader)
                    logger.info(f"epoch: {epoch}, dev_acc_score: {dev_acc}")
                    if dev_acc >= best_acc_dev:
                        best_acc_dev = dev_acc
                        best_acc_epoch = epoch
                        if args.exp_name != "debug":
                            torch.save(model.state_dict(), dev_model_output_path)
                        best_bac = balanced
                        best_sen = sen
                        best_spec = spec
                        best_auc = auc
                        best_f1 = f1
            acc_list.append(best_acc_dev)
            bac_list.append(best_bac)
            sen_list.append(best_sen)
            spec_list.append(best_spec)
            auc_list.append(best_auc)
            f1_list.append(best_f1)
            logger.info(f"best epoch: {best_acc_epoch}, dev_acc_score: {best_acc_dev}, dev_balanced_acc: {best_bac},sensitivity: {best_sen}, specificity:{best_spec}, AUC:{best_auc},F1:{best_f1}")
            del(model)
            del(train_set)
            del(dev_set)
        print("average accuracy: " + str(np.mean(acc_list)) + "    var:" + str(np.std(acc_list)))
        print("average balanced accuracy: " + str(np.mean(bac_list)) + "    var:" + str(np.std(bac_list)))
        print("average sensitivity: " + str(np.mean(sen_list)) + "    var:" + str(np.std(sen_list)))
        print("average specificity: " + str(np.mean(spec_list)) + "    var:" + str(np.std(spec_list)))
        print("average auc: " + str(np.mean(auc_list)) + "     var:" + str(np.std(auc_list)))
        print("average f1: " + str(np.mean(f1_list)) + "    var:" + str(np.std(f1_list)))
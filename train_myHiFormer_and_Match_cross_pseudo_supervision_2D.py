import argparse
import logging
import os
import random
import shutil
import sys
import time
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import ImageFilter
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler, Synapse_dataset, BaseDataSets_Synapse, BaseDataSets_BraTS,
                                 BaseDataSets_LA, BaseDataSets_FLARE22, RandomGenerator_slice)
from networks1.net_factory import net_factory
import losses, metrics, ramps
from val_2D import test_myHiFormer_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/MyHiFormer_and_Match_cross_pseudo_supervision', help='experiment_name')
parser.add_argument('--model_encoder', type=str,
                    default='myHiFormer', help='model_name')
parser.add_argument('--model_decoder', type=str,
                    default='unet_decoder', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# parser.add_argument("--list_dir", type=str, default="../data/Synapse/lists/lists_Synapse", help="list dir")
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=14,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model
def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]
# def patients_to_slices(dataset, patiens_num):
#      ref_dict = {"0": 0.05, "1": 0.1, "2": 0.2,
#                  "3": 0.3, "4": 0.4, "5": 0.5, "6": 0.6, "7": 0.7, "8": 0.8, "9": 0.9,"10": 1}
#      return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img
def tensor2Image(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)
def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False,net_type=args.model_encoder):
        # Network definition
        model = net_factory(net_type=net_type, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model_encoder = create_model()
    model_decoder1 = create_model(net_type=args.model_decoder)
    model_decoder2 = create_model(net_type=args.model_decoder)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator_slice(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    # db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", img_size=224,
    #                            transform=transforms.Compose([
    #                                    RandomGenerator(args.patch_size)
    #                                ]))
    # db_val = Synapse_dataset(base_dir=args.root_path, split="val", list_dir=args.list_dir, img_size=224)
    total_slices = len(db_train)
    labeled_slice =patients_to_slices(args.root_path, args.labeled_num)
    # labeled_slice = int(total_slices*patients_to_slices(args.root_path, args.labeled_num))
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn
                             )

    model_encoder.train()
    model_decoder1.train()
    model_decoder2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,num_workers=1)

    optimizer_encoder = optim.SGD(model_encoder.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer_decoder1 = optim.SGD(model_decoder1.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer_decoder2 = optim.SGD(model_decoder2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(1), label_batch.cuda(1)

            encoder_out = model_encoder(volume_batch)
            outputs1 = model_decoder1(encoder_out[1],encoder_out[3],encoder_out[5])
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2 = model_decoder2(encoder_out[0],encoder_out[2],encoder_out[4])
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            img_s1=deepcopy(volume_batch)
            for b in range(img_s1.size(0)):
                img_s = transforms.ToPILImage()(img_s1[b])
                if random.random() < 0.8:
                    img_s = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s)
                img_s = transforms.RandomGrayscale(p=0.2)(img_s)
                img_s = blur(img_s, p=0.5)
                img_s1[b] = transforms.ToTensor()(img_s)

            img_s2 = deepcopy(volume_batch)
            for b in range(img_s2.size(0)):
                img_s = transforms.ToPILImage()(img_s2[b])
                if random.random() < 0.8:
                    img_s = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s)
                img_s = transforms.RandomGrayscale(p=0.2)(img_s)
                img_s = blur(img_s, p=0.5)
                img_s2[b] = transforms.ToTensor()(img_s)
                
            encoder_out_s1 = model_encoder(img_s1)
            outputs1_s1 = model_decoder1(encoder_out_s1[1], encoder_out_s1[3], encoder_out_s1[5])
            outputs_soft1_s1 = torch.softmax(outputs1_s1, dim=1)
            encoder_out_s2 = model_encoder(img_s2)
            outputs2_s2 = model_decoder2(encoder_out_s2[0], encoder_out_s2[2], encoder_out_s2[4])
            outputs_soft2_s2 = torch.softmax(outputs2_s2, dim=1)

            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss1_s1 = 0.5 * (ce_loss(outputs1_s1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft1_s1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2_s1 = 0.5 * (ce_loss(outputs2_s2[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft2_s2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))

            pseudo_outputs1 = torch.argmax(outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs1_s1 = torch.argmax(outputs_soft1_s1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2_s2 = torch.argmax(outputs_soft2_s2[args.labeled_bs:].detach(), dim=1, keepdim=False)

            pseudo_supervision1 = ce_loss(outputs1[args.labeled_bs:], pseudo_outputs2)
            pseudo_supervision2 = ce_loss(outputs2[args.labeled_bs:], pseudo_outputs1)
            pseudo_supervision1_s1 = ce_loss(outputs1_s1[args.labeled_bs:], pseudo_outputs2)
            pseudo_supervision2_s2 = ce_loss(outputs2_s2[args.labeled_bs:], pseudo_outputs1)
            pseudo_supervision_s1 = ce_loss(outputs1_s1[args.labeled_bs:], pseudo_outputs2_s2)
            pseudo_supervision_s2 = ce_loss(outputs2_s2[args.labeled_bs:], pseudo_outputs1_s1)
            self_supervision1_s1 = ce_loss(outputs1_s1[args.labeled_bs:], pseudo_outputs1)
            self_supervision2_s2 = ce_loss(outputs2_s2[args.labeled_bs:], pseudo_outputs2)

            model1_loss = 0.5*(loss1+loss1_s1) + consistency_weight * (
                    pseudo_supervision1_s1
                    + pseudo_supervision_s1
                    + pseudo_supervision1
                    +self_supervision1_s1
            )
            model2_loss = 0.5*(loss2+loss2_s1) + consistency_weight * (
                    pseudo_supervision2_s2
                    +pseudo_supervision_s2
                    +pseudo_supervision2
                    +self_supervision2_s2
            )

            loss = model1_loss + model2_loss

            optimizer_encoder.zero_grad()
            optimizer_decoder1.zero_grad()
            optimizer_decoder2.zero_grad()

            loss.backward()

            optimizer_encoder.step()
            optimizer_decoder1.step()
            optimizer_decoder2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer_encoder.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer_decoder1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer_decoder2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (iter_num, model1_loss.item(), model2_loss.item()))
            if iter_num % 50 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model_encoder.eval()
                model_decoder1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i= test_myHiFormer_single_volume(sampled_batch["image"], sampled_batch["label"], [model_encoder,model_decoder1], classes=num_classes,patch_size=args.patch_size, option=[1,3,5])
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model_decoder1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best_decoder1 = os.path.join(snapshot_path,
                                             '{}_best_model_decoder1.pth'.format(args.model_decoder))
                    save_best_encoder1 = os.path.join(snapshot_path,
                                             '{}_best_model_encoder1.pth'.format(args.model_decoder))
                    torch.save(model_encoder.state_dict(), save_best_encoder1)
                    torch.save(model_decoder1.state_dict(), save_best_decoder1)
                    torch.save(model_decoder1.state_dict(), save_mode_path)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model_decoder1.train()

                model_decoder2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_myHiFormer_single_volume(
                        sampled_batch["image"], sampled_batch["label"], [model_encoder,model_decoder2], classes=num_classes,option=[0,2,4])
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95', mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model_decoder2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2)))
                    save_best_decoder2 = os.path.join(snapshot_path,
                                             '{}_best_model_decoder2.pth'.format(args.model_decoder))
                    save_best_encoder2 = os.path.join(snapshot_path,
                                                      '{}_best_model_encoder2.pth'.format(args.model_decoder))
                    torch.save(model_encoder.state_dict(), save_best_encoder2)
                    torch.save(model_decoder2.state_dict(), save_best_decoder2)
                    torch.save(model_decoder2.state_dict(), save_mode_path)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model_encoder.train()
                model_decoder2.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model_encoder_iter_' + str(iter_num) + '.pth')
                torch.save(model_encoder.state_dict(), save_mode_path)
                logging.info("save model_encoder to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model_decoder1_iter_' + str(iter_num) + '.pth')
                torch.save(model_decoder1.state_dict(), save_mode_path)
                logging.info("save model_decoder1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model_decoder2_iter_' + str(iter_num) + '.pth')
                torch.save(model_decoder2.state_dict(), save_mode_path)
                logging.info("save model_decoder2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, "myHiformer")
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)

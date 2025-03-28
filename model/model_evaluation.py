from loss import get_metrics, calc_loss
import numpy as np
from dataset import WSSegmentation
from model import U_NeXt_v1, U_NeXt_v2
from SegNet import SegNet
from proto_model import SSLSE
import torch
import os
import json

def evaluate_model(model, dataloader, device, paras_path=None):
    model_name = model.__class__.__name__
    if paras_path is None:
        paras_path = f'./exp/{model_name}/' + 'best_sslse_epoch_' + str(200)+ '_batchsize_' + str(batch_size) + '.pth'

    model.load_state_dict(torch.load(paras_path))
    model.eval()
    model.to(device)
    PAs = []
    Precisions = []
    Recalls = []
    F1s = []
    ious = []
    dices = []
    valid_loss = 0.0
    train_loss = 0.0
    with torch.no_grad():  
        for x1, y1 in dataloader:
            x1, y1 = x1.to(device), y1.to(device)

            y_pred1 = model(x1)
            lossL = calc_loss(y_pred1, y1)     # Dice_loss Used

            PA, Precision, Recall, F1, iou, dice = get_metrics(y_pred1, y1)
            PAs.append(PA.cpu())
            Precisions.append(Precision.cpu())
            Recalls.append(Recall.cpu())
            F1s.append(F1.cpu())
            ious.append(iou.cpu())
            dices.append(dice.cpu())  # 计算指标

            valid_loss += lossL.item() * x1.size(0)
            x_size1 = lossL.item() * x1.size(0)
        

    #######################################################
    # To write in Tensorboard
    #######################################################

    train_loss = train_loss / len(train_dataset)
    valid_loss = valid_loss / len(val_dataset)
    PAs = np.array(PAs).mean()
    Precisions = np.array(Precisions).mean()
    Recalls = np.array(Recalls).mean()
    F1s = np.array(F1s).mean()
    ious = np.array(ious).mean()
    dices = np.array(dices).mean()

    print(f'model: {model_name}')
    print('Training Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss, valid_loss))
    print('PA: {:.6f} \tPrecision: {:.6f} \tRecall: {:.6f} \tF1: {:.6f} \tIoU: {:.6f} \tDice: {:.6f}'.format(PAs, Precisions, Recalls, F1s, ious, dices))

    # 检查文件是否存在以及是否包含相同的模型信息
    if os.path.exists("model_info.json") and os.path.getsize("model_info.json") > 0:
        try:
            with open("model_info.json", "r") as f:
                existing_info = json.load(f)
        except json.JSONDecodeError:
            existing_info = {}
    else:
        existing_info = {}

    # 更新现有信息或添加新信息
    if model_name in existing_info:
        if "evaluation_metrics" in existing_info[model_name]:
            model_info = existing_info[model_name]["evaluation_metrics"]
        else:
            existing_info[model_name]["evaluation_metrics"] = {}
            model_info = {}
    else:
        existing_info[model_name] = {}
        existing_info[model_name]["evaluation_metrics"] = {}
        model_info = {}

    # 更新评估指标
    model_info.update({
        "PA": round(float(PAs), 6),
        "Precision": round(float(Precisions), 6),
        "Recall": round(float(Recalls), 6),
        "F1": round(float(F1s), 6),
        "IoU": round(float(ious), 6),
        "Dice": round(float(dices), 6),
        "train_loss": round(float(train_loss), 6),
        "valid_loss": round(float(valid_loss), 6)
    })

    existing_info[model_name]["evaluation_metrics"] = model_info

    # 将字典写入文件
    with open("model_info.json", "w") as f:
        json.dump(existing_info, f, indent=4)
    

if __name__=='__main__':

    model_name = input("请输入模型名称：")
    path = None
    if model_name == 'SSLSE':
        model = SSLSE()
        path = r'C:/Users/chenj/Desktop/训练数据/原模型训练数据_石/sslse_epoch_200_batchsize_8.pth'

    elif model_name == 'U_NeXt_v1':
        model = U_NeXt_v1(in_channels=1, out_channels=1)
    elif model_name == 'U_NeXt_v2':
        model = U_NeXt_v2()
    elif model_name == 'SegNet':
        model = SegNet(num_classes=1)
    else:
        print("模型名称输入错误！")
        exit(0)
    # model = SSLSE()
    # model = U_NeXt_v1()
    # model = U_NeXt_v2()
    batch_size = 8
    num_workers = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #######################################################
    # Dataset and Dataloader
    #######################################################

    train_dataset = WSSegmentation(r"C:\Users\chenj\Desktop\曾国豪毕业资料\毕业设计试验程序及结果\dataset\WeldSeam\WeldSeam",
                                train=True,    
                                txt_name="train.txt")
    val_dataset = WSSegmentation(r"C:\Users\chenj\Desktop\曾国豪毕业资料\毕业设计试验程序及结果\dataset\WeldSeam\WeldSeam",
                                    train=False,   
                                    txt_name="val.txt")

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    shuffle=True,
                                    pin_memory=True,
                                    collate_fn=train_dataset.collate_fn)
    valid_loader = torch.utils.data.DataLoader(val_dataset,
                                    batch_size=1,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    collate_fn=val_dataset.collate_fn)
    
    evaluate_model(model, valid_loader, device, paras_path=path)



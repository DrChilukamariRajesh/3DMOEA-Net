import os
import json
import time
import torch
import numpy as np
from monai import data
from monai import transforms
from datetime import datetime
from .utils import *
from monai.data import decollate_batch


def datafold_read(datalist, basedir, fold=0, key="training"):

    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)
    
    return tr, val


def save_checkpoint(model, epoch, root_dir, filename="model.pt", best_acc=0):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    # filename = os.path.join(root_dir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)
    
    
## Setup dataloader
def get_loader(batch_size, data_dir, json_list, fold, roi, nuw):
    data_dir = data_dir
    datalist_json = json_list
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=fold)
    print(f"Train size {len(train_files)}, Validation size {len(validation_files)}")
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[roi[0], roi[1], roi[2]],
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[roi[0], roi[1], roi[2]],
                random_size=False,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    train_ds = data.Dataset(data=train_files, transform=train_transform)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nuw,
        pin_memory=True,
    )
    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    val_loader = data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nuw,
        pin_memory=True,
    )
    return train_loader, val_loader


## Define Train Epoch
def train_epoch(model, loader, batch_size, optimizer, epoch, max_epochs, loss_func, device):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        run_loss.update(loss.item(), n=batch_size)
        print(
            "Epoch {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )
        start_time = time.time()
    return run_loss.avg


## Define Validation Epoch
def val_epoch(
    model,
    loader,
    epoch,
    max_epochs, 
    acc_func,
    hdm_func,
    device,
    model_inferer=None,
    post_sigmoid=None,
    post_pred=None,
):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()
    run_hdm = AverageMeter()
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor))
                for val_pred_tensor in val_outputs_list]
            
            #dice
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            dice_tc = run_acc.avg[0]
            dice_wt = run_acc.avg[1]
            dice_et = run_acc.avg[2]

   

            #hdm
            hdm_func.reset()
            hdm_func(y_pred=val_output_convert, y=val_labels_list)
            hd, not_nans = hdm_func.aggregate()
            run_hdm.update(hd.cpu().numpy(), n=not_nans.cpu().numpy())
            hdm_tc = run_hdm.avg[0]
            hdm_wt = run_hdm.avg[1]
            hdm_et = run_hdm.avg[2]
            
            print(
                "Val {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                ", dice_tc:", dice_tc,  ", dice_wt:", dice_wt, ", dice_et:", dice_et,
                ", hdm_tc:", hdm_tc,  ", hdm_wt:", hdm_wt, ", hdm_et:", hdm_et,
                ", time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()
    return run_acc.avg, run_hdm.avg



## Trainer
def trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    hdm_func,
    scheduler,
    root_dir, 
    checkpoint_name,
    device,
    batch_size,
    model_inferer=None,
    start_epoch=0,
    max_epochs=40, 
    val_every=4,
    post_sigmoid=None,
    post_pred=None,
):

    val_acc_max = 0.0
    best_metric_epoch = -1
    dices_tc = []
    dices_wt = []
    dices_et = []
    dices_avg = []
    hdms_tc = []
    hdms_wt = []
    hdms_et = []
    hdms_avg = []
    loss_epochs = []
    trains_epoch = []
    for epoch in range(start_epoch, max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            batch_size,
            optimizer,
            epoch=epoch,
            max_epochs=max_epochs, 
            loss_func=loss_func,
            device=device,
        )
        print(
            "Final training  {}/{}".format(epoch, max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )

        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()
            val_acc, val_hdm = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                max_epochs=max_epochs, 
                acc_func=acc_func,
                hdm_func=hdm_func,
                device=device,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )
            dice_tc = val_acc[0]
            dice_wt = val_acc[1]
            dice_et = val_acc[2]
            hdm_tc = val_hdm[0]
            hdm_wt = val_hdm[1]
            hdm_et = val_hdm[2]
            val_avg_acc = np.mean(val_acc)            
            val_avg_hdm = np.mean(val_hdm)
            print("Final validation stats {}/{}".format(epoch, max_epochs - 1),
                  ", dice_tc:", dice_tc, ", dice_wt:", dice_wt, ", dice_et:", dice_et, ", Dice_Avg:", val_avg_acc, 
                  ", hdm_tc:", hdm_tc, ", hdm_wt:", hdm_wt, ", hdm_et:", hdm_et, ", Dice_Avg:", val_avg_hdm, 
                  ", time {:.2f}s".format(time.time() - epoch_time),)
            dices_tc.append(dice_tc)
            dices_wt.append(dice_wt)
            dices_et.append(dice_et)
            dices_avg.append(val_avg_acc)
            hdms_tc.append(hdm_tc)
            hdms_wt.append(hdm_wt)
            hdms_et.append(hdm_et)
            hdms_avg.append(val_avg_hdm)
            if val_avg_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                best_metric_epoch = epoch
                save_checkpoint(
                    model,
                    epoch,
                    root_dir,
                    filename=checkpoint_name,
                    best_acc=val_acc_max,
                )
            scheduler.step()
            if val_acc_max < 0.15 and epoch>=15 and val_avg_acc < 0.15:
                print("Early stopping as val_avg_acc < 0.15")
                break
    print("Training Finished !, Best Accuracy: ", val_acc_max)
    return (
        val_acc_max,
        best_metric_epoch,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        hdms_tc,
        hdms_wt,
        hdms_et,
        hdms_avg,
        loss_epochs,
        trains_epoch
    )


def conv_json(data_dir, mode, case_num):
    return [
    {
        "image": [
            os.path.join(
                data_dir, mode,
                "BraTS2021_"
                + case_num
                + "/BraTS2021_"
                + case_num
                + "_flair.nii.gz",
            ),
            os.path.join(
                data_dir, mode,
                "BraTS2021_"
                + case_num
                + "/BraTS2021_"
                + case_num
                + "_t1ce.nii.gz",
            ),
            os.path.join(
                data_dir, mode,
                "BraTS2021_"
                + case_num
                + "/BraTS2021_"
                + case_num
                + "_t1.nii.gz",
            ),
            os.path.join(
                data_dir, mode,
                "BraTS2021_"
                + case_num
                + "/BraTS2021_"
                + case_num
                + "_t2.nii.gz",
            ),
        ],
        "label": os.path.join(
            data_dir, mode,
            "BraTS2021_"
            + case_num
            + "/BraTS2021_"
            + case_num
            + "_seg.nii.gz",
        ),
    }
]
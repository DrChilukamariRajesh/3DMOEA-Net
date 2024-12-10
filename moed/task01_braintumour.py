import torch
import json
import numpy as np
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandAffined,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.inferers import sliding_window_inference

VAL_AMP = True

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d

# define inference method
def inference(input, model):
    def _compute(input, model):
        return sliding_window_inference(
            inputs=input,
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )
    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input, model)
    else:
        return _compute(input, model)
    

def datafold_read(data_dir, file_name, k):
    with open(file_name) as f:
        json_data = json.load(f)

    train_files = [data_dir+'/all/images/'+i for i in json_data['folds'][k]["Train"]]
    train_labels = [data_dir+'/all/labels/'+i for i in json_data['folds'][k]["Train"]]

    val_images = [data_dir+'/all/images/'+i for i in json_data['folds'][k]["Val"]]
    val_labels = [data_dir+'/all/labels/'+i for i in json_data['folds'][k]["Val"]]

    test_images = [data_dir+'/all/images/'+i for i in json_data['folds'][k]["Test"]]
    test_labels = [data_dir+'/all/labels/'+i for i in json_data['folds'][k]["Test"]]
    return (train_files, train_labels), (val_images, val_labels), (test_images, test_labels)


def transform_train(img_size):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(keys=["image", "label"], roi_size=[img_size, img_size, img_size], \
                             random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )

    
def transform_val(img_shape):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandAffined(
                keys=['image', 'label'],
                mode=('bilinear', 'nearest'),
                prob=1.0, spatial_size=img_shape,
                rotate_range=(0, 0, np.pi/15),
                scale_range=(0.1, 0.1, 0.1)),
        ]
    )


def transform_test_org(img_shape):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandAffined(
                keys=['image', 'label'],
                mode=('bilinear', 'nearest'),
                prob=1.0, spatial_size=img_shape,
                rotate_range=(0, 0, np.pi/15),
                scale_range=(0.1, 0.1, 0.1)),
        ]
    )


def transform_post(test_org_transforms):
    return Compose([
        Invertd(
            keys="pred",
            transform=test_org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            device="cpu",
        ),
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", threshold=0.5),
    ])




## Testing
def Test_model_True(model):    
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    hdm = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean")
    hdm_batch = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")    
    mean_iou = MeanIoU(include_background=True, reduction="mean")   
    with torch.no_grad():
        for test_data in test_org_loader:
            test_inputs, test_labels = (
                test_data["image"].to(device),
                test_data["label"].to(device),
            )
            test_outputs = inference(test_inputs, model)
            test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
            # compute metric for current iteration
            dice_metric(y_pred=test_outputs, y=test_labels)
            dice_metric_batch(y_pred=test_outputs, y=test_labels)
            hdm(y_pred=test_outputs, y=test_labels)
            hdm_batch(y_pred=test_outputs, y=test_labels)            
            mean_iou(y_pred=test_outputs, y=test_labels)            
        # aggregate the final mean result
        metric_org = dice_metric.aggregate().item()
        metric_batch_org = dice_metric_batch.aggregate()
        hmetric_org = hdm.aggregate().item()
        hmetric_batch_org = hdm_batch.aggregate()
        iou_metric_org = mean_iou.aggregate().item()        
        dice_metric.reset()
        dice_metric_batch.reset()
        hdm.reset()
        hdm_batch.reset()
        mean_iou.reset()
    metric_tc, metric_wt, metric_et = metric_batch_org[0].item(),metric_batch_org[1].item(),metric_batch_org[2].item()
    hmetric_tc, hmetric_wt, hmetric_et = hmetric_batch_org[0].item(), hmetric_batch_org[1].item(), \
    hmetric_batch_org[2].item()    
    print(f"Dice: {metric_org}, tc: {metric_tc:.4f}, wt: {metric_wt:.4f}, et: {metric_et:.4f}, \
            HDM: {hmetric_org}, tc: {hmetric_tc:.4f}, wt: {hmetric_wt:.4f}, et: {hmetric_et:.4f}, \
            IOU: {iou_metric_org}")    
    return [metric_org, metric_tc, metric_wt, metric_et], [hmetric_org, hmetric_tc, hmetric_wt, hmetric_et], iou_metric_org



## Testing
def Test_model_False(model):
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")
    hdm = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    hdm_batch = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch")
    mean_iou = MeanIoU(include_background=False, reduction="mean")    
    with torch.no_grad():
        for test_data in test_org_loader:
            test_inputs, test_labels = (
                test_data["image"].to(device),
                test_data["label"].to(device),
            )
            test_outputs = inference(test_inputs, model)
            test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]            
            # compute metric for current iteration
            dice_metric(y_pred=test_outputs, y=test_labels)
            dice_metric_batch(y_pred=test_outputs, y=test_labels)
            hdm(y_pred=test_outputs, y=test_labels)
            hdm_batch(y_pred=test_outputs, y=test_labels)
            mean_iou(y_pred=test_outputs, y=test_labels)            
        # aggregate the final mean result
        metric_org = dice_metric.aggregate().item()
        metric_batch_org = dice_metric_batch.aggregate()
        hmetric_org = hdm.aggregate().item()
        hmetric_batch_org = hdm_batch.aggregate()
        iou_metric_org = mean_iou.aggregate().item()        
        # reset the status for next validation round
        dice_metric.reset()
        dice_metric_batch.reset()
        hdm.reset()
        hdm_batch.reset()
        mean_iou.reset()        
    metric_wt, metric_et = metric_batch_org[0].item(), metric_batch_org[1].item()
    hmetric_wt, hmetric_et = hmetric_batch_org[0].item(), hmetric_batch_org[1].item()    
    print(f"Dice: {metric_org}, wt: {metric_wt:.4f}, et: {metric_et:.4f} \
          HDM: {hmetric_org}, wt: {hmetric_wt:.4f}, et: {hmetric_et:.4f} \
          IOU: {iou_metric_org}")
    return [metric_org, metric_wt, metric_et], [hmetric_org, hmetric_wt, hmetric_et], iou_metric_org


# Saving the test image results
def save_results(model, res_save_path):
    with torch.no_grad():
        for k in [5,13,16,60]:
            print(k)
            test_input = test_org_ds[k]["image"].unsqueeze(0).to(device)
            roi_size = (128, 128, 64)
            sw_batch_size = 4
            test_output = inference(test_input, model)
            test_output = post_trans(test_output[0])        
            slice_num = int(test_org_ds[k]["label"].shape[-1] /2)
            for i in range(4):
                plt.imsave(res_save_path+"/"+str(k)+"_image"+str(i)+".png", 
                           test_org_ds[k]["image"][i, :, :, slice_num].detach().cpu(), cmap="gray")
            for i in range(3):
                plt.imsave(res_save_path+"/"+str(k)+"_label"+str(i)+".png", 
                           test_org_ds[k]["label"][i, :, :, slice_num].detach().cpu(), cmap="gray")
                cmap = "gray"
                plt.imsave(res_save_path+"/"+str(k)+"_output"+str(i)+".png",  
                           test_output[i, :, :, slice_num].detach().cpu(), cmap=cmap)


def evaluate_save_model(model, model_name):
    checkpoint_name = model_name+"_"+dataset_name+"_best_metric.pth"
    checkpoint_path = os.path.join('comparison', 'files', checkpoint_name)
    print(checkpoint_path)

    dice1, hdm1, iou1 = Test_model_True(model, checkpoint_path)
    dice2, hdm2, iou2  = Test_model_False(model, checkpoint_path)

    l=[]
    l.extend([model_name, cal_params(model), dice1[0], dice1[1], dice1[2], dice1[3], dice2[0], dice2[1], dice2[2], hdm1[0], hdm1[1], hdm1[2], hdm1[3], hdm2[0], hdm2[1], hdm2[2], iou1, iou2, task_name])
    with open(file_name_res,'a') as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(l)

    res_save_path = 'comparison/'+dataset_name+'_'+model_name
    create_dir(res_save_path)
    save_results(model, res_save_path)
    
 
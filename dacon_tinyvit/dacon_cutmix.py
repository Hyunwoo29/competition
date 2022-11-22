import random
import pandas as pd
import numpy as np
import os
import cv2
import math
import multiprocessing
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.optim import lr_scheduler
from torch.autograd import Variable
from Cream.TinyViT.models.tiny_vit import tiny_vit_21m_224, tiny_vit_21m_384
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
from sklearn import preprocessing
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torchvision.models as models

from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings(action='ignore') 

device = torch. device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'IMG_SIZE':384,
    'EPOCHS':10,
    'LEARNING_RATE':1e-4,
    'BATCH_SIZE':4,
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

train_df['암의 장경'] = train_df['암의 장경'].fillna(train_df['암의 장경'].mean())
train_df = train_df.fillna(0.1)

test_df['암의 장경'] = test_df['암의 장경'].fillna(train_df['암의 장경'].mean())
test_df = test_df.fillna(0.1)

le = preprocessing.LabelEncoder()
train_df['N_category'] = le.fit_transform(train_df['N_category'].values)
df = train_df
train_df, val_df, _, _ = train_test_split(df, df['N_category'].values, test_size=0.2, random_state=CFG['SEED'])
# train_df, val_df, train_labels, val_labels = train_test_split(
#                                                     train_df.drop(columns=['N_category']), 
#                                                     train_df['N_category'], 
#                                                     test_size=0.2, 
#                                                     random_state=CFG['SEED']
#                                                 )
def get_data(df, infer=False):
    if infer:
        return df['img_path'].values
    return df['img_path'].values, df['N_category'].values

train_img_paths, train_labels = get_data(train_df)
val_img_paths, val_labels = get_data(val_df)
all_img_paths, all_labels = get_data(df)
# def get_values(value):
#     return value.values.reshape(-1, 1)

# numeric_cols = ['나이', '암의 장경', 'ER_Allred_score', 'PR_Allred_score', 'KI-67_LI_percent', 'HER2_SISH_ratio']
# ignore_cols = ['ID', 'img_path', 'mask_path', '수술연월일', 'N_category']

# for col in train_df.columns:
#     if col in ignore_cols:
#         continue
#     if col in numeric_cols:
#         scaler = StandardScaler()
#         train_df[col] = scaler.fit_transform(get_values(train_df[col]))
#         val_df[col] = scaler.transform(get_values(val_df[col]))
#         test_df[col] = scaler.transform(get_values(test_df[col]))
#     else:
#         le = LabelEncoder()
#         train_df[col] = le.fit_transform(get_values(train_df[col]))
#         val_df[col] = le.transform(get_values(val_df[col]))
#         test_df[col] = le.transform(get_values(test_df[col]))
class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, transforms=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths['img_path'].iloc[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        if self.labels is not None:
            label = self.labels[index]
            return image, label
        else:
            return image
    
    def __len__(self):
        return len(self.img_paths)
# class CustomDataset(Dataset):
#     def __init__(self, medical_df, labels, transforms=None):
#         self.medical_df = medical_df
#         self.transforms = transforms
#         self.labels = labels
        
#     def __getitem__(self, index):
#         img_path = self.medical_df['img_path'].iloc[index]
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         if self.transforms is not None:
#             image = self.transforms(image=image)['image']
                
#         if self.labels is not None:
#             tabular = torch.Tensor(self.medical_df.drop(columns=['ID', 'img_path', 'mask_path', '수술연월일']).iloc[index])
#             label = self.labels[index]
#             return image, tabular, label
#         else:
#             tabular = torch.Tensor(self.medical_df.drop(columns=['ID', 'img_path', '수술연월일']).iloc[index])
#             return image, tabular
        
from numpy.lib.function_base import angle
train_transform = A.Compose([
                            A.Resize(800,800), 
                            A.RandomCrop(384,384), # 약 4분의1 size로 crop
                            A.HorizontalFlip(p=0.5),
                            # A.RandomRotate90(p=0.5),
                            # A.VerticalFlip(p=0.5),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])
val_transform = A.Compose([
                            A.Resize(800,800),  # 약 4분의1 size로 crop
                            A.RandomCrop(384,384),

                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

test_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),  # test와 val transform 분리
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

## cutmix
def cutmix(batch, alpha=1.0):
    '''
    alpha 값을 1.0으로 설정하여 beta 분포가 uniform 분포가 되도록 함으로써, 
    두 이미지를 랜덤하게 combine하는 Cutmix
    '''
    data, targets = batch
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets = (targets, shuffled_targets, lam)
    
    return data, targets


class CutMixCollator:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = cutmix(batch, self.alpha)
        return batch


class CutMixCriterion:
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss().to(device)

    def __call__(self, preds, targets):
        targets1, targets2, lam = targets
        return lam * self.criterion(
            preds, targets1) + (1 - lam) * self.criterion(preds, targets2)
        
collator = CutMixCollator(alpha=1.0)

train_dataset = CustomDataset(train_df, train_labels, train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=multiprocessing.cpu_count() // 2, collate_fn = collator)

val_dataset = CustomDataset(val_df, val_labels, val_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=multiprocessing.cpu_count() // 2)

class BaseModel(nn.Module):
    def __init__(self, num_classes=len(le.classes_)):
        super(BaseModel, self).__init__()
        self.backbone = tiny_vit_21m_384(pretrained=True) # backbone 모델을 tiny_vit384로 설정
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    

train_transform_dataset = CustomDataset(all_img_paths, all_labels, train_transform)
val_transform_dataset = CustomDataset(all_img_paths, all_labels, val_transform)

test_df = pd.read_csv('test.csv')
test_img_paths = get_data(test_df, infer=True)
test_dataset = CustomDataset(test_img_paths, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=multiprocessing.cpu_count() // 2)

def getDataloader(train_transform_dataset, val_transform_dataset, train_idx, valid_idx, batch_size, num_workers, collator):

    # train_transform이 적용된 train_transform_dataset에서 train_idx에 해당하는 Subset 추출
    train_set = torch.utils.data.Subset(train_transform_dataset,
                                        indices=train_idx)
    # val_transform이 적용된 val_transform_dataset에서 valid_idx에 해당하는 Subset 추출
    val_set   = torch.utils.data.Subset(val_transform_dataset,
                                        indices=valid_idx)
    
    # 추출된 Train Subset으로 DataLoader 생성
    train_loader = DataLoader(train_dataset, 
                              batch_size = batch_size, 
                              shuffle=True, 
                              num_workers=num_workers, 
                              collate_fn = collator)
    # 추출된 Valid Subset으로 DataLoader 생성
    val_loader = DataLoader(val_dataset, 
                            batch_size= batch_size, 
                            shuffle=False, 
                            num_workers=num_workers)
    
    # 생성한 DataLoader 반환
    return train_loader, val_loader

def train_stratified_kfold(device):

    
    n_splits = 7  # 7 fold
    skf = StratifiedKFold(n_splits=n_splits)
    labels = train_df.N_category.to_list()

    train_criterion = CutMixCriterion()
    val_criterion = nn.CrossEntropyLoss().to(device)
    
    patience = 10  # 10 epoch동안 성능 향상 없을 시, early stopping
    oof_pred = None
    
    for i, (train_idx, valid_idx) in enumerate(skf.split(train_df.img_path.to_list(), labels)):
        
        num_workers=multiprocessing.cpu_count() // 2
        train_loader, val_loader = getDataloader(train_transform_dataset, val_transform_dataset, train_idx, valid_idx, CFG['BATCH_SIZE'], num_workers, collator)
            
        model = BaseModel()
        model.to(device)
        
        optimizer = torch.optim.Adam(params = model.parameters(), lr = 0)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=1, eta_max=0.001,  T_up=3, gamma=0.5)
        
        best_score = 0
        best_model = None
        counter = 0
        
        for epoch in range(1,CFG["EPOCHS"]+1):
            model.train()
            train_loss = []
            for img, label in tqdm(iter(train_loader)):
                img = img.float().to(device)
                targets1, targets2, lam = label  # cutmix 사용하기에 label split
                label = (targets1.to(device), targets2.to(device), lam)

                optimizer.zero_grad()

                model_pred = model(img)

                loss = train_criterion(model_pred, label)

                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

            tr_loss = np.mean(train_loss)

            val_loss, val_score = validation(model, val_criterion, val_loader, device)

            print(f'Epoch [{epoch}], Train Loss : [{tr_loss:.5f}] Val Loss : [{val_loss:.5f}] Val F1 Score : [{val_score:.5f}]')

            if scheduler is not None:
                scheduler.step()

            if best_score < val_score:
                best_model = model
                best_score = val_score
                counter=0
                # 갱신 시마다  best model 저장 -> fold별 마지막 weight이 fold별 best weight
                torch.save(model, f"weights/fold{i}_{epoch:03}_f1score_{val_score:4.2%}.pt")
            else:
                counter+=1
                
            if counter > patience:
                print("Early Stopping...")
                break
           
        all_predictions = []
        with torch.no_grad():
            for images in test_loader:
                
                images = images.float().to(device)
            
                # Test Time Augmentation
                pred = best_model(images) / 2  # 원본 이미지를 예측하고
                pred += best_model(torch.flip(images, dims=(-1,))) / 2  # horizontal_flip으로 뒤집어 예측하여 누적
                all_predictions.extend(pred.cpu().numpy())

            fold_pred = np.array(all_predictions)

        # OOF
        if oof_pred is None:
            oof_pred = fold_pred / n_splits
        else:
            oof_pred += fold_pred / n_splits
        
        
        oof_pred_list = []
        if i == n_splits-1:

            # 제출용 csv 생성
            oof_pred = torch.from_numpy(oof_pred)
            oof_pred_ans = oof_pred.argmax(dim=-1)
            oof_pred_ans = oof_pred_ans.detach().cpu().numpy().tolist()
            preds = le.inverse_transform(oof_pred_ans)
            submit = pd.read_csv('sample_submission.csv')
            submit['N_category'] = preds
            save_answer_path = './output/stratified_7fold_tta_answer.csv'
            submit.to_csv(save_answer_path, index=False)
            
            # ensemble을 위한 logit csv 생성
            oof_pred = oof_pred.detach().cpu().numpy().tolist()
            submit_logit = pd.read_csv('sample_submission.csv')
            submit_logit['N_category'] = oof_pred
            save_answer_path2 = './output/stratified_7fold_tta_logit.csv'
            submit_logit.to_csv(save_answer_path2, index=False)
            
            
            

            print(f"Inference Done! Inference result saved at {save_answer_path}")
            
def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")

def validation(model, criterion, val_loader, device):
    model.eval()
    
    model_preds = []
    true_labels = []
    
    val_loss = []
    
    with torch.no_grad():
        for img, label in tqdm(iter(val_loader)):
            img, label = img.float().to(device), label.to(device)
            
            model_pred = model(img)
            
            loss = criterion(model_pred, label)
            
            val_loss.append(loss.item())
            
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()
        
    val_f1 = competition_metric(true_labels, model_preds)
    return np.mean(val_loss), val_f1

class CosineAnnealingWarmUpRestarts(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (
                        1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

torch.cuda.empty_cache()

model = BaseModel()

model.eval()

optimizer = torch.optim.Adam(params = model.parameters(), lr = 0)
scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=1, eta_max=0.001,  T_up=3, gamma=0.5)            


if __name__ == '__main__':

    train_stratified_kfold(device)
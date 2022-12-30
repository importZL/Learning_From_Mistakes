from PIL import Image
import os
import os.path
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms


class DataIterator(object):
    def __init__(self, dataloader):
        assert isinstance(dataloader, torch.utils.data.DataLoader), 'Wrong loader type'
        self.loader = dataloader
        self.iterator = iter(self.loader)

    def __next__(self):
        try:
            x, y = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            x, y = next(self.iterator)

        return x, y

def _fix_cls_to_idx(ds):
    for cls in ds.class_to_idx:
        ds.class_to_idx[cls] = int(cls)

def prepare_data(args):
    num_classes = 14

    # resnet recommended normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # transform
    # Note: rescaling to 224 and center-cropping already processed in img folders    
    transform = transforms.Compose([
        transforms.ToTensor(), # to [0,1]
        normalize
    ])
    
    # train_data_gold = torchvision.datasets.ImageFolder('../data/clothing1M/clean_train', transform=transform)
    # train_data_silver = torchvision.datasets.ImageFolder('../data/clothing1M/noisy_train', transform=transform)

    train_data = torchvision.datasets.ImageFolder('../data/clothing1M/clothing1M/total_train', transform=transform)
    val_data = torchvision.datasets.ImageFolder('../data/clothing1M/clothing1M/clean_val', transform=transform)
    test_data = torchvision.datasets.ImageFolder('../data/clothing1M/clothing1M/clean_test', transform=transform)

    # fix class idx to equal to class name
    # _fix_cls_to_idx(train_data_gold)
    # _fix_cls_to_idx(train_data_silver)
    _fix_cls_to_idx(train_data)
    _fix_cls_to_idx(val_data)
    _fix_cls_to_idx(test_data)

    batch_size = args.batch_size
        
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                                          num_workers=args.prefetch, pin_memory=True, drop_last=True)
    val_loader  = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                              num_workers=args.prefetch, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                              num_workers=args.prefetch, pin_memory=True,)

    return train_loader, val_loader, test_loader


def create_img_folder(img_list, label_list, folder_name, root='../data/clothing1M/clothing1M'):
    # load label dict
    label_dict = {}
    with open(os.path.join(root, label_list), 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            label_dict[parts[0]] = int(parts[1])

    # following previous works on cloth1m
    preprocess_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
    ])

    img_tensor_list = []
    label_list = []
    file_list = []

    # create folder if not exist
    if not os.path.isdir(os.path.join(root, folder_name)):
        os.mkdir(os.path.join(root, folder_name))
    
    cnt = 0
    name_set = set()
    with open(os.path.join(root, img_list), 'r') as f:
        for line in f:
            img_name = line.strip()
            label = label_dict[img_name]

            out_dir = os.path.join(root, folder_name, str(label))
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)

            trailing_name = img_name.split('/')[-1]
            assert trailing_name not in name_set, 'Image duplicates!'
            name_set.add(trailing_name)
            
            with Image.open(os.path.join(root, img_name)) as img:
                processed_img = preprocess_transform(img)

            processed_img.save(os.path.join(out_dir, trailing_name))
            cnt += 1

            if (cnt % 10000 == 0):
                print ('%d images processed' % cnt)

    print ('In total: %d images processed.' % cnt)


if __name__ == '__main__':
    create_img_folder('noisy_train_key_list.txt', 'noisy_label_kv.txt', 'noisy_train', root='../data/clothing1M/clothing1M')
    create_img_folder('clean_train_key_list.txt', 'clean_label_kv.txt', 'clean_train', root='../data/clothing1M/clothing1M')
    create_img_folder('clean_val_key_list.txt',   'clean_label_kv.txt', 'clean_val', root='../data/clothing1M/clothing1M')
    create_img_folder('clean_test_key_list.txt',  'clean_label_kv.txt', 'clean_test', root='../data/clothing1M/clothing1M')
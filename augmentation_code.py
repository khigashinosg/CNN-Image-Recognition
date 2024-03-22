import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import sampler
from torchvision import datasets, transforms
import copy

ON_COLAB = False

# When we import the images we want to first convert them to a tensor. 
# It is also common in deep learning to normalise the the inputs. This 
# helps with stability.
# To read more about this subject this article is a great one:
# https://towardsdatascience.com/understand-data-normalization-in-machine-learning-8ff3062101f0

# transforms is a useful pytorch package which contains a range of functions
# for preprocessing data, for example applying data augmentation to images 
# (random rotations, blurring the image, randomly cropping the image). To find out
# more please refer to the pytorch documentation:
# https://pytorch.org/docs/stable/torchvision/transforms.html

mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])

# Original transforms given in the coursework
transform_original = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist()),
])

# Augmentation transforms to produce agumented data
transform_augmentation = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),  # some augmentation
    transforms.RandomRotation(10),  # random rotation  (-10 to 10 degrees)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # random color jitter
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist()),
])

train_path = ('/content/' if ON_COLAB else '') + 'NaturalImageNetTrain'
test_path = ('/content/' if ON_COLAB else '') +'NaturalImageNetTest'

train_dataset = datasets.ImageFolder(train_path, transform=transform_original) # Original dataset

test_dataset = datasets.ImageFolder(test_path, transform=transform_original)

# Create train val split
n = len(train_dataset)
n_val = int(n/10)

train_set_original, val_set = torch.utils.data.random_split(train_dataset, [n-n_val, n_val])

train_set_augmented = copy.deepcopy(train_set_original)
train_set_augmented.dataset.transform = transform_augmentation
extended_train_set = ConcatDataset([train_set_original, train_set_augmented]) # Combine datasets


print(len(extended_train_set), len(val_set), len(test_dataset))


# The number of images to process in one go. If you run out of GPU
# memory reduce this number! 
batch_size = 8

# Dataloaders are a great pytorch functionality for feeding data into our AI models.
# see https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader
# for more info.

loader_train = DataLoader(extended_train_set, batch_size=batch_size, shuffle=True, num_workers=2)
loader_val = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import random
import sys
import os


#################### CONSTANTS / VARIABLES ####################
EPOCHS = 50
TRAINING_DATASET_LENGTH = 60000
# Max 60,000 (e.g. 10000 means 10000 images will be used for training each separate net, per epoch)
NUM_TRAINING_IMGS = 20000
BATCH_SIZE = 50
NUM_IN_PAINT_EXAMPLES = 10
NUM_PIXELS_IN_IMAGE = 784
INDEX_FIRST_PIXEL_TO_PREDICT = 448
TOTAL_NUM_NETS = NUM_PIXELS_IN_IMAGE - INDEX_FIRST_PIXEL_TO_PREDICT
LR = 0.001
IMAGE_OUTPUT_PATH = './generated_images/'
FOLDER_NAME = IMAGE_OUTPUT_PATH.replace("./", "").replace("/", "")

start_index = 0


#################### CHECKING CUDA ####################
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using cuda...')
else:
    device = torch.device('cpu')
    print('Using cpu...')

# device = torch.device('cpu')
# print('Overriding to cpu...')


#################### RANGE CONVERSION FUNCTIONS ####################
# converts number in range of 0 to 1 to number in range of 0 to 255
def convert_0_1_to_0_255(x):
    return int(x * 255)


# converts number in range of 0 to 1 to number in range of 0 to 255
def convert_0_255_to_0_1(x):
    return x / 255


#################### DATASETS ####################
class CroppedMnistDatasetTrain(Dataset):
    """
    CroppedMnistDatasetTrain will make an object of cropped mnist digits and the next pixel,
    e.g. the first item may be the first 400 pixels of an image and the 401st pixel as the answer.
    """
    def __init__(self, pixels, start_idx):
        mnist_train = datasets.MNIST(root='./', train=True, transform=transforms.ToTensor(), download=True)

        self.mnist_train_cropped_imgs_capped = torch.zeros([NUM_TRAINING_IMGS, pixels], dtype=torch.float32).to(device)
        self.mnist_train_next_pixel_one_hots_capped = torch.zeros([NUM_TRAINING_IMGS, 256], dtype=torch.float32).to(device)

        for idx, (image, label) in enumerate(mnist_train):
            if idx < start_idx:
                continue

            if idx == start_idx + NUM_TRAINING_IMGS:
                break

            # Capped as it is not the whole dataset, only NUM_TRAINING_IMGS long
            index_for_capped_data = idx - start_idx

            self.mnist_train_cropped_imgs_capped[index_for_capped_data] = image.view(784)[:pixels].to(device)

            next_pixel = image.view(784)[pixels].to(device)
            next_pixel_0_255 = convert_0_1_to_0_255(float(next_pixel))
            self.mnist_train_next_pixel_one_hots_capped[index_for_capped_data] = f.one_hot(torch.tensor(next_pixel_0_255), 256).to(device)

    def __len__(self):
        return len(self.mnist_train_cropped_imgs_capped)

    def __getitem__(self, item):
        return self.mnist_train_cropped_imgs_capped[item], self.mnist_train_next_pixel_one_hots_capped[item]


class CroppedMnistDatasetTest(Dataset):
    """
    CroppedMnistDatasetTest will make an object of cropped mnist digits, e.g. a dataset of the first 400 pixels of the
    mnist digits in the test set
    """
    def __init__(self, pixels):
        mnist_test = datasets.MNIST(root='./', train=False, transform=transforms.ToTensor(), download=True)

        self.mnist_test_cropped_imgs = torch.zeros([len(mnist_test), pixels], dtype=torch.float32).to(device)

        for idx, (image, label) in enumerate(mnist_test):
            self.mnist_test_cropped_imgs[idx] = image.view(784)[:pixels].to(device)

    def __len__(self):
        return len(self.mnist_test_cropped_imgs)

    def __getitem__(self, item):
        return self.mnist_test_cropped_imgs[item]


#################### NET ####################
# class Net(nn.Module):
#     def __init__(self, in_nodes, out_nodes):
#         super().__init__()
#         self.l1 = nn.Linear(in_nodes, in_nodes, dtype=torch.float32)
#         self.l2 = nn.Linear(in_nodes, out_nodes, dtype=torch.float32)
#
#     def forward(self, x):
#         x = self.l1(x)
#         x = self.l2(x)
#         return f.softmax(x, dim=1)
class Net(nn.Module):
    def __init__(self, in_nodes, out_nodes):
        super().__init__()
        self.l1 = nn.Linear(in_nodes, 400, dtype=torch.float32)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(4, 4), stride=4, dtype=torch.float32)
        self.l2 = nn.Linear(256 * 5 * 5, out_nodes, dtype=torch.float32)

    def forward(self, x):
        x = self.l1(x)
        x = x.view(-1, 1, 20, 20)
        x = self.conv1(x)
        x = x.view(-1, 256 * 5 * 5)
        x = self.l2(x)
        return f.softmax(x, dim=1)


#################### WRITE PERMISSION ####################
user_agrees_to_output_path = input(f"Type 'y' if data in '{IMAGE_OUTPUT_PATH}' can be overwritten: ")

if user_agrees_to_output_path != 'y':
    sys.exit()
else:
    if os.name == 'nt':
        if os.path.exists(IMAGE_OUTPUT_PATH):
            os.system(f'rd /s /q {FOLDER_NAME}')
            os.system(f'md {FOLDER_NAME}')
        else:
            os.system(f'md {FOLDER_NAME}')
    elif os.name == 'posix':
        if os.path.exists(IMAGE_OUTPUT_PATH):
            os.system(f'rm -r {FOLDER_NAME}')
            os.system(f'mkdir {FOLDER_NAME}')
        else:
            os.system(f'mkdir {FOLDER_NAME}')
    else:
        print('OS not recognized, exiting...')
        sys.exit()


#################### NOTE ON TRAINING DATASETS AND DATALOADERS ####################
# NOTE: due to memory constraints train datasets and loaders are created during training
# (there is a separate dataset and loader for each net, e.g. one for 392 input pixels, 393, etc...)


#################### INSTANTIATE TEST DATASET AND DATALOADER ####################
test_dataset = CroppedMnistDatasetTest(INDEX_FIRST_PIXEL_TO_PREDICT)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=1)

num_test_imgs = len(test_dataset)

assert(NUM_IN_PAINT_EXAMPLES <= num_test_imgs, 'Number of requested in paint examples is higher than test dataset '
                                               'examples, please lower NUM_IN_PAINT_EXAMPLES')


#################### INSTANTIATE NETS AND OPTIMIZERS ####################
nets = []
optimizers = []

# List containing number of image pixels for 1st net, 2nd net, etc...
input_pixels_for_each_net = []
for i in range(TOTAL_NUM_NETS):
    input_pixels_for_each_net.append(INDEX_FIRST_PIXEL_TO_PREDICT + i)

# Makes nets to predict next pixel
# (e.g. first net may take in 100 pixels to predict the 101st pixel (starting from pixel number 1))
for i in range(TOTAL_NUM_NETS):
    net = Net(input_pixels_for_each_net[i], 256).to(device)

    optimizer = optim.AdamW(net.parameters(), lr=LR)

    nets.append(net)
    optimizers.append(optimizer)


#################### TRAINING AND TESTING ####################
for epoch in range(EPOCHS):
    if start_index + NUM_TRAINING_IMGS > TRAINING_DATASET_LENGTH:
        start_index = 0

    for i in range(TOTAL_NUM_NETS):
        train_dataset = CroppedMnistDatasetTrain(INDEX_FIRST_PIXEL_TO_PREDICT + i, start_index)
        num_batches = len(train_dataset) / BATCH_SIZE
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

        for X, y in train_loader:
            optimizers[i].zero_grad()

            num_of_pixels = input_pixels_for_each_net[i]

            output = nets[i](X)

            loss = f.mse_loss(output, y)
            loss.backward()

            optimizers[i].step()

            if torch.isnan(loss):
                raise Exception('MODEL IS RETURNING NANS')

        print(f'Epoch {epoch + 1} {(i + 1) / TOTAL_NUM_NETS * 100:.2F}% completed.', end='\r')

    print('')

    with torch.no_grad():
        for j, img in enumerate(test_loader):
            if j == NUM_IN_PAINT_EXAMPLES:
                break

            for i in range(TOTAL_NUM_NETS):
                predicted_pixel_brightness = convert_0_255_to_0_1(int(torch.argmax(nets[i](img)[0])))
                predicted_pixel_brightness = torch.tensor([predicted_pixel_brightness]).to(device)
                img = torch.cat((img[0], predicted_pixel_brightness)).view(1, INDEX_FIRST_PIXEL_TO_PREDICT + i + 1)

            plt.imsave(f'{IMAGE_OUTPUT_PATH}Epoch{epoch + 1}Example{j + 1}.png', img.to(torch.device('cpu')).view(28, 28))

    start_index += NUM_TRAINING_IMGS



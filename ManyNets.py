import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import os


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using cuda...')
else:
    device = torch.device('cpu')
    print('Using cpu...')

device = torch.device('cpu')
print('Overriding to cpu...')


class Net(nn.Module):
    def __init__(self, in_nodes, out_nodes):
        super().__init__()
        self.l1 = nn.Linear(in_nodes, in_nodes, dtype=torch.float32)
        self.l2 = nn.Linear(in_nodes, out_nodes, dtype=torch.float32)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return f.softmax(x, dim=0)


# converts number in range of 0 to 1 to number in range of 0 to 255
def convert_0_1_to_0_255(x):
    return int(x * 255)


# converts number in range of 0 to 1 to number in range of 0 to 255
def convert_0_255_to_0_1(x):
    return x / 255


#################### SETTING UP ####################
EPOCHS = 10
NUM_IN_PAINT_EXAMPLES = 10
NUM_PIXELS_IN_IMAGE = 784
# Starting at 0
INDEX_FIRST_PIXEL_TO_PREDICT = 392
TOTAL_NUM_NETS = NUM_PIXELS_IN_IMAGE - INDEX_FIRST_PIXEL_TO_PREDICT
LR = 0.001
IMAGE_OUTPUT_PATH = './generated_images/'
FOLDER_NAME = IMAGE_OUTPUT_PATH.replace("./", "").replace("/", "")

user_agrees_to_output_path = input(f"Type 'y' if data in '{IMAGE_OUTPUT_PATH}' can be overwritten: ")

if user_agrees_to_output_path != 'y':
    sys.exit()
else:
    if os.path.exists(IMAGE_OUTPUT_PATH):
        os.system(f'rd /s /q {FOLDER_NAME}')
        os.system(f'md {FOLDER_NAME}')
    else:
        os.system(f'md {FOLDER_NAME}')

mnist_train = datasets.MNIST(root='./', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root='./', train=False, transform=transforms.ToTensor(), download=True)

mnist_train_imgs = torch.zeros([len(mnist_train), 784], dtype=torch.float32).to(device)
mnist_test_imgs = torch.zeros([len(mnist_test), 784], dtype=torch.float32).to(device)

num_train_imgs = len(mnist_train_imgs)
num_test_imgs = len(mnist_test_imgs)

assert(NUM_IN_PAINT_EXAMPLES <= num_test_imgs, 'Number of requested in paint examples is higher than test dataset '
                                               'examples, please lower NUM_IN_PAINT_EXAMPLES')

for i, (X, y) in enumerate(mnist_train):
    mnist_train_imgs[i] = X.view(-1, 784)

for i, (X, y) in enumerate(mnist_test):
    mnist_test_imgs[i] = X.view(-1, 784)

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

    optimizer = optim.Adam(net.parameters(), lr=LR)

    nets.append(net)
    optimizers.append(optimizer)


#################### TRAINING AND TESTING ####################
for epoch in range(EPOCHS):
    for m, img in enumerate(mnist_train_imgs):
        for i in range(TOTAL_NUM_NETS):
            optimizers[i].zero_grad()

            num_of_pixels = input_pixels_for_each_net[i]
            cropped_img = img[:num_of_pixels]

            pixel_to_predict = img[num_of_pixels]
            pixel_to_predict_0_255 = convert_0_1_to_0_255(float(pixel_to_predict))

            output = nets[i](cropped_img)

            one_hot = f.one_hot(torch.tensor(pixel_to_predict_0_255), 256).float().to(device)

            loss = f.mse_loss(output, one_hot)
            loss.backward()

            optimizers[i].step()

        print(f'Epoch {epoch + 1} {(m + 1) / num_train_imgs * 100:.2F}% completed. Loss = {loss}.', end='\r')

    print('')

    with torch.no_grad():
        for j, img in enumerate(mnist_test_imgs):
            if j == NUM_IN_PAINT_EXAMPLES:
                break

            starting_num_of_pixels = input_pixels_for_each_net[0]
            in_paint_example = img[:starting_num_of_pixels]

            for i in range(TOTAL_NUM_NETS):
                predicted_pixel_brightness = convert_0_255_to_0_1(int(torch.argmax(nets[i](in_paint_example))))
                predicted_pixel_brightness = torch.tensor([predicted_pixel_brightness], dtype=torch.float32).to(device)
                in_paint_example = torch.cat((in_paint_example, predicted_pixel_brightness))

            plt.imsave(f'{IMAGE_OUTPUT_PATH}Epoch{epoch + 1}Example{j + 1}.png', in_paint_example.view(28, 28))







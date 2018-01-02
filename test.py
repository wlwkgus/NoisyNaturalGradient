import time
from data.data_loader import get_data_loader
from models.models import create_model
from option_parser import TrainingOptionParser, TestingOptionParser
from utils.visualizer import Visualizer
import torch

parser = TestingOptionParser()
opt = parser.parse_args()

data_loader = get_data_loader(opt)

print("[INFO] batch size : {}".format(opt.batch_size))
print("[INFO] training batches : {}".format(len(data_loader)))

model = create_model(opt)
model.load(opt.epoch)

total_steps = 0
corrects = 0

for i, data in enumerate(data_loader):
    batch_start_time = time.time()
    total_steps += opt.batch_size

    # data : list
    # TODO : The network I implemented only works in MNIST dataset.
    # TODO : Add more networks to benchmark.
    data[0] = data[0].view(opt.batch_size, -1)
    model.set_input(data)
    result = model.test()[1]
    corrects += torch.eq(result.cpu().data, data[1]).sum()

    batch_end_time = time.time()

print(corrects)
print(total_steps)

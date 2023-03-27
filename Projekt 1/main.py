#!/usr/bin/env python3
# in order to run tensorboard:
#tensorboard --logdir=tb-logger/ --host localhost --port 8088

import torch
import datetime

from utils import *
from networks import *


torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

BATCH_SIZE = 64
COMMITTEE_SIZE = 5
RUN_ID = datetime.datetime.now().strftime("%Y.%m.%d %H.%M")


train_loader, test_loader = get_loaders_single_model(BATCH_SIZE)

model = SimpleNet([3, 6, 16, 32, 64], [5, 5, 5, 5], [64 * 16 * 16, 120, 84, 32, 10])
run_model(model, train_loader, test_loader, draw=False, save=True, savefile="SimpleNet_kernel_5_Single", run_id=RUN_ID)

model = SimpleNet([3, 6, 16, 32, 64], [3, 3, 3, 3], [64 * 24 * 24, 120, 84, 32, 10])
run_model(model, train_loader, test_loader, draw=False, save=True, savefile="SimpleNet_kernel_3_Single", run_id=RUN_ID) 

model = PoolingNet([3, 6, 16, 32], [3, 3, 3, 3], [True, True, False], [32 * 4 * 4, 120, 84, 64, 32, 10])
run_model(model, train_loader, test_loader, draw=False, save=True, savefile="PoolingNet_Single", run_id=RUN_ID)

model = ResidualNet()
run_model(model, train_loader, test_loader, draw=False, save=True, savefile="ResidualNet_Single", run_id=RUN_ID)


train_loaders, test_loader = get_loaders_committee(BATCH_SIZE, COMMITTEE_SIZE)

models = [SimpleNet([3, 6, 16, 32, 64], [5, 5, 5, 5], [64 * 16 * 16, 120, 84, 32, 10]) for _ in range(COMMITTEE_SIZE)]
run_models(models, train_loaders, test_loader, draw=False, save=True, savefile="SimpleNet_kernel_5_Committee", run_id=RUN_ID)

models = [SimpleNet([3, 6, 16, 32, 64], [3, 3, 3, 3], [64 * 16 * 16, 120, 84, 32, 10]) for _ in range(COMMITTEE_SIZE)]
run_models(models, train_loaders, test_loader, draw=False, save=True, savefile="SimpleNet_kernel_3_Committee", run_id=RUN_ID)

models = [PoolingNet([3, 6, 16, 32], [3, 3, 3, 3], [True, True, False], [32 * 4 * 4, 120, 84, 64, 32, 10]) for _ in range(COMMITTEE_SIZE)]
run_models(models, train_loaders, test_loader, draw=False, save=True, savefile="PoolingNet_Committee", run_id=RUN_ID)

models = [ResidualNet() for _ in range(COMMITTEE_SIZE)]
run_models(models, train_loaders, test_loader, draw=False, save=True, savefile="ResidualNet_Single", run_id=RUN_ID)


train_loader, test_loader = get_loaders_augmented(BATCH_SIZE*2)

model = SimpleNet([3, 6, 16, 32, 64], [5, 5, 5, 5], [64 * 16 * 16, 120, 84, 32, 10])
run_model(model, train_loader, test_loader, draw=False, save=True, savefile="SimpleNet_kernel_5_Augmented", run_id=RUN_ID)

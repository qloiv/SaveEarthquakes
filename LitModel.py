from datetime import datetime

import pytorch_lightning as pl

from LitNetwork import Net
from load2 import *

catalog_path = "/home/viola/WS2021/Code/Daten/Chile_small/catalog_ma.csv"
waveform_path = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
model_path = "/home/viola/WS2021/Code/Models"

network = Net()
trainer = pl.Trainer(gpus=1)
trainer.fit(network)

trainer.test(test_dataloaders=network.test_dataloader)

now = datetime.now().strftime("%Y-%m-%d %H:%M")
path = 'GPD_net_' + str(now) + '.pth'
torch.save(network.state_dict(), os.path.join(model_path, path))

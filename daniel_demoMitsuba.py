import sys
import os
import torch
from optimizerMitsuba import OptimizerMitsuba
from optimizer import Optimizer
from config import Config

sys.path.insert(0,'/content/NextFace') #verify ur path
config = Config()
config.fillFromDicFile('./optimConfig.ini')
# config.device = 'cuda' # torch not compiled with cuda
config.path = './baselMorphableModel/' #verify ur path

imagePath = './input/detailled_faces_unsplash/BikerMan.jpg'
outputDir = './output/' + os.path.basename(imagePath.strip('/'))
torch.cuda.set_device(0)
optimizer = OptimizerMitsuba(outputDir ,config)

#run the optimization now 
optimizer.run(imagePath,doStep1=False,doStep2=True, doStep3=False)
# optimizer.run(imagePath,checkpoint=outputDir+'/checkpoints/stage1_output.pickle', doStep1=False,doStep2=True, doStep3=False)
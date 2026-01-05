from network.detector import Detector
from network.detector_dino import Detector as DetectorDINO
from network.refiner import VolumeRefiner
from network.selector import ViewpointSelector
from network.mvs2d_refiner import VolumeRefiner as MVS2DRefiner
name2network={
    'refiner': VolumeRefiner,
    'detector': Detector,
    'selector': ViewpointSelector,
    'detector_dino': DetectorDINO,  
    'mvs2d_refiner': MVS2DRefiner
}


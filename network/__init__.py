from network.detector import Detector
from network.refiner import VolumeRefiner
from network.selector import ViewpointSelector
from network.cascade_refiner import VolumeRefiner as cascadeVolumeRefiner

name2network={
    'refiner': VolumeRefiner,
    'detector': Detector,
    'selector': ViewpointSelector,
    'cascade_refiner':cascadeVolumeRefiner
}


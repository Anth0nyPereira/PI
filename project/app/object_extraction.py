## @package app
#  This module contains object extraction class
# that is used to extract objects
#  More details.
import torch
from app.utils import objectExtractionThreshold

## Class object extraction that contains the loaded model when created
#
#  More details.
class ObjectExtract:
    def __init__(self):
        # exige que o utilizador esteja ligado à net da primeira vez (ele faz uns downloads)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        """ # exemplo de como se faz pra n ter de tar ligado à net
        path = '/some/local/path/pytorch/vision'
        model = torch.hub.load(path, 'resnet50', pretrained=True)
        """

    ## Get objects from an image
    #
    #  More details.
    def get_objects(self, image_path):
        results = self.model(image_path)

        res = results.pandas().xyxy[0][['confidence', 'name']]
        return [ (res['name']   [i], res['confidence'][i]) for i in range(res.shape[0]) if res['confidence'][i] >= objectExtractionThreshold]

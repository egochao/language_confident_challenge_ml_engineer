import sys

sys.path.append("..")

from utils.model_utils import count_parameters

from models.mobile_vit import MobileViTModelCustom


model = MobileViTModelCustom()

print(count_parameters(model))

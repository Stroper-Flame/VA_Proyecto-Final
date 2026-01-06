import torch.nn as nn                                              #Usaremos Fcn_ResNet50 como modelo base.
from torchvision.models.segmentation import fcn_resnet50

def modelo_fnc(num_classes=1):
    model = fcn_resnet50(pretrained=True)

    # Reemplazar la última capa para segmentación binaria
    model.classifier[4] = nn.Conv2d(
        in_channels=512,
        out_channels=num_classes,
        kernel_size=1
    )

    return model





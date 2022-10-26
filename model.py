from email.policy import default
import torchvision

def get_model(device):
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large()
    model = model.eval().to(device)
    return model


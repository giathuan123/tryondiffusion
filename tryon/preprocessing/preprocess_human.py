import os

import cv2
import numpy as np
import torch
from PIL import Image
import numpy as np
from skimage import io
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from .u2net import RescaleT, ToTensorLab, SalObjDataset, normPRED, load_human_segm_model


# upscaling to original image
def pred_to_image(predictions: np.ndarray, image: Image.Image):
    im = Image.fromarray(predictions)
    imo = im.resize((image.size[0], image.size[1]),
                    resample=Image.Resampling.BILINEAR)
    return imo


def segment_human(inputs_dir: str, output_dir: str):
    """
    Segment human using U-2-Net
    :param image_path: image path
    :param output_dir: output directory
    """
    model_name = "u2net"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = [os.path.join(inputs_dir, file)
              for file in sorted(os.listdir(inputs_dir))]

    # 1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=images,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    net = load_human_segm_model(device, model_name)

    # 2. inference
    for i_test, data_test in enumerate(test_salobj_dataloader):
        print("inferencing:", images[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred: np.ndarray = d1[:, 0, :, :].detach().numpy()
        pred = normPRED(pred)
        pred = pred.squeeze()
        original = Image.open(images[i_test])

        mask = pred_to_image(pred, original)
        mask_cv2 = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)
        print(mask_cv2)
        subimage = cv2.subtract(mask_cv2, cv2.imread(
            images[i_test]))
        subimage = Image.fromarray(subimage)
        #
        # subimage = subimage.convert("RGBA")
        # original = original.convert("RGBA")
        #
        # subdata = subimage.getdata()
        # ogdata = original.getdata()
        #
        # newdata = []
        # for i in range(subdata.size[0] * subdata.size[1]):
        #     if subdata[i][0] == 0 and subdata[i][1] == 0 and subdata[i][2] == 0:
        #         newdata.append((231, 231, 231, 231))
        #     else:
        #         newdata.append(ogdata[i])
        # subimage.putdata(newdata)
        #
        # subimage.save(os.path.join(
        #     output_dir, f"{images[i_test].split(os.sep)[-1].split('.')[0]}.png"))
        #
        # del d1, d2, d3, d4, d5, d6, d7

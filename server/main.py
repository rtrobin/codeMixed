import io
import json
import base64

import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

from fastapi import FastAPI
import asyncio
from pydantic import BaseModel

app = FastAPI()

imagenet_class_index = json.load(open('./imagenet_class_index.json'))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = models.densenet121(pretrained=False).to(device)
model.eval()

class Input(BaseModel):
    salt_uuid: str = 'test1234'
    image_data: str = None

def load_image(image_str):
    try:
        image_stream = io.BytesIO(base64.b64decode(image_str))
        img = Image.open(image_stream)
        # print(img.size)
        return img
    except:
        print('image load failed.')
        return None

def transform_image(image):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    return my_transforms(image).unsqueeze(0)

def get_prediction(image):
    tensor = transform_image(image).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        _, y_hat = outputs.max(1)
        predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

@app.post("/predict")
async def predict(data: Input):
    ret = {
        'message': u'失败',
        'code': 5201,
        'salt_uuid': data.salt_uuid,
    }

    # python3.9
    img = await asyncio.to_thread(load_image, data.image_data)

    # python>=3.7
    # loop = asyncio.get_running_loop()
    # img = await loop.run_in_executor(None, load_image, data.image_data)

    if img == None:
        ret['message'] = u'图片解析失败'
        ret['code'] = 5202
        return ret

    # python3.9
    pred = await asyncio.to_thread(get_prediction, img)

    # python>=3.7
    # loop = asyncio.get_running_loop()
    # pred = await loop.run_in_executor(None, get_prediction, img)

    ret['message'] = u'请求成功'
    ret['code'] = 5200
    ret['result'] = pred

    return ret

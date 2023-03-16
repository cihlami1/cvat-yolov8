import base64
from ultralytics import YOLO
import json
import io
from PIL import Image
import yaml


def init_context(context):
    context.logger.info("Init context...  0%")

    # Read the DL model
    model = YOLO("yolov8m.pt")
    context.user_data.model = model

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run yolo-v8 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    context.user_data.model.conf = threshold
    image = Image.open(buf)
    yolo_results = context.user_data.model.predict(source=image, stream=True, imgsz=1980)

    with open("/opt/nuclio/function.yaml", 'rb') as function_file:
        functionconfig = yaml.safe_load(function_file)

    labels_spec = functionconfig['metadata']['annotations']['spec']
    labels = {item['id']: item['name'] for item in json.loads(labels_spec)}

    # print(yolo_results)
    encoded_results = []
    for result in yolo_results:
        # print(result.boxes.cls)
        for idx, cls in enumerate(result.boxes.cls):
            encoded_results.append({
                'confidence': result.boxes.conf[idx].item(),
                'label': labels[cls.item()],
                'points': result.boxes.xyxy[idx].tolist(),
                'type': 'rectangle'
            })

    return context.Response(body=json.dumps(encoded_results), headers={},
                            content_type='application/json', status_code=200)

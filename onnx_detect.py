import time
import onnxruntime as ort
import onnx
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import cv2

def onnx_detect(image_path : str, onnx_model_path : str):

    resized_image_path = Path(image_path).parent / "resized.jpg"
    input_image = cv2.imread(str(resized_image_path))
    input_image = input_image.transpose(2, 0, 1)

    # Load the ONNX model
    ort_session = ort.InferenceSession(onnx_model_path)
    # Prepare the input data
    input_data = np.expand_dims(input_image, axis=0).astype(np.float32)

    print(input_data.shape)
    # input_data = np.reshape(input_image, (1, 3, 224, 224)).astype(np.float32)
 
    # Run the model
    if 'finetune' in onnx_model_path:
        output = ort_session.run(None, {'onnx::QuantizeLinear_0': input_data})
    else:
        output = ort_session.run(None, {'input_0': input_data})


    # Return the result

    # import openvino as ov
    
    # core = ov.Core()
    # model_onnx = core.read_model(model=onnx_model_path)
    # compiled_model_onnx = core.compile_model(model=model_onnx, device_name="CPU")

    # # # save compiled model
    # # ov.save_model(compiled_model_onnx, "vino-model.xml")

    # # # load compiled model
    # # compiled_model_onnx = core.read_model(model="vino-model.xml")


    
    # input_image = input_image.reshape(1, 3, 224, 224).astype(np.float32)
    
    # output_layer = compiled_model_onnx.output(0)
    # result_infer = compiled_model_onnx([input_image])[output_layer]
    first_result = output[0][0]


    # for i in range(0, len(first_result)):
    #     x = first_result[i][0]
    #     y = first_result[i][1]
    #     w = first_result[i][2]
    #     h = first_result[i][3]
    #     p = first_result[i][4]
    #     class_id = np.argmax(first_result[i][5:])

        # if p > 0.7:
        #     print(f"Found object with class {class_id} with probability {p} at {x}, {y}, {w}, {h}")


parser = argparse.ArgumentParser(description='Run yolov5 in a command line')
parser.add_argument('--model_path', type=str, help='Path to the onnx model file')
parser.add_argument('--image_path', type=str, help='Path to the image file')
args = parser.parse_args()
image_path = args.image_path
onnx_model_path = args.model_path

start_time = time.time()
output = onnx_detect(image_path, onnx_model_path)
output = onnx_detect(image_path, onnx_model_path)
output = onnx_detect(image_path, onnx_model_path)
output = onnx_detect(image_path, onnx_model_path)
output = onnx_detect(image_path, onnx_model_path)

end_time = time.time()
print(f"Time taken: {end_time - start_time}")
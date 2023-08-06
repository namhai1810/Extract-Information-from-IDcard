from PIL import Image
import gradio as gr
import requests
import io
import numpy as np
import cv2
import json
from align_image.ocr.ocr import ocr
api_url_align = "http://0.0.0.0:5203/upload"
api_url_classify = "http://0.0.0.0:5201/classification"

from io import BytesIO

def predict_align(image):
    # Convert to bytesio
    img_receive = io.BytesIO()
    image.save(img_receive, format="PNG")
    # receive response
    img_receive.seek(0)
    response = requests.post(api_url_align, data = {"id":20},files={"image": img_receive})
    img_bytes_io = io.BytesIO(response.content)
    result = Image.open(img_bytes_io)
    
    return result

def predict_classify(image):
    # Convert to bytesio
    img_receive = io.BytesIO()
    image.save(img_receive, format="PNG")
    # receive response
    img_receive.seek(0)
    response = requests.post(api_url_classify, data = {"id":20},files={"image": img_receive})
    # print(response.status_code)
    response = response.json()
    return response["result"]

def predict(image):
    
    img = predict_align(image)
    cv2.imwrite("output.png",  np.array(img))
    classify = predict_classify(img)
    l = ["others","citizen ID_ back", "citizen ID with chip_ back",'ID_ back']
    print(classify)
    if classify not in l:
        img_cmt, id_num,img_ten, name, img_birth, dob,  img_hometown, hometown= ocr(img, classify)
    else:
        img_error = Image.open('load_error.png')
        img_cmt = img_ten = img_birth = img_hometown = img_error
        id_num = name = dob = hometown = "NONE"
    return img, classify, img_cmt, id_num,img_ten, name, img_birth, dob,img_hometown, hometown

if __name__ == '__main__':

    with gr.Blocks() as demo:
        with gr.Row(scale=1,  max_height=200):
            input_align = gr.inputs.Image(type="pil")
            output_align = gr.Image(type="pil")
        with gr.Row():
            image_align_button = gr.Button("Run")
        with gr.Row():
            classify = gr.Textbox(label="class of image")
        with gr.Row():
            img_cmt = gr.Image(type="pil")
            id_num = gr.Textbox(label="ID number")
        with gr.Row():
            img_ten = gr.Image(type="pil")
            name = gr.Textbox(label="Name")
        with gr.Row():
            img_birth = gr.Image(type="pil")
            dob = gr.Textbox(label="Date of birth")
        with gr.Row():
            img_hometown = gr.Image()
            hometown = gr.Textbox(label="Hometown")

        image_align_button.click(predict, inputs = input_align, outputs = [output_align, classify, img_cmt,
                                                                            id_num,img_ten, name, img_birth, dob,
                                                                            img_hometown, hometown])


        
    demo.launch(server_name="0.0.0.0",server_port=7878)

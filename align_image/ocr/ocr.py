import numpy as np
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import gradio as gr
import cv2
def ocr(img, name):
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained']=False
    config['device'] = 'cpu'
    detector = Predictor(config)
    width,height = img.size
    print(width, height)
    fixed_width = 1280
    fixed_height = 720

    if name == 'citizen ID with chip_ front':
        img_cmt = img.crop((500 /fixed_width *width , 290/fixed_height * height,
                            990 /fixed_width *width, 350/fixed_height * height))

        img_ten = img.crop((350/fixed_width *width, 390/fixed_height * height,
                            1100/fixed_width *width, 450/fixed_height * height))

        img_birth = img.crop((730/fixed_width *width, 455/fixed_height * height,
                            1200/fixed_width *width, 495/fixed_height * height))

        img_que_tren = img.crop((750/fixed_width *width, 530/fixed_height * height,
                                1280/fixed_width *width, 600/fixed_height * height))

        img_que_duoi = img.crop((375/fixed_width *width, 585/fixed_height * height,
                                1000/fixed_width *width, 640/fixed_height * height))

        # img_expiry = img.crop((16/fixed_width *width, 645/fixed_height * height,
        #                         360/fixed_width *width, 710/fixed_height * height))


    if name == 'ID_ front':
        img_cmt = img.crop((647.1 /fixed_width *width, 192/fixed_height * height,
                            1051 /fixed_width *width, 240/fixed_height * height))

        img_ten = img.crop((530/fixed_width *width, 267/fixed_height * height,
                            1241/fixed_width *width, 320/fixed_height * height))

        img_birth = img.crop((750/fixed_width *width, 389/fixed_height * height,
                            1052/fixed_width *width, 450/fixed_height * height))

        img_que_tren = img.crop((736/fixed_width *width, 449/fixed_height * height,
                                1280/fixed_width *width, 520/fixed_height * height))
        img_que_duoi = img.crop((570/fixed_width *width, 511/fixed_height * height,
                                1114/fixed_width *width, 580/fixed_height * height))

        # img_expiry = ""


    if name == 'citizen ID_ front':
        img_cmt = img.crop((580 /fixed_width *width , 185/fixed_height * height, 
                            1125 /fixed_width *width, 250/fixed_height * height))

        img_ten = img.crop((620/fixed_width *width, 277/fixed_height * height,
                            1280/fixed_width *width, 340/fixed_height * height))

        img_birth = img.crop((770/fixed_width *width, 372/fixed_height * height,
                            1015/fixed_width *width, 415/fixed_height * height))

        img_que_tren = img.crop((600/fixed_width *width, 495/fixed_height * height,
                                1280/fixed_width *width, 580/fixed_height * height))

        img_que_duoi = ""

        # img_expiry = img.crop((263/fixed_width *width, 679/fixed_height * height,
        #                         455/fixed_width *width, 712/fixed_height * height))

    images = [img_cmt, img_ten, img_birth, img_que_tren, img_que_duoi]
    results = []
    for i in range(0,len(images)):
        # images[i].show()
        if images[i] != "":
            results.append(detector.predict(images[i]))
        else:
            results.append("")

    id_num = 'ID number: '+results[0]+ "\n "
    name = 'Name: ' + results[1] + '\n'
    dob = 'Dob: ' + results[2] + '\n'

    if ' ' not in results[3]:
        results[3] =""
    else:
        results[3] += ', '
    if ' ' not in results[4]:
        results[4] =""
    #concat image
    if img_que_duoi != "":
        img_que_tren = img_que_tren.resize((img_que_duoi.size[0], img_que_duoi.size[1]))
        print(img_que_tren.size, img_que_duoi.size)
        img_hometown = cv2.vconcat([np.array(img_que_tren), np.array(img_que_duoi)])
    else:
        img_hometown = img_que_tren
    hometown = 'Local: ' + results[3]  + results[4] +'\n'
    return img_cmt, id_num, img_ten, name, img_birth, dob, img_hometown, hometown


if __name__ == '__main__':
    # img = Image.open('/home/namhai18/study/Load_align/cccdc.png')
    # print(ocr(img))
    input = gr.inputs.Image(type="pil")
    demo = gr.interface.Interface(fn=ocr, inputs=input, outputs="text")
    demo.launch()
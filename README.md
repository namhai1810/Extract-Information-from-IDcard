# Extract-Information-from-IDcard
## Author: namhai18
## Description
- ID card play an important role to supply imformation to identify human and this applications can detect in IDcard with chip, ID card without chip and namecard in Vietnam.
- This application uses a deep learning model that utilizes OCR techniques to extract important information from name cards. It can process images of name cards and automatically extract relevant fields such as ID numbers, names, date of birth, hometown.
## Installation
### Run by command line in ubuntu

> git clone https://github.com/namhai1810/Extract-Information-from-IDcard.git

> pip install -r requirements.txt

#### **Run website**
> python gra.py

#### **Run alignment**
> cd align_image && python server/app.py

#### **Run classifications**
> cd classify_image && python server/app.py

### Run by docker
>  docker build -t vn-id-ekyc .

> docker compose up

-   However, gradio app now have some problems that doesn't allow us to run parrallel (maybe this is a bug in gradio). Therefore, you can run it by command 

## Demo Images:
![image](https://github.com/namhai1810/Extract-Information-from-IDcard/assets/117570211/0ebb0e5f-51e6-4944-ab32-f19638a00ff5)
![image](https://github.com/namhai1810/Extract-Information-from-IDcard/assets/117570211/8cb587db-4b8d-4581-92a2-e1a5efaae7aa)










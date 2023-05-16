# static_img.py
from typing import List
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
import os, shutil

app = FastAPI()

# Mount the static files directory
image_root_directory = "C:\\Users\\georg\\Pictures\\TrainDetector\\"
image_folders = ['inbox', 'results', 'training_data']

for folder in image_folders:
    app.mount(f"/{folder}", StaticFiles(directory = os.path.join(image_root_directory, folder)), name = folder)

templates = Jinja2Templates(directory="templates")
#image_folder = "inbox/148/2023-05-07"
current_folder = "."

def get_images(folder = ''):
    file_list = os.listdir(folder)
    return_list = []

    for f in file_list:
        f_elements = f.split('.')
        if len(f_elements) > 1:
            if f_elements[1:][0] in ['jpg', 'jpeg']:
                return_list.append(os.path.join(folder, f).replace("\\","/")[1:])  # .replace("\\","/"), use [1:] to remove leading "."

    return return_list

"""
In this updated code, the route definition @app.get("/{directory_path:path}") and @app.post("/{directory_path:path}/select-images")
include the {directory_path:path} part as a route parameter. The :path component in the route parameter captures the entire URL path,
including slashes. This allows you to process the URL dynamically.
"""

def generate_browser(request, folder):

    # get subfolders from OS
    root_folder = "."
    os_target_path = os.path.join(root_folder, *folder.split('/'))
    folder_list = [item for item in os.listdir(os_target_path) if os.path.isdir(os.path.join(os_target_path, item))]
    print(f"folder_list: {folder_list}")

    # current folder
    print(f"current_folder: {folder}")

    # determine parent folder for navigation
    parent_folder = "/" + "/".join(folder.split('/')[:-1])
    print(f"parent_folder: {parent_folder}")

    # get any images in current folder
    image_filenames = get_images(os_target_path)

    return templates.TemplateResponse("browse.html", {"request": request, "folders": folder_list, "current_folder": folder, "parent_folder": parent_folder, "images": image_filenames})


@app.get("/browse/{folder:path}")
@app.post("/browse/{folder:path}")
async def browse(request: Request, folder):

    return generate_browser(request, folder)

"""
Ensure that the name attribute of the checkboxes in the HTML template (name="selected_images") matches the parameter name (selected_images) in the select_images() function.
"""
@app.post("/select-image")
async def select_image(request: Request, current_folder: str = Form(...), selected_images: List[str] = Form(...), action: str = Form(...)):
    # Handle the selected image
    # You can perform further processing or logic based on the selected image
    print(f"current folder: {current_folder}")
    print(f"Selected images: {selected_images}")
    print(f"Action: {action}")

    file_list = []

    for file in selected_images:
        file = os.path.join(image_root_directory, *file.split("/"))
        #print(f"checking if file exists: {file}")
        if os.path.isfile(file):
            #print("YES")
            file_list.append(file)


    try:
        # handle file according to the selected action:
        if action == "Delete":
            for file in file_list:
                os.remove(file)

        elif action == "Train":
            for file in file_list:
                file_parts = file.split('\\')[-1:][0].split('_')
                #print(file_parts)
                cam_num = file_parts[0]

                shutil.move(file, os.path.join(image_root_directory, "training_data", cam_num, "train"))

        elif action == "No Train":
            for file in file_list:
                file_parts = file.split('\\')[-1:][0].split('_')
                #print(file_parts)
                cam_num = file_parts[0]

                shutil.move(file, os.path.join(image_root_directory, "training_data", cam_num, "no_train"))

    except Exception as ex:
        print(str(ex))

    return RedirectResponse(url = f"/browse/{current_folder}")
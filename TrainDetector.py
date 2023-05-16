#-------------------------------------------------------------------------------
# Name:        TrainDetector
# Purpose:     Collect traffic camera images from 511 Québec and classiy them for passing trains
#
# Author:      Georg Denoix
#
# Created:     28-04-2023
# Copyright:   (c) Georg Denoix 2023
# Licence:     open source
#-------------------------------------------------------------------------------

import requests, os, time, cv2, shutil, math, hashlib, imutils, logging
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import json, facebook
from facebook import GraphAPI
#import sockets


class image_capture:

    def __init__(self, root_folder = '', input_root_folder = 'inbox', output_root_folder = 'results', cam_num = 0, configuration_file = 'train_detection_configuration.csv', fb_credential_file = ''):

        # event logging
        self.log = logging.getLogger(__name__)

        # base parameters
        self.configuration_file = configuration_file
        self.fb_credential_file = fb_credential_file
        self.fb_access_token = ''
        self.root_folder = root_folder
        self.base_file_name = ''
        self.image_file_names = []
        self.video_file_name = ''
        self.cam_num = cam_num
        self.date = ''
        self.week_day = ''
        self.time = ''

        # camera data
        self.new_data = False
        self.last_data_checksum = 0x00

        # classification results
        self.result = {
            'shows_train': False,
            'motion_value': 0,
            'motion_detected': False,
            'classified': False,
            'nr_confidence': 0,
            'archived': False,
            'detection_count': 0,
            'detection_threshold': 1,
            'cnt_pic': 0,
            'cnt_train': 0,
            'cnt_no_train': 0,
            'caption': '',
            'fb_url': ''
        }

        # folder configuration
        #self.output_folder_train_template = 'media\\train\\{cam_num}\\{date_str}'
        #self.output_folder_notrain_template = 'media\\no_train\\{cam_num}\\{date_str}'
        self.input_root_folder = input_root_folder
        self.input_folder = ''
        self.output_root_folder = output_root_folder
        self.output_folder_train = ''
        self.output_folder_notrain = ''

        # configuration (to be read from configuration file)
        self.cam_config = {}

        self.nr_disable = False  # keep in case model cannot be loaded
        self.loaded_nr_version = -1   # keep to identify if new version specified

        # read configuration
        self.read_configuration()

        # load classification model  ==>   put into function, so that model could be loaded dynamically at runtime whenever a new configuration is found
        self.load_model()

        # read facebook credentials
        self.page_graph = False
        try:
            page_credentials = self.read_fb_creds(os.path.join(root_folder, self.fb_credential_file))
            self.fb_access_token = page_credentials['access_token']
            self.page_graph = GraphAPI(access_token = self.fb_access_token)
        except Exception as ex: # work on python 3.x
            self.log.error('image_capture (init): Could not load fb credentials: ' + str(ex))

        self.fb = fb_helper(self.page_graph)

        # create root folder
        self.set_root_folder(root_folder)

        # if cam num is already specified, create folder structure
        if os.path.isdir(self.root_folder):
            self.set_input_folder()
            if cam_num > 0:
                self.set_output_folder()

    def load_model(self):
        nr_version = self.conf_value('nr_version')
        model_file_name = ''

        # Load model when configured version larger than previously loaded version
        if self.conf_value('nr_enable') and nr_version != self.loaded_nr_version:
            self.log.info(f'load_model: Loading new model for camera {self.cam_num}; new version: {nr_version}, previous: {self.loaded_nr_version}')
            try:
                model_version = int(nr_version)
                model_file_name = os.path.join(self.root_folder, 'models\\neural networks', f'cam{self.cam_num}_model_v{model_version}.h5')
            except Exception as ex:
                self.loaded_nr_version = -1
                self.nr_model = False
                self.nr_disable = True
                self.log.error(f'load_model: Could not determine model version:' + str(ex))

            try:
                self.nr_model = load_model(model_file_name)
                self.loaded_nr_version = nr_version
                self.log.info(f'load_model: successfully loaded {model_file_name} for camera {self.cam_num}.')
            except Exception as ex:
                # disable classification if loading model failed
                self.nr_model = False
                self.nr_disable = True
                self.log.error(f'load_model: Could not load model {model_file_name}:' + str(ex))
        else:
            self.log.debug(f'load_model: Loading model skipped for cam {self.cam_num}.')
            self.nr_disable = True
            self.nr_model = False

    # set the root folder for all further operations
    def set_root_folder(self, folder_name = ''):

        if not os.path.isdir(folder_name):
            try:
                os.makedirs(folder_name)
            except Exception as ex:
                self.log.error(f'set_root_folder: could not create {folder_name}: ' + str(ex))

        return True

    def set_input_folder(self):

        if self.root_folder == '':
            self.log.error('set_input_folder: need to define root folder first')
            return False

        if self.cam_num == 0:
            self.log.error('set_output_folder: cam num needs to be defined first')
            return False

        date_str = datetime.now().strftime("%Y-%m-%d")
        self.input_folder = os.path.join(self.root_folder, self.input_root_folder, str(self.cam_num), date_str)

        if not os.path.isdir(self.input_folder):
            os.makedirs(self.input_folder)

        return True

    # todo: use templates for folder names
    def set_output_folder(self):

        if self.root_folder == '':
            self.log.error('set_output_folder: need to define root folder first')
            return False

        if self.cam_num == 0:
            self.log.error('set_output_folder: cam num needs to be defined first')
            return False

        output_folder_root = os.path.join(self.root_folder, self.output_root_folder)

        if not os.path.isdir(output_folder_root):
            try:
                os.makedirs(output_folder_root)
            except Exception as ex:
                self.log.error(f'set_output_folder: could not create {output_folder_root}: ' + str(ex))

        # create further structure underneath
        media_folder = os.path.join(output_folder_root, 'media')
        if not os.path.isdir(media_folder):
            try:
                os.makedirs(media_folder)
            except Exception as ex:
                self.log.error(f'set_output_folder: could not create {media_folder}: ' + str(ex))

        # train branch
        date_str = datetime.now().strftime("%Y-%m-%d")

        self.output_folder_train = os.path.join(output_folder_root, 'media', 'train', str(self.cam_num), date_str)
        output_folder_train = os.path.join(self.root_folder, self.output_folder_train)
        if not os.path.isdir(output_folder_train):
            try:
                os.makedirs(output_folder_train)
            except Exception as ex:
                self.log.error(f'set_output_folder: could not create {output_folder_train}' + str(ex))

        # no_train branch
        self.output_folder_notrain = os.path.join(output_folder_root, 'media', 'no_train', str(self.cam_num), date_str)
        output_folder_notrain = os.path.join(self.root_folder, self.output_folder_notrain)
        if not os.path.isdir(output_folder_notrain):
            try:
                os.makedirs(output_folder_notrain)
            except Exception as ex:
                self.log.error(f'set_output_folder: could not create {output_folder_notrain}' + str(ex))

        return True

    # function to access individual configuration value
    def conf_value(self, key):

        # initialise with FALSE
        ret_val = False
        use_config = {}

        # check if list, if so, take first element
        if isinstance(self.cam_config, list):
            if len(self.cam_config) > 0:
                use_config = self.cam_config[0]
        else:
            use_config = self.cam_config

        # needs to be dictionary to work
        if isinstance(use_config, dict):

            if key in use_config.keys():
                read_val = use_config[key]
            else:
                self.log.error(f'conf_value - key {key} does not exist.')
                read_val = False

            # clean strings
            if isinstance(read_val, str):
                ret_val = read_val.replace(u'\xa0', u' ')

            if isinstance(read_val, int) or isinstance(read_val, float):
                ret_val = read_val

            # 'x' means TRUE boolean configuration
            if read_val == 'x':
                ret_val = True

            # nan means FALSE
            if isinstance(read_val, float):
                if math.isnan(read_val):
                    ret_val = False
        else:
            self.log.error('conf_value: configuration passed is not a dictionary.')

        return ret_val

    # Read camera configuration from csv file
    def read_configuration(self):

        incomplete_reading = False

        try:
            configuration_pd = pd.read_csv(os.path.join(self.root_folder, self.configuration_file), encoding = "ISO-8859-1")
            self.log.debug(f'Configuration for camera {self.cam_num} successfully read from configuration file: {self.configuration_file}')
        except Exception as ex:
            self.log.error(f'read_configuration: could not read {self.configuration_file}: {str(ex)}')
            return False

        # check if configuration contains an entry for this camera
        if not (configuration_pd['cam_id'] == self.cam_num).any():
            self.log.error(f'read_configuration: {self.configuration_file} has no entry for camera {self.cam_num}.')
            return False

        self.cam_config = configuration_pd.loc[configuration_pd['cam_id'] == self.cam_num].to_dict(orient = 'records')

        return not incomplete_reading

    # Read facebook credentials (access token)
    def read_fb_creds(self, filename):
        with open(filename) as f:
            credentials = json.load(f)
        return credentials

    def acquire_camera_data(self, cam_num = 0):

        # initialize results
        self.result["shows_train"] = False
        self.result["motion_value"] = 0
        self.result["motion_detected"] = False
        self.result["classified"] = False
        self.result["archived"] = False

        # pre-checks
        if (cam_num == 0) and (self.cam_num == 0):
            self.log.error('acquire_camera_data: no camera number specified.')
            return False

        # always re-set input folder to keep correct folder correct for current date
        self.set_input_folder()

        if (self.input_folder == '') or (not os.path.isdir(os.path.join(self.root_folder, self.input_folder))):
            self.log.error(f'acquire_camera_data: input folder not set ({self.input_folder})')
            return False

        # override cam_num if necessary
        if cam_num > 0:
            self.cam_num = cam_num

        # define time and file name
        time_str =  datetime.now().strftime("%Y-%m-%d_%Hh%M_%S")
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.time = datetime.now().strftime("%H:%M:%S")
        self.week_day = datetime.now().strftime("%w")
        self.base_file_name = f'{cam_num}_{time_str}'

        # construct new file name
        last_file_name = self.video_file_name
        if self.conf_value('cam_type') == '511':
            cam_URL = f'https://www.quebec511.info/Carte/Fenetres/camera.ashx?id={self.cam_num}&format=mp4'
            self.video_file_name = f'{self.base_file_name}.mp4'
        elif self.conf_value('cam_type') == 'vdm':
            cam_URL = f'https://ville.montreal.qc.ca/Circulation-Cameras/GEN{self.cam_num}.jpeg'
            self.video_file_name = f'{self.base_file_name}.jpeg'
        elif self.conf_value('cam_type') == 'vdo':
            cam_URL = f'https://traffic.ottawa.ca/beta/camera?id={self.cam_num}'
            self.video_file_name = f'{self.base_file_name}.jpg'
        else:
            cam_type = self.conf_value('cam_type')
            self.log.error(f'acquire_camera_data: no URL for camera of type: ({cam_type})')

        video_file_path = os.path.join(self.root_folder, self.input_folder, self.video_file_name)
        try:
            # open video file for writing
            video_file = open(video_file_path, 'wb')

            # access video stream
            response = requests.get(cam_URL, stream=True)
        except:
            self.log.error(f'acquire_camera_data: something went wrong downloading data from {cam_URL}')
            return False

        # save video file
        if response.ok:
            try:
                for block in response.iter_content(1024):
                    if not block:
                        break
                    video_file.write(block)
            except:
                self.log.error(f'acquire_camera_data: could not save: {video_file.name}')
                return False
            #print(f'acquire_camera_data: written: {video_file.name}')
            video_file.close()
        else:
            self.log.error('acquire_camera_data: response not ok')

        # check if downloaded data is new, compared to last data
        new_checksum = hashlib.md5(open(video_file_path, 'rb').read()).hexdigest()
        self.new_data = new_checksum != self.last_data_checksum

        if self.new_data:
            self.last_data_checksum = new_checksum
            self.result["cnt_pic"] += 1
            return True
        else:
            #restore last file name
            self.video_file_name = last_file_name
            os.remove(video_file_path)
            return False

    # todo: make number of extracted images variable
    def extract_frames(self):

        full_video_file_name = os.path.join(self.root_folder, self.input_folder, self.video_file_name)

        if not os.path.isfile(full_video_file_name):
            self.log.error('extract_frames: video file {full_video_file_name} does not exist.')
            return False

        self.image_file_names = []

        # if camera delivers mp4 video, extract frames:
        if self.conf_value('cam_format') == 'mp4':
            cam = cv2.VideoCapture(full_video_file_name)

            currentframe = 0
            while(True):
                ret, frame = cam.read()
                if ret:
                    frame_base_name = f'{self.base_file_name}_{currentframe}.jpg'
                    frame_file = os.path.join(self.root_folder, self.input_folder, frame_base_name)
                    if (currentframe % 90) == 0:
                        cv2.imwrite(frame_file, frame)
                        self.image_file_names.append(frame_base_name)
                        self.log.debug('extract_frames: Written ' + frame_file)
                    currentframe += 1
                else:
                    break

            # release all memory from cv
            cam.release()
            cv2.destroyAllWindows()

        # otherwise (assume jpg images), use image file as-is:
        else:
            self.image_file_names.append(self.video_file_name)

    # todo: add classification fo type of train
    def classify_nr(self):

        if self.nr_disable:
            self.log.warning(f'classify_nr: skipping classification for {self.cam_num} because nr_disable = True')
            return False

        self.result["shows_train"] = False

        try:
            if len(self.image_file_names) > 0:
                for imgFile in self.image_file_names:
                    if not self.result["shows_train"]:
                        try:
                            # open image and pre-process for use with model
                            img_path = os.path.join(self.root_folder, self.input_folder, imgFile)
                            img = tf.keras.utils.load_img(img_path, target_size=(256, 256)) #why was one model trained with size = 255?
                            img = tf.keras.utils.img_to_array(img)
                            img = np.expand_dims(img, axis=0)
                        except Exception as ex:
                            self.log.error(f'classify_nr: could not open {imgFile}: ' + str(ex))
                            pass

                        try:
                            # use model to predict if this image contains a train
                            self.result["nr_confidence"] = self.nr_model.predict(img)[0][0]

                            # If confidence is more than 0.5, probably found a train:
                            if (self.result["nr_confidence"] > 0.5):
                                # set result to "train":
                                self.result["shows_train"] = True
                                self.result["cnt_train"] += 1
                                self.result["detection_count"] += 1
                            else:
                                # set result to "no train":
                                self.result["shows_train"] = False
                                self.result["cnt_no_train"] += 1
                                self.result["detection_count"] = 0

                            self.result["classified"] = True
                            self.log.info(f'classify_nr: {imgFile} is showing a train: {self.result["nr_confidence"]} => {self.result["shows_train"]}, detection count: {self.result["detection_count"]}, threshold: {self.result["detection_threshold"]}; cnt: {self.result["cnt_pic"]}, train: {self.result["cnt_train"]}, no_train: {self.result["cnt_no_train"]}')

                        except Exception as ex:
                            self.log.error(f'classify_nr: something went wrong processing {img_path}: ' + str(ex))
                            pass
                    else:
                        self.result["classified"] = True
                        self.log.warning(f'classify: skipping {imgFile}, train already detected.')

        except Exception as ex:
            self.log.error('classify_nr: ran into a problem: ', str(ex))

        return self.result["shows_train"]

    def detect_motion(self):

        # image width
        image_width = 350

        # initialize the first frame in the video stream
        firstFrame = None

        # Initialize frame counter
        frameNo = 0

        # minimum size of motion detection
        min_area = 170

        # blur radius
        blur_radius = 21

        # maximum distance for object tracking
        max_dist = 15

        # minimum motion needed
        min_motion = 4

        # store centers of detected motion from previous frame
        pfCenters = []

        # list of tracked objects / areas
        trackedObjects = {}
        trackedObjectHistory = []
        trackId = 0
        self.result["motion_detected"] = False

        # Read video from file
        video_file_path = os.path.join(self.root_folder, self.input_folder, self.video_file_name)
        video = cv2.VideoCapture(video_file_path)

        # Read the mask
        use_mask = True
        if self.conf_value('mask'):
            full_mask_file_name = os.path.join(self.root_folder, 'models\\masks', f'{self.cam_num}_mask.png')
            try:
                mask = imutils.resize(cv2.imread(full_mask_file_name, 0), width = image_width)
            except:
                self.log.error(f'detect_motion: failed to read mask {full_mask_file_name}.')
                use_mask = False
        else:
            use_mask = False


        while True:
            # grab current frame
            check, curFrame = video.read()

            # when no frame can be grabbed, we reached the end of the file
            if curFrame is None:
                break

            # Current frame's center points
            cfCenters = []

            # resize for uniformity
            curFrame = imutils.resize(curFrame, width = image_width) # not really necessary, but keep line when processing higher resolution

            # apply mask
            maskedImage = curFrame
            if use_mask:
                try:
                    maskedImage = cv2.bitwise_and(curFrame, curFrame, mask = mask)
                except:
                    pass

            # convert it to grayscale, and blur it to remove noise (radius 21 px)
            grayImage = cv2.cvtColor(maskedImage, cv2.COLOR_BGR2GRAY)
            grayImage = cv2.GaussianBlur(grayImage, (blur_radius, blur_radius), 0)

            # if the first frame is None, initialize it
            if firstFrame is None:
                firstFrame = grayImage

            # compute the absolute difference between the current frame and
            # first frame
            frameDelta = cv2.absdiff(firstFrame, grayImage)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

            # dilate the thresholded image to fill in holes, then find contours
            # on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # loop over the contours
            for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < min_area:
                    continue
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(curFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # calculate box centers and draw circle
                cx = int(x + w / 2)
                cy = int(y + h / 2)
                cfCenters.append((cx, cy))


            # track EXISTING objects in current frame
            for objectId, pt2 in trackedObjects.copy().items():
                objectExists = False

                # go through object positions in current frame
                for pt in cfCenters.copy():
                    # calculate distance between the two points (of current and previous frame)
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                    # if distance is below threshold, consider it's the same object, and *update* position
                    if distance < max_dist:
                        objectExists = True
                        trackedObjects[objectId] = pt
                        # remove from list when tracked, so we can later compare with last frame to find new objects
                        cfCenters.remove(pt)
                        continue

                # remove object from tracking if not existing any more
                if not objectExists:
                    trackedObjects.pop(objectId)
                    #pass

            # find NEW objects by measure distance of remaining objects between current frame to previous frame
            for pt in cfCenters:
                for pt2 in pfCenters:
                    # calculate distance between the two points (of current and previous frame)
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                    # if distance is below threshold, consider it's the same object
                    if distance < max_dist:
                        trackedObjects[trackId] = pt
                        trackId += 1

            # store tracked object history
            trackedObjectHistory.append(trackedObjects.copy())

            # count frames
            frameNo += 1

            # copy center points of current frame
            pfCenters = cfCenters.copy()

        # free any used memory
        video.release()
        cv2.destroyAllWindows()

        # collect each object's start and end point
        trackedObjects = {}

        for objectList in trackedObjectHistory:
            # check each item in object history
            for objectId, pt in objectList.items():

                # check if object is already tracked, then only update second coordinates:
                if objectId in trackedObjects.keys():
                    trackedObjects[objectId] = (trackedObjects[objectId][0], pt)

                # if new, add to list
                else:
                    trackedObjects[objectId] = (pt, pt)

        # calculate each objects absolute movement
        for key, currentObject in trackedObjects.items():
            dx = currentObject[1][0] - currentObject[0][0]
            dy = currentObject[1][1] - currentObject[0][1]
            trackedObjects[key] = (currentObject[0], currentObject[1], (dx, dy))

        # calculate total x and y movement of all objects combined
        X = 0
        Y = 0

        for currentObject in trackedObjects.values():
            X += currentObject[2][0]
            Y += currentObject[2][1]

        # store detected motion value
        self.result["motion_value"] = X
        if X > self.conf_value('motion_threshold'):
            self.result["motion_detected"] = True
            self.log.info(f'Objects on camera {self.cam_num} were mostly moving RIGHT')
        elif X < -self.conf_value('motion_threshold'):
            self.result["motion_detected"] = True
            self.log.info(f'Objects on camera {self.cam_num} were mostly moving LEFT')
        else:
            self.log.info(f'NO MOTION was detected on camera {self.cam_num}.')


        # Code below should be in separate function, re-use from
        if self.result["motion_detected"] and self.conf_value('det_motion_only'):
            self.result["shows_train"] = True
            self.result["cnt_train"] += 1
            self.result["detection_count"] += 1
            self.log.info(f'{self.cam_num} shows movement and is configured to recognize train without NR.')
            self.log.info(f'classify_nr: {self.cam_num} is showing a train: {self.result["shows_train"]}, detection count: {self.result["detection_count"]}, threshold: {self.result["detection_threshold"]}; cnt: {self.result["cnt_pic"]}, train: {self.result["cnt_train"]}, no_train: {self.result["cnt_no_train"]}')

        elif self.conf_value('det_motion_only'):
            self.result["shows_train"] = False
            self.result["cnt_no_train"] += 1
            self.result["detection_count"] = 0

        return True

    # function to generate image caption
    def generate_caption(self):

        # Construct message for posting
        try:
            cam_text = self.conf_value('cam_description')
        except:
            cam_text = str(self.cam_num)

        motion_text= ''
        if self.conf_value('motion_enable'):
            motion_text = '(no motion detected)'

        if self.result["motion_detected"]:
            if self.result["motion_value"] > 0:
                motion_text = 'heading ' + self.conf_value('right_orientation')
            else:
                motion_text = 'heading ' + self.conf_value('left_orientation')

        caption = 'Train ' + motion_text + ' detected at ' + self.time + ' on camera ' + cam_text + ' [' + str(self.cam_num) + '] on ' + self.date + '.'

        self.result["caption"] = caption

        return caption


    # function to archive current record
    def archive_files(self, delete_no_train = True, delete_jpg = False, ignore_threshold = False):

        if self.output_folder_notrain == '':
            self.log.error(f'archive_files: output folder not defined.')
            return False

        if self.output_folder_train == '':
            self.log.error(f'archive_files: output folder not defined.')
            return False

        self.set_output_folder()

        # Image shows a train, so move it to the right folder
        if self.result["shows_train"] and ((self.result["detection_count"] <= self.result["detection_threshold"]) or ignore_threshold):

            if delete_jpg:
                try:
                    for img_file in self.image_file_names:
                        os.remove(os.path.join(self.root_folder, self.input_folder, img_file))
                except Exception as ex:
                    self.log.error(f'archive_files: could not delete {img_file}: ' + str(ex))
            else:
                try:
                    # move image files
                    for img_file in self.image_file_names:
                        shutil.move(os.path.join(self.root_folder, self.input_folder, img_file), self.output_folder_train)
                        pass
                except Exception as ex:
                    self.log.error(f'archive_files: could not move {img_file} to {self.output_folder_train}: ' + str(ex))
                    pass

            try:
                shutil.move(os.path.join(self.root_folder, self.input_folder, self.video_file_name), self.output_folder_train)
            except Exception as ex:
                self.log.error(f'archive_files: could not move {self.video_file_name} to {self.output_folder_train}: ' + str(ex))

        elif delete_no_train:
            try:
                for img_file in self.image_file_names:
                    os.remove(os.path.join(self.root_folder, self.input_folder, img_file))
            except Exception as ex:
                self.log.error(f'archive_files: could not delete {img_file}: ' + str(ex))

            try:
                os.remove(os.path.join(self.root_folder, self.input_folder, self.video_file_name))
            except Exception as ex:
                self.log.error(f'archive_files: could not delete {self.video_file_name}: ' + str(ex))

        else:
            if delete_jpg:
                try:
                    for img_file in self.image_file_names:
                        os.remove(os.path.join(self.root_folder, self.input_folder, img_file))
                except Exception as ex:
                    self.log.error(f'archive_files: could not delete {img_file}: ' + str(ex))
            else:
                try:
                    # move image files
                    for img_file in self.image_file_names:
                        shutil.move(os.path.join(self.root_folder, self.input_folder, img_file), self.output_folder_notrain)
                except Exception as ex:
                    self.log.error(f'archive_files: could not move {img_file} to {self.output_folder_notrain}: ' + str(ex))

            try:
                shutil.move(os.path.join(self.root_folder, self.input_folder, self.video_file_name), self.output_folder_notrain)
            except Exception as ex:
                self.log.error(f'archive_files: could not move {self.video_file_name} to {self.output_folder_notrain}: ' + str(ex))

        self.result["archived"] = True
        return True

    def delete_files(self):
        pass

    # todo: add more metadata
    # - name of location (subdivision & milepost)
    # - GPS coordinates
    # - type of train
    # - weekday
    def store_record(self, ignore_threshold = False):

        data_file = os.path.join(self.root_folder, self.output_root_folder, 'train_sightings.csv')

        # extend to store complete results
        current_record = {'camera': [self.cam_num], 'date': [self.date], 'weekday': [self.week_day], 'time': [self.time], 'file name': [self.base_file_name]}

        df = pd.DataFrame(current_record | self.result)

        # check if output file exists, if yes - append, if no - create new
        try:
            if os.path.isfile(data_file):
                df.to_csv(data_file, mode='a', index=False, header=False)
            else:
                df.to_csv(data_file, index=False)
            return True

        except:
            return False

    # integrated function to process camera
    def process(self):

        #0 - Reload configuration and re-load model if new version
        self.read_configuration()

        #0a - Load new model if different version obtained
        if self.conf_value('nr_version') != self.loaded_nr_version:
            self.nr_disable = False
            self.load_model()

        #1 - Access camera data (download mp4 video to Inbox):
        if not self.conf_value('disable_capture'):
            self.acquire_camera_data(self.cam_num)
        else:
            # quit function, nothing to do.
            return True

        if not self.new_data:
            self.log.debug(f'process: no new data from camera {self.cam_num}, skipping.')
            # quit function, nothing to do.
            return False

        #2a - Extract still frame (always do, so that collected data can be used for training)
        self.extract_frames()

        #2b - Extract still frame
        if not self.nr_disable and self.conf_value('nr_enable'):
            self.classify_nr()

        #3 - Determine motion direction
        if self.conf_value('motion_enable'):
            self.detect_motion()

        #3 - Move files to archive destination
        delete_no_train = self.conf_value('nr_enable') and not self.conf_value('nr_training') # or something similar...
        delete_jpg = False

        self.generate_caption()

        if self.conf_value('nr_enable') or self.conf_value('motion_enable'):
            self.archive_files(delete_no_train, delete_jpg, ignore_threshold = True)

        #4 - Store record
        if self.result["shows_train"]:
            # keep record if a train was detected by the nr model or by motion detection
            self.store_record(ignore_threshold = False)

        #5 - Post on facebook
            if self.conf_value('fb_posting'):
                self.fb.publish_train(self, ignore_threshold = False)

        return True

class fb_helper:

    def __init__(self, page_graph = False):
        # event logging
        self.log = logging.getLogger(__name__)
        self.page_graph = page_graph

    def publish_train(self, train = False, ignore_threshold = False):
        if not train:
            self.log.error('publish_train: no train object provided.')
            return False

        if not self.page_graph:
            self.log.error('publish_train: no page graph provided.')
            return False

        if not train.result["shows_train"]:
            self.log.warning('publish_train: does not show train, so not posting it.')
            return False

        if len(train.image_file_names) == 0:
            self.log.error(f'publish_train: cannot post image for camera {train.cam_num} because image_file_names is empty.')

        # initialize return
        ret = False

        # determine if we should post at all
        if (train.result["detection_count"] <= train.result["detection_threshold"]) or ignore_threshold:

            # if cam_type is "511", we should have a
            if train.video_file_name[-3:] == 'mp4':
                post_video = True
            else:
                post_video = False

            # determine which file to use for posting
            if post_video:
                image_file_name = train.video_file_name
            else:
                image_file_name = train.image_file_names[0]

            # determine path to image file to be posted
            if train.result["archived"]:
                image_location = os.path.join(train.root_folder, train.output_folder_train, image_file_name)
            elif train.result["classified"]:
                image_location = os.path.join(train.root_folder, train.input_folder, image_file_name)
            else:
                self.log.warning(f'publish_train: image {image_file_name} is neither classified nor archived, so cannot post it.')
                return False

            # Construct message for posting
            message_content = train.result["caption"]

            # Publish image or video
            try:
                self.log.info('publish_train: trying to publish ' + image_location + ' with message ' + message_content)

                # determine to post photo or video
                if post_video:
                    # publish video
                    # Construct the Graph API endpoint URL for video uploads
                    endpoint_url = f"https://graph.facebook.com/v12.0/me/videos?access_token={train.fb_access_token}"
                    # Open the video file in binary mode
                    with open(image_location, 'rb') as video_file:
                        # Create a POST request with the video file
                        response = requests.post(endpoint_url, files={'source': video_file}, data={'description': message_content})

                    # Access the video ID from the response
                    ret = response.json()
                    video_id = ret['id']

                    #video_url = f"https://www.facebook.com/video.php?fbid={video_id}"
                    video_url = f"https://www.facebook.com/AutomatedTrainSightings/videos/{video_id}"
                    train.result['fb_url'] = video_url

                    self.log.info(f'publish_train: posted video to {video_url}')
                else:
                    # publish photo
                    ret = self.page_graph.put_photo(open(image_location, "rb"), message = message_content)

                    photo_id = ret['id']
                    photo_url = f"https://www.facebook.com/photo.php?fbid={photo_id}"
                    train.result['fb_url'] = photo_url

                    self.log.info(f'publish_train: posted photo to {photo_url}')
            except Exception as ex:
                self.log.error('publish_train: posting failed: ' + str(ex))
                ret = False
        else:
            self.log.warning('publish_facebook: not posting, detection threshold exceeded.')
            ret = False

        return ret

    def post_start(self, cam_list = []):

        start_message = 'Train detection engine started :)'

        try:
            if len(cam_list) > 0:
                start_message += '\nCameras tracked:'
                for cam in cam_list:
                    if cam.conf_value('fb_posting'):
                        nr_version = cam.conf_value('nr_version')
                        cam_description = cam.conf_value('cam_description')
                        start_message += f'\n {cam.cam_num} (v{nr_version}): {cam_description}'
        except Exception as ex:
            self.log.error('post_start: Could not read camera list information: ' + str(ex))

        try:
            self.page_graph.put_object('me', 'feed', message = start_message)
            self.log.info(f'post_start: Posted detection started, posted message:\n {start_message}')
        except Exception as ex:
            self.log.error('post_start: posting failed: ' + str(ex))

    def post_end(self):
        try:
            self.page_graph.put_object('me', 'feed', message = 'Train detection engine stopped for now. Will post when started again!')
            self.log.info('post_end: Posted detection ended.')
        except Exception as ex:
            self.log.error('post_end: posting failed: ' + str(ex))

"""
class td_manager_connection:
    def __init__(self):

        self.log = logging.getLogger(__name__)

        # Create a socket object
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Define the host and port to listen on
        host = '127.0.0.1'  # localhost
        port = 7429

        # Bind the socket to the host and port
        server_socket.bind((host, port))

        # Listen for incoming connections
        server_socket.listen(1)

        # Accept a client connection
        self.client_socket, self.client_address = server_socket.accept()
        self.log.info('Connected by', self.client_address)

        return True

    # Function to handle receiving data from the client
    def receive_data(self):
        # Receive data from the client
        self.data = self.client_socket.recv(1024).decode()
        if not self.data:
            return False

        self.log.info(f"received: {self.data}")

        return self.data
"""

def main():

    # setup logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level = 'INFO')

    # base configuration
    log.info('Starting TrainDetector.')
    root_OneDrive_folder = 'C:\\Users\\georg\\OneDrive\\Documents\\Eisenbahn\\TrainDetector'
    root_folder = 'C:\\Users\\georg\\Pictures\\TrainDetector'

    configuration_file = 'train_detection_configuration.csv'
    fb_credential_file = 'page_credentials.json'

    # Read configuration and determine cameras
    log.info('Reading configuration file.')
    configuration = pd.read_csv(os.path.join(root_folder, configuration_file), encoding = "ISO-8859-1")
    log.info(f'Loaded data for {len(configuration)} cameras.')

    # construct camera list
    log.info('Initializing cameras.')
    cam_list = []
    for cam_id in configuration['cam_id']:
        cam_list.append(image_capture(root_folder, 'inbox', 'results', cam_id, configuration_file, fb_credential_file))

    # capture data
    log.info('Start capturing process.')
    #cam_list[0].fb.post_start(cam_list)

    try:
        while(True):
            time.sleep(18)
            for cam in cam_list:
                cam.process()
    except KeyboardInterrupt:
        pass

    log.info('Terminating capturing process.')
    #cam_list[0].fb.post_end()

if __name__ == '__main__':
    main()

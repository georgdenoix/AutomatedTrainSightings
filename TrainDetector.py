#-------------------------------------------------------------------------------
# Name:        TrainDetector
# Purpose:     Collect traffic camera images from 511 Québec and classiy them for passing trains
#
# Author:      Georg Denoix
#
# Updated:     17-05-2023
# Copyright:   (c) Georg Denoix 2023
# Licence:     open source
#-------------------------------------------------------------------------------

import requests, os, time, cv2, shutil, math, hashlib, imutils, logging
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import json
from facebook import GraphAPI
import sys
import sqlite3
from google.transit import gtfs_realtime_pb2


class image_capture:

    def __init__(self, root_folder = '', input_root_folder = 'inbox', output_root_folder = 'results', cam_num = 0, configuration_file = 'train_detection_configuration.csv', fb_credential_file = '', db_file = ''):

        # event logging
        self.log = logging.getLogger(__name__)

        # base parameters
        self.configuration_file = configuration_file
        self.fb_credential_file = fb_credential_file
        self.db_file = db_file
        self.csv_file = 'train_sightings.csv'
        self.fb_access_token = ''
        self.root_folder = root_folder
        self.base_file_name = ''
        self.image_file_names = []
        self.file_name = ''
        self.cam_num = cam_num

        # camera data
        self.new_data = False
        self.last_data_checksum = 0x00

        # classification results
        self.nearby_trains = []
        self.record = {
            'camera': self.cam_num,
            'date': '',
            'weekday': '',
            'time': '',
            'file_path': '',
            'file_name': '',
            'file_type': '',
            'shows_train': False,
            'motion_value': 0,
            'motion_detected': False,
            'classified': False,
            'nr_confidence': 0,
            'direction': '',
            'archived': False,
            'detection_count': 0,
            'detection_threshold': 1,
            'cnt_pic': 0,
            'cnt_train': 0,
            'cnt_no_train': 0,
            'caption': '',
            'fb_id': '',
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
        self.train_location_timeout = 150
        self.train_location_distance = 850

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
        self.record["shows_train"] = False
        self.record["motion_value"] = 0
        self.record["motion_detected"] = False
        self.record["classified"] = False
        self.record["archived"] = False
        self.record["fb_id"] = ''
        self.record["fb_url"] = ''
        self.record["direction"] = ''
        self.record["file_name"] = ''
        self.nearby_trains = []

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
        self.record["date"] = datetime.now().strftime("%Y-%m-%d")
        self.record["time"] = datetime.now().strftime("%H:%M:%S")
        self.record["weekday"] = datetime.now().strftime("%w")

        self.base_file_name = f'{cam_num}_{time_str}'

        # construct new file name
        last_file_name = self.file_name
        if self.conf_value('cam_type') == '511':
            cam_URL = f'https://www.quebec511.info/Carte/Fenetres/camera.ashx?id={self.cam_num}&format=mp4'
            self.file_name = f'{self.base_file_name}.mp4'
            self.record["file_type"] = 'mp4'

        elif self.conf_value('cam_type') == 'vdm':
            cam_URL = f'https://ville.montreal.qc.ca/Circulation-Cameras/GEN{self.cam_num}.jpeg'
            self.file_name = f'{self.base_file_name}.jpeg'
            self.record["file_type"] = 'jpg'

        elif self.conf_value('cam_type') == 'vdo':
            cam_URL = f'https://traffic.ottawa.ca/beta/camera?id={self.cam_num}'
            self.file_name = f'{self.base_file_name}.jpg'
            self.record["file_type"] = 'jpg'

        else:
            cam_type = self.conf_value('cam_type')
            self.record["file_type"] = ''
            self.log.error(f'acquire_camera_data: no URL for camera of type: ({cam_type})')

        # store file name in record too
        self.record["file_name"] = self.file_name

        input_file_path = os.path.join(self.root_folder, self.input_folder, self.file_name)
        try:
            # open video file for writing
            video_file = open(input_file_path, 'wb')

            # access video stream
            response = requests.get(cam_URL, stream = True, verify = False)
        except Exception as ex:
            self.log.error(f'acquire_camera_data: something went wrong downloading data from {cam_URL}: ' + str(ex))
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
        new_checksum = hashlib.md5(open(input_file_path, 'rb').read()).hexdigest()
        self.new_data = new_checksum != self.last_data_checksum

        if self.new_data:
            self.last_data_checksum = new_checksum
            self.record["cnt_pic"] += 1

        else:
            #restore last file name
            self.file_name = last_file_name

            # delete new file (because its content is identical yo the last file
            try:
                os.remove(input_file_path)
            except Exception as ex:
                self.log.error('acquire_camera_data: ' + str(ex))

        # update record with file path
        self.record["file_path"] = self.input_folder

        return True

    # todo: make number of extracted images variable
    def extract_frames(self):

        full_file_name = os.path.join(self.root_folder, self.input_folder, self.file_name)

        if not os.path.isfile(full_file_name):
            self.log.error('extract_frames: video file {full_file_name} does not exist.')
            return False

        self.image_file_names = []

        # if camera delivers mp4 video, extract frames:
        if self.conf_value('cam_format') == 'mp4':
            cam = cv2.VideoCapture(full_file_name)

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
            self.image_file_names.append(self.file_name)

    # todo: add classification fo type of train
    def classify_nr(self):

        if self.nr_disable:
            self.log.warning(f'classify_nr: skipping classification for {self.cam_num} because nr_disable = True')
            return False

        self.record["shows_train"] = False

        try:
            if len(self.image_file_names) > 0:
                for imgFile in self.image_file_names:
                    if not self.record["shows_train"]:
                        try:
                            # open image and pre-process for use with model
                            img_path = os.path.join(self.root_folder, self.input_folder, imgFile)
                            img = tf.keras.utils.load_img(img_path, target_size=(256, 256))
                            img = tf.keras.utils.img_to_array(img)
                            img = np.expand_dims(img, axis=0)
                        except Exception as ex:
                            self.log.error(f'classify_nr: could not open {imgFile}: ' + str(ex))
                            pass

                        try:
                            # use model to predict if this image contains a train
                            self.record["nr_confidence"] = self.nr_model.predict(img)[0][0]

                            # If confidence is more than 0.5, probably found a train:
                            if (self.record["nr_confidence"] > 0.5):
                                # set result to "train":
                                self.record["shows_train"] = True
                                self.record["cnt_train"] += 1
                                self.record["detection_count"] += 1
                            else:
                                # set result to "no train":
                                self.record["shows_train"] = False
                                self.record["cnt_no_train"] += 1
                                self.record["detection_count"] = 0

                            self.record["classified"] = True
                            self.log.info(f'classify_nr: {imgFile} is showing a train: {self.record["nr_confidence"]} => {self.record["shows_train"]}, detection count: {self.record["detection_count"]}, threshold: {self.record["detection_threshold"]}; cnt: {self.record["cnt_pic"]}, train: {self.record["cnt_train"]}, no_train: {self.record["cnt_no_train"]}')

                        except Exception as ex:
                            self.log.error(f'classify_nr: something went wrong processing {img_path}: ' + str(ex))
                            pass
                    else:
                        self.record["classified"] = True
                        self.log.warning(f'classify: skipping {imgFile}, train already detected.')

        except Exception as ex:
            self.log.error('classify_nr: ran into a problem: ', str(ex))

        return self.record["shows_train"]

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
        self.record["motion_detected"] = False

        # Read video from file
        input_file_path = os.path.join(self.root_folder, self.input_folder, self.file_name)
        video = cv2.VideoCapture(input_file_path)

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
        self.record["motion_value"] = X
        if X > self.conf_value('motion_threshold'):
            self.record["motion_detected"] = True
            self.record["direction"] = self.conf_value('right_orientation')
            self.log.info(f'Objects on camera {self.cam_num} were mostly moving RIGHT')

        elif X < -self.conf_value('motion_threshold'):
            self.record["motion_detected"] = True
            self.record["direction"] = self.conf_value('left_orientation')
            self.log.info(f'Objects on camera {self.cam_num} were mostly moving LEFT')

        else:
            self.record["direction"] = 'no motion detected'
            self.log.info(f'NO MOTION was detected on camera {self.cam_num}.')


        # Code below should be in separate function, re-use from
        if self.record["motion_detected"] and self.conf_value('det_motion_only'):
            self.record["shows_train"] = True
            self.record["cnt_train"] += 1
            self.record["detection_count"] += 1
            self.log.info(f'{self.cam_num} shows movement and is configured to recognize train without NR.')
            self.log.info(f'classify_nr: {self.cam_num} is showing a train: {self.record["shows_train"]}, detection count: {self.record["detection_count"]}, threshold: {self.record["detection_threshold"]}; cnt: {self.record["cnt_pic"]}, train: {self.record["cnt_train"]}, no_train: {self.record["cnt_no_train"]}')

        elif self.conf_value('det_motion_only'):
            self.record["shows_train"] = False
            self.record["cnt_no_train"] += 1
            self.record["detection_count"] = 0

        return True

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the distance between two GPS coordinates using the Haversine formula.
        Args:
            lat1 (float): Latitude of the first coordinate.
            lon1 (float): Longitude of the first coordinate.
            lat2 (float): Latitude of the second coordinate.
            lon2 (float): Longitude of the second coordinate.
        Returns:
            float: Distance between the two coordinates in meters.
        """
        # Convert latitude and longitude to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Radius of the Earth in meters
        earth_radius = 6371000

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = earth_radius * c

        return distance

    # function to get trains from operator
    def get_trains(self, agency_feed = False, agency_name = ''):

        # initialize empty train list
        nearby_trains = []
        rejected_trains = []

        # Check for valif agency
        agencies = ['VIA', 'exo']

        if not agency_feed or not (agency_name in agencies):
            log.error(f'get_trains: no agency feed or unknown agency name: {agency_name}')
            return nearby_trains

        #2 get camera coordinates
        lon = self.conf_value('longitude')
        lat = self.conf_value('latitude')

        if (lon == 0) or (lat == 0):
            self.log.debug(f'get_trains: skipping, location of camera {self.cam_num} is not set.')
            return nearby_trains
        else:
            self.log.debug(f'get_trains: comparing train positions with camera location {lat}, {lon}.')

        # Read trips and routes from static GTFS
        try:
            if agency_name == 'VIA':
                trips = pd.read_csv(os.path.join(self.root_folder, 'models', 'gtfs', 'viarail', 'trips.txt'))
                routes = pd.read_csv(os.path.join(self.root_folder, 'models', 'gtfs', 'viarail', 'routes.txt'))

            elif agency_name == 'exo':
                trips = pd.read_csv(os.path.join(self.root_folder, 'models', 'gtfs', 'exo', 'trips.txt'))
                routes = pd.read_csv(os.path.join(self.root_folder, 'models', 'gtfs', 'exo', 'routes.txt'))

        except Exception as ex:
            trips = False
            routes = False
            self.log.error('get_trains: ' + str(ex))

        # loop through trains:
        for train in agency_feed.entity:

            if train.HasField('vehicle'):
                vehicle = train.vehicle

                # age of record
                timedelta = datetime.now() - datetime.fromtimestamp(vehicle.timestamp)

                # distance:
                dst = self.calculate_distance(lat, lon, vehicle.position.latitude, vehicle.position.longitude)
                dst_str = str(round(dst)) # distance in m

                # initialize result string
                train_str = ''
                train_no = ''
                train_destination = ''
                route_name = ''

                # resolve train no. from trips table
                try:
                    # because route_id is numeric, pandas stores it as integer
                    try:
                        trip_id = int(vehicle.trip.trip_id)
                    except:
                        trip_id = vehicle.trip.trip_id

                    condition = (trips['trip_id'] == trip_id)

                    train_no_list = trips.loc[condition, 'trip_short_name'].values
                    if len(train_no_list) > 0:
                        train_no = str(train_no_list[0])

                    train_destination_list = trips.loc[condition, 'trip_headsign'].values
                    if len(train_destination_list) > 0:
                        train_destination = train_destination_list[0]

                except Exception as ex:
                    self.log.error('get_trains: ' + str(ex))

                # resolve route name from routes table
                try:
                    # because route_id is numeric, pandas stores it as integer
                    try:
                        route_id = int(vehicle.trip.route_id)
                    except:
                        route_id = vehicle.trip.route_id

                    condition = (routes['route_id'] == route_id)
                    route_name_list = routes.loc[condition, 'route_long_name'].values
                    if len(route_name_list) > 0:
                        route_name = route_name_list[0]

                except Exception as ex:
                    self.log.error('get_trains: ' + str(ex))


                # assign vehicle id and route number
                try:
                    if agency_name == 'VIA':
                        try:
                            asm_url = f'https://asm.transitdocs.com/train/{vehicle.trip.start_date[0:4]}/{vehicle.trip.start_date[4:6]}/{vehicle.trip.start_date[6:8]}/V/{vehicle.vehicle.id[15:]}'
                        except:
                            asm_url = ''
                    else:
                        asm_url = ''

                    if agency_name == 'exo':
                        vehicle_id = vehicle.vehicle.id
                        train_str = f'EXO {vehicle_id} on '

                    # build train string
                    train_str += f'{agency_name} {train_no}, route: {route_name}, destination: {train_destination}'
                    if (len(asm_url) > 0) and agency_name == 'VIA':
                        train_str += f', details: {asm_url}'

                    # log
                    self.log.debug(f'get_trains: found train {train_str}, at distance: {dst_str} m, age: {timedelta}')
                except Exception as ex:
                    self.log.error(f'get_trains: ' + str(ex))
                    train_str = vehicle.vehicle.id

                # check distance and time limit
                if (dst <= self.train_location_distance) and (timedelta.seconds <= self.train_location_timeout):
                    nearby_trains.append(train_str)
                else:
                    rejected_trains.append(train_str)

        return nearby_trains

    # funtion to determine closest VIA train to camera location
    def find_closest_train(self):

        try:
            #1a get EXO feed
            # read exo credentials from external file (todo: could make file name configurable...)
            exo_cred_file = os.path.join(self.root_folder, 'exo_credentials.txt')
            exo_creds = ''
            try:
                with open(exo_cred_file, 'r') as f:
                    exo_creds = str(f.read()).strip()
            except Exception as ex:
                self.log.error('find_closest_train: trying to read exo credentials: ' + str(ex))

            exo_vehicle_url = f'https://opendata.exo.quebec/ServiceGTFSR/VehiclePosition.pb?token={exo_creds}&agency=TRAINS'

            response = requests.get(exo_vehicle_url)
            if response.status_code == 200:
                exo_feed = gtfs_realtime_pb2.FeedMessage()
                exo_feed.ParseFromString(response.content)
            else:
                self.log.error('find_closest_train: could not access EXO GTFS feed, error code: ' + str(response.status_code))
                exo_feed = False

            #1b get VIA feed
            via_feed_url = 'https://asm-backend.transitdocs.com/gtfs/via'
            response = requests.get(via_feed_url)
            if response.status_code == 200:
                via_feed = gtfs_realtime_pb2.FeedMessage()
                via_feed.ParseFromString(response.content)
            else:
                self.log.error('find_closest_train: could not access VIA GTFS feed, error code: ' + str(response.status_code))
                via_feed = False

            #3 loop through all trains and determine distance

            # check for VIA trains:
            if not via_feed:
                pass
            else:
                via_trains = self.get_trains(via_feed, 'VIA')

            # check for EXO trains:
            if not exo_feed:
                pass
            else:
                # now go through vehicles for location and assign train numbers from trips
                exo_trains = self.get_trains(exo_feed, 'exo')

            self.nearby_trains = via_trains + exo_trains

            # log results
            if len(self.nearby_trains) > 0:
                self.log.info(f'find_closest_train: accepted trains on cam {self.cam_num}: ' + str(self.nearby_trains))

        except Exception as ex:
            self.log.error('find_closest_train: error: ' + str(ex))

        return self.nearby_trains


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

        if self.record["motion_detected"]:
            if self.record["motion_value"] > 0:
                motion_text = 'heading ' + self.conf_value('right_orientation')
            else:
                motion_text = 'heading ' + self.conf_value('left_orientation')

        caption = 'Train ' + motion_text + ' detected at ' + self.record["time"] + ' on camera ' + cam_text + ' [' + str(self.cam_num) + '] on ' + self.record["date"] + '.'

        # NEW: try to find matching trains
        try:
            possible_trains = self.find_closest_train()
            if len(possible_trains) > 0:
                caption += ' Train on picture: ' + ' or '.join(possible_trains)
                pass
        except Exception as ex:
            self.log.error('generate_caption: ' + str(ex))
            pass

        self.record["caption"] = caption

        return caption


    # function to archive current record
    def archive_files(self, delete_no_train = True, delete_jpg = False, ignore_threshold = False):

        if self.output_folder_notrain == '':
            self.log.error(f'archive_files: output folder not defined.')
            return False

        if self.output_folder_train == '':
            self.log.error(f'archive_files: output folder not defined.')
            return False

        # set (create) output folder, just to be sure
        self.set_output_folder()

        source_file = os.path.join(self.root_folder, self.input_folder, self.file_name)

        # Image shows a train, so move it to the right folder
        if self.record["shows_train"] and ((self.record["detection_count"] <= self.record["detection_threshold"]) or ignore_threshold):

            # Showing train
            if delete_jpg and self.record["file_type"] == 'mp4':
                # delete (do not archive) extracted jpg files from video
                try:
                    for img_file in self.image_file_names:
                        os.remove(os.path.join(self.root_folder, self.input_folder, img_file))
                except Exception as ex:
                    self.log.error(f'archive_files: could not delete {img_file}: ' + str(ex))

            elif self.record["file_type"] == 'mp4':
                # move extracted jpg files
                try:
                    # move image files
                    for img_file in self.image_file_names:
                        shutil.move(os.path.join(self.root_folder, self.input_folder, img_file), self.output_folder_train)
                except Exception as ex:
                    self.log.error(f'archive_files: could not move {img_file} to {self.output_folder_train}: ' + str(ex))

            try:
                # move main image/video file
                if os.path.isfile(source_file):
                    shutil.move(source_file, self.output_folder_train)
                self.record["file_path"] = self.output_folder_train
            except Exception as ex:
                self.log.error(f'archive_files: could not move {self.file_name} to {self.output_folder_train}: ' + str(ex))

        elif delete_no_train:
            # delete extracted image files
            try:
                for img_file in self.image_file_names:
                    os.remove(os.path.join(self.root_folder, self.input_folder, img_file))
            except Exception as ex:
                self.log.error(f'archive_files: could not delete {img_file}: ' + str(ex))

            # delete main image/video file
            try:
                os.remove(source_file)
            except Exception as ex:
                self.log.error(f'archive_files: could not delete {self.file_name}: ' + str(ex))

        else:
            # treat as "no_train"
            if delete_jpg:
                try:
                    for img_file in self.image_file_names:
                        os.remove(os.path.join(self.root_folder, self.input_folder, img_file))
                except Exception as ex:
                    self.log.error(f'archive_files: could not delete {img_file}: ' + str(ex))
            else:
                try:
                    # move image files (no_train)
                    for img_file in self.image_file_names:
                        shutil.move(os.path.join(self.root_folder, self.input_folder, img_file), self.output_folder_notrain)
                except Exception as ex:
                    self.log.error(f'archive_files: could not move {img_file} to {self.output_folder_notrain}: ' + str(ex))

            try:
                if os.path.isfile(source_file):
                    shutil.move(source_file, self.output_folder_notrain)
                self.record["file_path"] = self.output_folder_notrain
            except Exception as ex:
                self.log.error(f'archive_files: could not move {self.file_name} to {self.output_folder_notrain}: ' + str(ex))

        self.record["archived"] = True
        return True

    # prototype function to clean up old data?
    def delete_files(self):
        pass

    # todo: add more metadata
    # - name of location (subdivision & milepost) >> camera metadata
    # - GPS coordinates >> should be in metadata for cameras
    # - type of train

    def store_record(self, ignore_threshold = False, to_csv = False, to_db = False):

        """
        Rework to store complete "record" into database (plus csv file if requested)
        """
        try:
            df = pd.DataFrame(self.record, index = [0])
        except Exception as ex:
            self.log.error(f'store_record: error creating data frame: ' + str(ex))

        # store to db
        if to_db:
            try:
                db_conn = sqlite3.connect(os.path.join(self.root_folder, self.output_root_folder, self.db_file))

                try:
                    ret = df.to_sql('trains', db_conn, if_exists = 'append', index = False)
                    self.log.debug(f'store_record: added {ret} records to database {self.db_file}.')
                except Exception as ex:
                    self.log.error(f'store_record: error storing record to database {self.db_file}: ' + str(ex))
                finally:
                    db_conn.close()

            except Exception as ex:
                self.log.error(f'store_record: error storing record to database {self.db_file}: ' + str(ex))

        # Save to csv file
        if to_csv:
            data_file = os.path.join(self.root_folder, self.output_root_folder, self.csv_file)
            # check if output file exists, if yes - append, if no - create new
            try:
                if os.path.isfile(data_file):
                    df.to_csv(data_file, mode='a', index=False, header=False)
                else:
                    df.to_csv(data_file, index=False)

            except Exception as ex:
                self.log.error(f'store_record: could not write to csv file: ' + str(ex))

        return True

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

        #4 - Move files to archive destination, could be extended to become
        #    configurable per camera (from config file)
        delete_no_train = self.conf_value('nr_enable') and not self.conf_value('nr_training') # or something similar...
        delete_jpg = False

        self.generate_caption()

        if self.conf_value('nr_enable') or self.conf_value('motion_enable'):
            self.archive_files(delete_no_train, delete_jpg, ignore_threshold = True)

        # Store record
        if self.record["shows_train"]:
            #5 - Post on facebook (need to do *before* storing record, to capture fb id)
            if self.conf_value('fb_posting'):
                # ignore threshold when VIA/EXO train is nearby:  === does not work well on Ottawa cameras with lots of captures per train passing, will create multiple postings, need to find different approach
                #ignore_threshold = (len(self.nearby_trains) > 0)
                ignore_threshold = False
                self.fb.publish_train(self, ignore_threshold)

            #6 keep record if a train was detected by the nr model or by motion detection
            self.store_record(ignore_threshold = False, to_csv = True, to_db = True)

        return True

class fb_helper:

    def __init__(self, page_graph = False):
        # event logging
        self.log = logging.getLogger(__name__)
        self.page_graph = page_graph

    # function to publish photo
    def publish_photo(self, train = False, image_location = False, message_content = False):

        self.log.info(f'publish_photo: trying to publish photo {image_location} with message {message_content}')
        try:
            ret = self.page_graph.put_photo(open(image_location, "rb"), message = message_content)

            photo_id = ret['id']
            train.record['fb_id'] = photo_id

            photo_url = f"https://www.facebook.com/photo.php?fbid={photo_id}"
            train.record['fb_url'] = photo_url

            self.log.info(f'publish_photo: posted photo to {photo_url}')

        except Exception as ex:
            train.record['fb_id'] = ''
            train.record['fb_url'] = ''
            self.log.error('publish_photo: error ' + str(ex))
            return False

        return True

    # function to publish video
    def publish_video(self, train = False, image_location = False, message_content = False):

        self.log.info(f'publish_video: trying to publish video {image_location} with message {message_content}')
        if not train or not image_location or not message_content:
            return False

        ret_val = False

        try:
            # Construct the Graph API endpoint URL for video uploads
            endpoint_url = f"https://graph.facebook.com/v12.0/me/videos?access_token={train.fb_access_token}"

            # Open the video file in binary mode
            with open(image_location, 'rb') as video_file:
                # Create a POST request with the video file
                response = requests.post(endpoint_url, files={'source': video_file}, data={'description': message_content})

            # Access the video ID from the response
            ret = response.json()
            if 'id' in ret.keys():
                video_id = ret['id']
                #video_url = f"https://www.facebook.com/video.php?fbid={video_id}"
                video_url = f"https://www.facebook.com/AutomatedTrainSightings/videos/{video_id}"

                self.log.info(f'publish_video: posted video to {video_url}')
                ret_val = True

            elif 'error' in ret.keys():
                video_id = 0
                video_url = '(not posted)'
                self.log.error('publish_video: posting failed: ' + str(ret['error']))

            train.record["fb_id"] = video_id
            train.record['fb_url'] = video_url

        except Expection as ex:
            self.log.error(f'publish_video: error ' + str(ex))
            return False

        return ret_val

    def publish_train(self, train = False, ignore_threshold = False):
        if not train:
            self.log.error('publish_train: no train object provided.')
            return False

        if not self.page_graph:
            self.log.error('publish_train: no page graph provided.')
            return False

        if not train.record["shows_train"]:
            self.log.warning('publish_train: does not show train, so not posting it.')
            return False

        if len(train.image_file_names) == 0:
            self.log.error(f'publish_train: cannot post image for camera {train.cam_num} because image_file_names is empty.')

        # initialize return
        ret = False

        # determine if we should post at all
        if (train.record["detection_count"] <= train.record["detection_threshold"]) or ignore_threshold:

            # check if main file is a video file
            if train.file_name[-3:] == 'mp4':
                post_video = True
            else:
                post_video = False

            # something's wrong with the video posts... >> disable here:
            post_video = False

            # determine which file to use for posting
            if post_video:
                image_file_name = train.file_name
            else:
                image_file_name = train.image_file_names[0]

            # determine path to image file to be posted
            if train.record["archived"]:
                image_location = os.path.join(train.root_folder, train.output_folder_train, image_file_name)
            elif train.record["classified"]:
                image_location = os.path.join(train.root_folder, train.input_folder, image_file_name)
            else:
                self.log.warning(f'publish_train: image {image_file_name} is neither classified nor archived, so cannot post it.')
                return False

            # Construct message for posting
            message_content = train.record["caption"]

            # initialize ret
            ret = False
            photo_id = 0
            photo_url = "(not posted)"

            # Publish image or video
            try:
                # determine to post photo or video
                if post_video:
                    # publish video
                    if not self.publish_video(train, image_location, message_content):
                        self.publish_photo(train, image_location, message_content)

                else:
                    self.publish_photo(train, image_location, message_content)

            except Exception as ex:
                self.log.error('publish_train: posting failed: ' + str(ex) + ' ret:' + str(ret))
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

def main():

    requests.packages.urllib3.disable_warnings()  # Disable SSL warnings

    # Set the root logger level to the lowest level you want to capture
    logging.getLogger().setLevel(logging.DEBUG)

    # Create a formatter for the log messages
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # Create a console handler and set its level to capture info-level messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Create a file handler and set its level to capture debug-level messages
    file_handler = logging.FileHandler("debug.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Get the root logger and add the handlers
    log = logging.getLogger()
    log.addHandler(console_handler)
    log.addHandler(file_handler)


    # base configuration
    log.info('Starting TrainDetector.')
    root_OneDrive_folder = 'C:\\Users\\georg\\OneDrive\\Documents\\Eisenbahn\\TrainDetector'
    root_folder = 'C:\\Users\\georg\\Pictures\\TrainDetector'

    configuration_file = 'train_detection_configuration.csv'
    fb_credential_file = 'page_credentials.json'
    database_file = 'train_sightings.db'

    # Read configuration and determine cameras
    log.info('Reading configuration file.')
    try:
        configuration = pd.read_csv(os.path.join(root_folder, configuration_file), encoding = "ISO-8859-1")
    except Exception as ex:
        log.error(f'Error reading configuration {configuration_file}: ' + str(ex))

    log.info(f'Loaded data for {len(configuration)} cameras.')

    # construct camera list
    log.info('Initializing cameras.')
    cam_list = []
    try:
        for cam_id in configuration['cam_id']:
            cam_list.append(image_capture(root_folder = root_folder, input_root_folder = 'inbox', output_root_folder = 'results', cam_num = cam_id, configuration_file = configuration_file, fb_credential_file = fb_credential_file, db_file = database_file))
    except Exception as ex:
        log.error(f'Error setting up image captures: ' + str(ex))

    # capture data
    log.info('Start capturing process.')
    #cam_list[0].fb.post_start(cam_list)

    try:
        while(True):
            for cam in cam_list:
                try:
                    cam.process()
                except Exception as ex:
                    log.error(f'Error processing camera: {cam.cam_num}' + str(ex))

            time.sleep(18)

    except KeyboardInterrupt:
        pass

    log.info('Terminating capturing process.')
    #cam_list[0].fb.post_end()

if __name__ == '__main__':
    main()

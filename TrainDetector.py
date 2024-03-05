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

import requests, os, time, cv2, shutil, math, hashlib, imutils, logging, logging.handlers
from datetime import datetime, timedelta
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import json
from facebook import GraphAPI
import sys
import sqlite3
from google.transit import gtfs_realtime_pb2

class train_location:

    def __init__(self, agencies = [], static_gtfs_folder = ''):

        self.agencies = agencies

        # event logging
        self.log = logging.getLogger(__name__)

        # configure static gtfs data folder
        self.static_gtfs_folder = static_gtfs_folder

        # list of valid agencies
        self.valid_agencies = ['via', 'exo']

        # initialize data
        self.reset(agencies.copy())

        # initialize list of trains
        self.train_list = []

        pass

    def read_credentials(self, agency = ''):

        creds = ''

        if agency in self.agency_credential_file.keys():
            cred_file = os.path.join(self.agency_credential_file[agency])
        else:
            self.log.error(f'read_credentials: {self.agency_credential_file} does not contain information for {agency}.')
            return creds

        try:
            with open(cred_file, 'r') as f:
                creds = str(f.read()).strip()

        except Exception as ex:
            self.log.error('read_credentials: trying to read credentials for {agency} from file {cred_file}: ' + str(ex))

        return creds

    def reset(self, agency_list = []):

        # create copy of list to remove any link with self.agencies from function call
        agency_list_c = agency_list.copy()

        # reset agency list
        self.agencies = []

        # if supplied in parameter, use that
        for agency in agency_list_c:
            if agency in self.valid_agencies:
                self.agencies.append(agency)

        # read from gtfs folder otherwise
        if len(self.agencies) == 0 and os.path.isdir(self.static_gtfs_folder):
            gtfs_agencies = os.listdir(self.static_gtfs_folder)
            for agency in gtfs_agencies:
                if agency in self.valid_agencies:
                    self.agencies.append(agency)

        # initialise empty train list
        self.trains = []

        self.agency_credential_file = {
            'exo': 'exo_credentials.txt'
        }

        self.agency_credential = {
            'exo': ''
        }

        for agency in self.agency_credential.keys():
            self.agency_credential[agency] = self.read_credentials(agency)

        self.agency_feed_url = {
            'via': 'https://asm-backend.transitdocs.com/gtfs/via',
            'exo': f'https://opendata.exo.quebec/ServiceGTFSR/VehiclePosition.pb?token={self.agency_credential["exo"]}&agency=TRAINS'
        }

        # read trips and routes for each agency
        self.trips = {}
        self.routes = {}

        for agency in self.agencies:
            trips_file = os.path.join(self.static_gtfs_folder, agency, 'trips.txt')
            if os.path.isfile(trips_file):
                self.trips[agency] = pd.read_csv(trips_file)
            else:
                self.trips[agency] = False

            routes_file = os.path.join(self.static_gtfs_folder, agency, 'routes.txt')
            if os.path.isfile(routes_file):
                self.routes[agency] = pd.read_csv(routes_file)
            else:
                self.routes[agency] = False

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

    def train_description(self, train_info = {}):

        train_str = ''

        try:

            if train_info['agency'] == 'exo':
                train_str = f'EXO {train_info["vehicle_id"]} on '

            # build train string
            train_str += f'{train_info["agency"]} {train_info["id"]}, route: {train_info["route_name"]}, destination: {train_info["destination"]}'
            if (len(train_info["asm_url"]) > 0) and train_info['agency'] == 'via':
                train_str += f', details: {train_info["asm_url"]}'

        except Exception as ex:
            self.log.error('train_description: ' + str(ex))

        return train_str

    # generic function to get trains from all agencies
    def read_trains(self, agencies = []):

        # initialize empty train list
        train_list = []

        # if no list of agencies supplied, go through all
        if len(agencies) == 0:
            agencies = self.agencies

        # go through all agencies
        for agency in agencies:

            # do not process if agency is not in list of valid agencies
            if not agency in self.valid_agencies:
                self.log.error(agency + ' is not in ' + str(self.valid_agencies))
                continue

            try:
                # read train list from live gtfs feed
                response = requests.get(self.agency_feed_url[agency])
                if response.status_code == 200:
                    vehicles = gtfs_realtime_pb2.FeedMessage()
                    vehicles.ParseFromString(response.content)
                else:
                    self.log.error(f'get_trains: error accessing gtfs feed from {agency} at {self.agency_feed_url[agency]}: {response.status_code}')
                    continue
            except Exception as ex:
                self.log.error(f'get_trains: error accessing gtfs feed from {agency} at {self.agency_feed_url[agency]}: {ex}')
                continue

            #collect train information
            for train in vehicles.entity:

                if train.HasField('vehicle'):
                    vehicle = train.vehicle

                    #self.log.debug('train_list is currently ' + str(len(train_list)) + ' elements long. Current agency: ' + agency + ', current train: ' + train.id)

                    for stored_train in train_list:
                        if stored_train['trip_id'] == vehicle.trip.trip_id:
                            self.log.debug('duplicate trip ' + vehicle.trip.trip_id)
                            continue

                    train_info = {}

                    train_info['agency'] = agency
                    train_info['label'] = agency[:3]
                    train_info['vehicle_id'] = vehicle.vehicle.id

                    # resolve train no. from trips table
                    try:
                        try:
                            trip_id = vehicle.trip.trip_id if isinstance(vehicle.trip.trip_id, int) else int(vehicle.trip.trip_id)   # fixed
                        except:
                            trip_id = vehicle.trip.trip_id

                        condition = (self.trips[agency]['trip_id'] == trip_id)
                        train_info['trip_id'] = vehicle.trip.trip_id

                        trip_short_name = self.trips[agency].loc[condition, 'trip_short_name'].values
                        if len(trip_short_name) > 0:
                            train_info['id'] = str(trip_short_name[0])
                        else:
                            self.log.error(f'read_trains: could not find matching "trip_short_name" (train number) in {agency} trips for trip_id {trip_id} and vehicle {vehicle.vehicle.id}.')
                            train_info['id'] = ''

                        trip_headsign = self.trips[agency].loc[condition, 'trip_headsign'].values
                        if len(trip_headsign) > 0:
                            train_info['destination'] = str(trip_headsign[0])
                        else:
                            self.log.error(f'read_trains: could not find matching "trip_headsign" (destination) in {agency} trips for trip_id {trip_id} and vehicle {vehicle.vehicle.id}.')
                            train_info['destination'] = ''

                    except Exception as ex:
                        self.log.error(ex)
                        pass

                    # resolve route name from routes table
                    try:
                        try:
                            route_id = vehicle.trip.route_id if isinstance(vehicle.trip.route_id, int) else int(vehicle.trip.route_id)  # fixed
                        except:
                            route_id = vehicle.trip.route_id

                        condition = (self.routes[agency]['route_id'] == route_id)

                        route_long_name = self.routes[agency].loc[condition, 'route_long_name'].values
                        if len(route_long_name) > 0:
                            train_info['route_name'] = str(route_long_name[0])
                        else:
                            self.log.error(f'read_trains: could not find matching "route_long_name" (route name) in {agency} routes for route_id {route_id} and vehicle {vehicle.vehicle.id}.')
                            train_info['route_name'] = ''

                    except Exception as ex:
                        self.log.error(ex)
                        pass

                    # if agency is via, construct asm_url
                    if agency == 'via':
                        try:
                            train_info['asm_url'] = f'https://asm.transitdocs.com/train/{vehicle.trip.start_date[0:4]}/{vehicle.trip.start_date[4:6]}/{vehicle.trip.start_date[6:8]}/V/{vehicle.vehicle.id[15:]}'
                        except:
                            train_info['asm_url'] = ''
                    else:
                        train_info['asm_url'] = ''

                    # train position
                    train_info['latitude'] = vehicle.position.latitude
                    train_info['longitude'] = vehicle.position.longitude

                    # train speed
                    train_info['speed'] = str(round(vehicle.position.speed, 1))

                    # time stamp
                    train_info['time'] = datetime.fromtimestamp(vehicle.timestamp)

                    # generate train description
                    #train_info['description'] = self.train_description(train_info)

                    # append to list
                    train_list.append(train_info)
                else:
                    pass
                    #self.log.debug('skipping: ' + str(train.id))

        # store train list internally
        self.train_list = train_list

        self.log.info(f'train_location.read_trains: read information about {len(train_list)} trains.')

        # also return train list
        return train_list

    def nearby_trains(self, location = (0, 0), max_dist = -1, max_age = -1):

        # initialize empty train list
        nearby_trains = []
        rejected_trains = []

        #2 get reference coordinates
        lat = location[0]
        lon = location[1]

        if (lon == 0) or (lat == 0):
            self.log.debug(f'nearby_trains: skipping, reference location is not set.')
            return nearby_trains
        else:
            self.log.debug(f'nearby_trains: comparing train positions with camera location {lat}, {lon}.')

        # loop through trains:
        for train_info in self.train_list:

            timedelta = datetime.now() - train_info['time']

            # distance:
            dst = self.calculate_distance(lat, lon, train_info['latitude'], train_info['longitude'])
            dst_str = str(round(dst)) # distance in m

            # check distance and time limit
            if ((max_dist == -1) or (dst <= max_dist)) and ((max_age == -1) or (timedelta.seconds <= max_age)):
                nearby_trains.append(train_info)
                self.log.debug(f'nearby_trains: accepted train: {self.train_description(train_info)}')
            else:
                self.log.debug(f'nearby_trains: rejected train: {self.train_description(train_info)}, distance: {int(dst)} m, age: {timedelta.seconds} s')
                rejected_trains.append(train_info)

        return nearby_trains


class image_capture:

    def __init__(self, root_folder = '', input_root_folder = 'inbox', output_root_folder = 'results', cam_num = 0, configuration_file = 'train_detection_configuration.csv', fb_credential_file = '', db_file = ''):

        # event logging
        self.log = logging.getLogger(__name__)

        self.log.info(f'image_capture (init): start configuration of camera {cam_num}.')

        # base parameters
        self.configuration_file = configuration_file
        self.fb_credential_file = fb_credential_file
        self.fb_video_inhibit = datetime.now()
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

        self.log.debug(f'image_capture (init): basic configuration of camera {cam_num} completed.')

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
        self.train_location_timeout = 190
        self.train_location_distance = 850

        self.nr_disable = False  # keep in case model cannot be loaded
        self.loaded_nr_version = -1   # keep to identify if new version specified

        # read configuration
        self.read_configuration()

        # load classification model  ==>   put into function, so that model could be loaded dynamically at runtime whenever a new configuration is found
        try:
            self.load_model()
            self.log.debug(f'image_capture (init): Completed loading detection model for camera {cam_num}.')
        except:
            self.log.error(f'image_capture (init): Could not load detection model for camera {cam_num}: ' + str(ex))

        # read facebook credentials
        self.page_graph = False
        try:
            page_credentials = self.read_fb_creds(os.path.join(root_folder, self.fb_credential_file))
            self.fb_access_token = page_credentials['access_token']
            self.page_graph = GraphAPI(access_token = self.fb_access_token)
        except Exception as ex: # work on python 3.x
            self.log.error('image_capture (init): Could not load fb credentials: ' + str(ex))

        self.fb = fb_helper(self.page_graph)
        self.log.info(f'image_capture (init): fb access for camera {cam_num} completed.')

        # create root folder
        self.set_root_folder(root_folder)

        # if cam num is already specified, create folder structure
        if os.path.isdir(self.root_folder):
            self.set_input_folder()
            if cam_num > 0:
                self.set_output_folder()

        self.log.info(f'image_capture (init): copleted configuration of camera {self.cam_num}')

    def load_model(self):
        nr_version = self.conf_value('nr_version')
        model_file_name = ''

        # Load model when configured version larger than previously loaded version
        if self.conf_value('nr_enable') and nr_version != self.loaded_nr_version:
            self.log.info(f'load_model: Loading new model for camera {self.cam_num}; new version: {nr_version}, previous: {self.loaded_nr_version}')
            try:
                model_version = int(nr_version)
                model_file_name = os.path.join(self.root_folder, 'models', 'neural networks', f'cam{self.cam_num}_model_v{model_version}.h5')
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

    # function to set video posting inhibit to now + 24 hrs
    def set_video_post_inhibit(self):
        self.fb_video_inhibit = datetime.now() + timedelta(hours = 24)
        self.log.info(f'set_video_post_inhibit: video positing for camera {self.cam_num} inhibited until: {self.fb_video_inhibit}.')
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
            configuration_pd = pd.read_csv(os.path.join(self.root_folder, self.configuration_file)) #, encoding = "ISO-8859-1")
            self.log.debug(f'read_configuration: Configuration for camera {self.cam_num} successfully read from configuration file: {self.configuration_file}')
        except Exception as ex:
            self.log.error(f'read_configuration: Could not read {self.configuration_file}: {str(ex)}')
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
            self.log.error(f'acquire_camera_data: response not ok, response status code = {response.status_code}')

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

        # initialize result
        self.record["shows_train"] = False

        try:
            # validate image file name
            if len(self.image_file_names) > 0:

                # go through all image files
                for img_file in self.image_file_names:

                    # skip if a train was already detected on a previous image
                    if not self.record["shows_train"]:

                        img_path = os.path.join(self.root_folder, self.input_folder, img_file)

                        # if a mask is used for this camera, apply and store temporary copy
                        if self.conf_value('nr_mask'):

                            # determine image width
                            image_height, image_width, image_channels = cv2.imread(img_path).shape

                            # read mask for selected camera
                            full_mask_file_name = os.path.join(self.root_folder, 'models', 'masks', f'{self.cam_num}_mask.png')
                            mask = imutils.resize(cv2.imread(full_mask_file_name, 0), width = image_width, height = image_height)

                            # apply mask and save in temporary destination
                            if os.path.isfile(img_path):
                                img = cv2.imread(img_path)

                                try:
                                    # create massked image
                                    masked_image = cv2.bitwise_and(img, img, mask = mask)

                                    # save as temporary file, overwrite img_path to point to new file
                                    img_path = os.path.join(self.root_folder, self.input_folder, img_file + '_masked.jpg')
                                    cv2.imwrite(img_path, masked_image)
                                except Exception as ex:
                                    self.log.error(f'classify_nr: could not apply mask on {img_path} with mask width {image_width} and width {image_height}: ' + str(ex))

                                img = None
                                masked_image = None
                                self.log.debug(f'classify_nr: Applied mask to {img_file} for classification.')
                            pass
                        try:
                            # open image and pre-process for use with model
                            img = tf.keras.utils.load_img(img_path, target_size = (256, 256))
                            img = tf.keras.utils.img_to_array(img)
                            img = np.expand_dims(img, axis = 0)
                        except Exception as ex:
                            self.log.error(f'classify_nr: could not open {img_file}: ' + str(ex))
                            pass

                        try:
                            # use model to predict if this image contains a train
                            self.record["nr_confidence"] = self.nr_model.predict(img)[0][0]

                            # free up memory of img
                            img = None

                            # If confidence is more than 0.8, probably found a train:
                            if (self.record["nr_confidence"] > 0.8):
                                # set result to "train":
                                self.record["shows_train"] = True
                                self.record["cnt_train"] += 1
                                self.record["detection_count"] += 1 # do net set here, but later when we know it's a *valid* detection
                            else:
                                # set result to "no train":
                                self.record["shows_train"] = False
                                self.record["cnt_no_train"] += 1
                                self.record["detection_count"] = 0

                            # store that this image was processed (classified)
                            self.record["classified"] = True
                            self.log.info(f'classify_nr: {img_file} is showing a train: {self.record["nr_confidence"]} => {self.record["shows_train"]}, detection count: {self.record["detection_count"]}, threshold: {self.record["detection_threshold"]}; cnt: {self.record["cnt_pic"]}, train: {self.record["cnt_train"]}, no_train: {self.record["cnt_no_train"]}')

                        except Exception as ex:
                            self.log.error(f'classify_nr: something went wrong processing {img_path}: ' + str(ex))
                            pass

                        # if mask was applied, clean up temporary file
                        if self.conf_value('nr_mask'):
                            if os.path.isfile(img_path):
                                os.remove(img_path)
                            pass

                    else:
                        self.record["classified"] = True
                        self.log.warning(f'classify: skipping {img_file}, train already detected.')

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
            full_mask_file_name = os.path.join(self.root_folder, 'models', 'masks', f'{self.cam_num}_mask.png')
            try:
                mask = imutils.resize(cv2.imread(full_mask_file_name, 0), width = image_width)
            except Exception as ex:
                self.log.error(f'detect_motion: failed to read mask {full_mask_file_name}: {str(ex)}')
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

        # calculate each object's absolute movement
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
            self.log.info(f'detect_motion: Objects on camera {self.cam_num} were mostly moving RIGHT - motion value: {X}, with threshold: {self.conf_value("motion_threshold")}.')

        elif X < -self.conf_value('motion_threshold'):
            self.record["motion_detected"] = True
            self.record["direction"] = self.conf_value('left_orientation')
            self.log.info(f'detect_motion: Objects on camera {self.cam_num} were mostly moving LEFT - motion value: {X}, with threshold: {self.conf_value("motion_threshold")}.')

        else:
            self.record["direction"] = 'no motion detected'
            self.log.info(f'detect_motion: Objects on camera {self.cam_num} were NOT MOVING - motion value: {X}, with threshold: {self.conf_value("motion_threshold")}.')

        return True

    # function to generate image caption
    def generate_caption(self, train_info):

        # Construct message for posting
        try:
            cam_text = self.conf_value('cam_description')
        except:
            cam_text = str(self.cam_num)

        motion_text= ''
        if self.conf_value('motion_enable'):
            motion_text = ' (no motion detected)'

        if self.record["motion_detected"]:
            if self.record["motion_value"] > 0:
                motion_text = ' heading ' + self.conf_value('right_orientation')
            else:
                motion_text = ' heading ' + self.conf_value('left_orientation')

        caption = 'Train' + motion_text + ' detected at ' + self.record["time"] + ' on camera ' + cam_text + ' [' + str(self.cam_num) + '] on ' + self.record["date"] + '.'

        # try to find matching trains
        try:
            if len(self.nearby_trains) > 0:

                train_descriptions = []
                for train in self.nearby_trains:
                    train_descriptions.append(train_info.train_description(train))

                if len(train_descriptions) > 0:
                    caption += ' Train on picture: ' + ' or '.join(train_descriptions)

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
                    self.log.error(f'archive_files: could not delete {img_file} with type {self.record["file_type"]}: ' + str(ex))

            elif self.record["file_type"] == 'mp4':
                # move extracted jpg files
                try:
                    # move image files
                    for img_file in self.image_file_names:
                        shutil.move(os.path.join(self.root_folder, self.input_folder, img_file), self.output_folder_train)
                except Exception as ex:
                    self.log.error(f'archive_files: could not move {img_file} with type {self.record["file_type"]} to {self.output_folder_train}: ' + str(ex))

            try:
                # move main image/video file
                if os.path.isfile(source_file):
                    shutil.move(source_file, self.output_folder_train)
                self.record["file_path"] = self.output_folder_train
            except Exception as ex:
                self.log.error(f'archive_files: could not move {self.file_name} with type {self.record["file_type"]} to {self.output_folder_train}: ' + str(ex))

        elif delete_no_train:
            # delete extracted image files
            try:
                for img_file in self.image_file_names:
                    os.remove(os.path.join(self.root_folder, self.input_folder, img_file))
            except Exception as ex:
                self.log.error(f'archive_files: could not delete {img_file} with type {self.record["file_type"]}: ' + str(ex))

            # delete main image/video file
            try:
                os.remove(source_file)
            except Exception as ex:
                self.log.error(f'archive_files: could not delete {self.file_name} with type {self.record["file_type"]}: ' + str(ex))

        else:
            # treat as "no_train"
            if delete_jpg:
                try:
                    for img_file in self.image_file_names:
                        os.remove(os.path.join(self.root_folder, self.input_folder, img_file))
                except Exception as ex:
                    self.log.error(f'archive_files: could not delete {img_file} with type {self.record["file_type"]}: ' + str(ex))
            else:
                try:
                    # move image files (no_train)
                    for img_file in self.image_file_names:
                        shutil.move(os.path.join(self.root_folder, self.input_folder, img_file), self.output_folder_notrain)
                except Exception as ex:
                    self.log.error(f'archive_files: could not move {img_file} with type {self.record["file_type"]} to {self.output_folder_notrain}: ' + str(ex))

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
    def process(self, train_info = False):

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

        #2b - Extract still frame and determine if it shows a train
        if not self.nr_disable and self.conf_value('nr_enable'):
            self.classify_nr()

        #3 - Determine motion direction
        if self.conf_value('motion_enable'):
            self.detect_motion()

            #3b - check if motion is high enough to override nr classification
            try:
                if (self.conf_value("motion_override") > self.conf_value("motion_threshold")) and (self.record["motion_value"] > self.record["motion_override"]):
                    self.record["shows_train"] = True
                    self.log.info(f'process: detected train on camera {self.can_num} by motion {self.record["motion_value"]}, override threshold: {self.conf_value("motion_override")}')
            except Exception as ex:
                self.log.debug(f'process: camera {self.cam_num} failed to process motion override threshold: {str(ex)}')

        #4 - Move files to archive destination, could be extended to become
        #    configurable per camera (from config file)
        delete_no_train = self.conf_value('nr_enable') and not self.conf_value('nr_training') # or something similar...
        delete_jpg = False

        if self.conf_value('nr_enable') or self.conf_value('motion_enable'):
            self.archive_files(delete_no_train, delete_jpg, ignore_threshold = True)

        # Evaluate if conditions to accept train are given
        if not train_info:
            self.nearby_trains = []
            self.log.debug(f'process: no information about nearby trains available, not setting "nearby_trains"')
        else:
            lon = self.conf_value('longitude')
            lat = self.conf_value('latitude')
            self.nearby_trains = train_info.nearby_trains(location=(lat, lon), max_dist = self.train_location_distance, max_age = self.train_location_timeout)

        # Validate if detection is considered valid
        detection_valid = not self.conf_value("p_train") and not self.conf_value("p_motion") or (self.conf_value("p_train") and (len(self.nearby_trains) > 0)) or (self.conf_value("p_motion") and self.record["motion_detected"])
        self.log.debug(f'process: detection for camera {self.cam_num} is valid: {detection_valid}, based on: p_train: {self.conf_value("p_train")} and {len(self.nearby_trains)} nearby trains, p_motion: {self.conf_value("p_motion")} and motion detected: {self.record["motion_detected"]}.')

        if not detection_valid and self.record["detection_count"] > 0:
            # reset detection count if detection is considered not valid
            self.record["detection_count"] = 0
            self.log.debug(f'process: resetting detection count, as detection not valid.')

        # add info log about valid detection HERE

        # Store record and post on facebook
        if detection_valid and self.record["shows_train"]:
            # generate caption only if train detected
            self.generate_caption(train_info)

            # Evaluate if conditions for posting are given

            #5 - Post on facebook (need to do *before* storing record, to capture fb id)
            if self.conf_value('fb_posting'):
                ignore_threshold = False
                self.fb.publish_train(self, ignore_threshold)
            else:
                self.log.debug(f'process: not posting {self.file_name}, not configured for fb posting.')

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
            try:
                if train.file_name[-3:] == 'mp4':
                    post_video = train.fb_video_inhibit < datetime.now()
                    self.log.debug(f'publish_train: publishing video for camera {train.cam_num} is allowed: {post_video}; inhibited until: {train.fb_video_inhibit}.')
                else:
                    post_video = False
            except:
                post_video = False

            # temporary measure...
            #post_video = not train.fb_video_inhibit or (train.fb_video_inhibit < datetime.now())
            #post_video = False

            # disable video posting during night
            if post_video and ((datetime.now().hour > 19) or (datetime.now().hour < 6)):
                self.log.debug(f'publish_train: video posting disabled after 19h and before 6h.')
                post_video = False

            # determine which file to use for posting
            video_file_name = train.file_name
            image_file_name = train.image_file_names[0]

            # determine path to image file to be posted
            source_folder = ''
            if train.record["archived"]:
                source_folder = train.output_folder_train
            elif train.record["classified"]:
                source_folder = train.input_folder
            else:
                self.log.warning(f'publish_train: image {image_file_name} is neither classified nor archived, so cannot post it.')
                return False

            image_location = os.path.join(train.root_folder, source_folder, image_file_name)
            video_location = os.path.join(train.root_folder, source_folder, video_file_name)

            # Construct message for posting
            message_content = train.record["caption"]

            # initialize return
            ret = False
            photo_id = 0
            photo_url = "(not posted)"

            # Publish image or video
            try:
                # determine to post photo or video
                if post_video:
                    # post video
                    if not self.publish_video(train, video_location, message_content):
                        # inhibit positing video for 12h, then try again
                        self.log.error(f'publish_train: video posting for camera {train.cam_num} failed, inhibiting and trying to post picture instead.')
                        train.set_video_post_inhibit()

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

    #root_OneDrive_folder = 'C:\\Users\\georg\\OneDrive\\Documents\\Eisenbahn\\TrainDetector'
    root_folder = 'C:\\Users\\georg\\Documents\\Rail\\TrainDetector'

    # Setup Logging
    try:

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

        # Create a timed rotating file handler and set its level to capture debug-level messages
        # The log file name will be 'my_app.log' with a date suffix such as 'my_app.log.20240302'
        # The log file will rotate at midnight and keep 14 backups

        #file_handler = logging.FileHandler(f'td_{today_str}_debug.log')
        file_handler = logging.handlers.TimedRotatingFileHandler(os.path.join(root_folder, 'td_debug.log'), when = 'midnight', backupCount = 14)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        file_handler.suffix = "%Y%m%d"

        # Get the root logger and add the handlers
        log = logging.getLogger()
        log.addHandler(console_handler)
        log.addHandler(file_handler)
    except Exception as ex:
        print(f'Could not setup logging: {str(ex)}')
        return False

    # Read Configuration
    try:
        # base configuration
        log.info('Starting TrainDetector.')

        configuration_file = 'train_detection_configuration.csv'
        fb_credential_file = 'page_credentials.json'
        database_file = 'train_sightings.db'

        # Read configuration and determine cameras
        log.info(f'Reading configuration file {os.path.join(root_folder, configuration_file)}.')
        try:
            configuration = pd.read_csv(os.path.join(root_folder, configuration_file)) #, encoding = "ISO-8859-1")
        except Exception as ex:
            log.error(f'Error reading configuration {configuration_file}: ' + str(ex))

        log.info(f'Loaded data for {len(configuration)} cameras:\n {configuration}')
    except Exception as ex:
        log.error(str(ex))
        return False

    # construct camera list
    cam_list = []
    for cam_id in configuration['cam_id']:
        try:
            log.info(f'Initializing camera {cam_id}...')
            cam_list.append(image_capture(root_folder = root_folder, input_root_folder = 'inbox', output_root_folder = 'results', cam_num = cam_id, configuration_file = configuration_file, fb_credential_file = fb_credential_file, db_file = database_file))
            log.info(f'Initializing camera {cam_id} completed.')
        except Exception as ex:
            log.error(f'Error setting up image captures of camera {cam_id}.')

    # initialize train location
    train_info = train_location(agencies = ['via', 'exo'], static_gtfs_folder = os.path.join(root_folder, 'models', 'gtfs'))

    # capture data
    log.info('Start capturing process.')

    try:
        while(True):
            # get current train information before processing cameras
            train_info.read_trains()

            for cam in cam_list:
                try:
                    cam.process(train_info)
                except Exception as ex:
                    log.error(f'Error processing camera: {cam.cam_num}' + str(ex))

            time.sleep(18)

    except KeyboardInterrupt:
        pass

    log.info('Terminating capturing process.')

if __name__ == '__main__':
    main()

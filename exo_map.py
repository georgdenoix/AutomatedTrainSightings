#-------------------------------------------------------------------------------
# Name:        exo map
# Purpose:
#
# Author:      georg
#
# Created:     05-06-2023
# Copyright:   (c) georg 2023
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import json
import os
from google.transit import gtfs_realtime_pb2
import pandas as pd
import requests

app = FastAPI()

# Mount the static files directory
root_directory = "C:\\Users\\georg\\OneDrive\\Documents\\Eisenbahn\\TrainDetector"
static_folders = ['static', 'static/images']

for folder in static_folders:
    app.mount(f"/{folder}", StaticFiles(directory = os.path.join(root_directory, folder)), name = folder)

templates = Jinja2Templates(directory = "templates")


def get_trains(feed_url = '', agency = ''):

    trips = pd.read_csv(os.path.join('gtfs', agency, 'trips.txt'))
    routes = pd.read_csv(os.path.join('gtfs', agency, 'routes.txt'))

    response = requests.get(feed_url)
    if response.status_code == 200:
        vehicles = gtfs_realtime_pb2.FeedMessage()
        vehicles.ParseFromString(response.content)
    else:
        print(response.status_code)

    train_list = []

    #print(agency)

    #collect train locations
    for train in vehicles.entity:

        if train.HasField('vehicle'):
            vehicle = train.vehicle

            # exclude buses:
            if (agency == 'go') and vehicle.trip.route_id[-2:].isdigit():
                continue

            duplicate = False
            for stored_train in train_list:
                if stored_train['trip_id'] == vehicle.trip.trip_id:
                    duplicate = True

            # ignore if already in list
            if duplicate:
                continue

            train_info = {}

            train_info['agency'] = agency
            train_info['label'] = agency[:3]
            train_info['destination'] = ''
            train_info['id'] = ''
            train_info['trip_id'] = ''

            # resolve train no. from trips table
            try:


                ## TRAIN NUMBER ("id")
                #trip_id = int(vehicle.trip.trip_id) if isinstance(vehicle.trip.trip_id, int) else vehicle.trip.trip_id

                #condition = (trips['trip_id'] == trip_id)
                #train_info['trip_id'] = vehicle.trip.trip_id

                #trip_short_name = trips.loc[condition, 'trip_short_name'].values
                #if len(trip_short_name) > 0:
                #    train_info['id'] = str(trip_short_name[0])
                #else:
                #    train_info['id'] = ''

                try:
                    trip_id = vehicle.trip.trip_id if isinstance(vehicle.trip.trip_id, int) else int(vehicle.trip.trip_id)   # fixed
                except:
                    trip_id = vehicle.trip.trip_id

                train_info['trip_id'] = vehicle.trip.trip_id


                # VIA stores train number in last characters of vehicle id
                if (agency == 'via'):
                    if len(vehicle.vehicle.id) > 15:
                        train_info['id'] = vehicle.vehicle.id[15:]

                # GO stores train number in last digits of train id
                if (agency == 'go'):
                    if len(train.id) > 12:
                        train_info['id'] = train.id[12:]

                # get the train number from GTFS "trips" table as "trip_short_name" - EXO and VIA only, empty for GO
                if isinstance(trips, bool):
                    trip_short_name = ''
                    print("trips[agency] is type bool")
                else:
                    try:
                        condition = (trips['trip_id'] == trip_id)
                        trip_short_name = trips.loc[condition, 'trip_short_name'].values
                    except:
                        print("Could not resolve", trip_id, "in trips.txt.")

                # use look-up from trip list (trips.txt) in all other cases
                if (len(trip_short_name) > 0) and (train_info['id'] == ''):
                    train_info['id'] = str(trip_short_name[0])

            except Exception as ex:
                print(f"Error while getting train number for {agency} {vehicle.vehicle.id}: {str(ex)}")
                #train_info['id'] = ''
                pass

            try:
                ## DESTINATION
                #trip_headsign = trips.loc[condition, 'trip_headsign'].values
                #if len(trip_headsign) > 0:
                #    train_info['destination'] = str(trip_headsign[0])
                #else:
                #    train_info['destination'] = ''


                if isinstance(trips, bool):
                    trip_headsign = ''
                else:
                    trip_headsign = trips.loc[condition, 'trip_headsign'].values

                if len(trip_headsign) > 0:
                    train_info['destination'] = str(trip_headsign[0])
                else:
                    print(f'read_trains: could not find matching "trip_headsign" (destination) in {agency} trips for trip_id {trip_id} and vehicle {vehicle.vehicle.id}.')
                    #train_info['destination'] = ''

                # Specific to GO
                if (agency == 'go'):
                    # Destination is stored in Vehicle Label, get it from there if trips.txt could not resolve it:
                    if len(train_info['destination']) < 5:
                        train_info['destination'] = vehicle.vehicle.label

                    # strip first characters as they repeat the route:
                    if len(train_info['destination']) > 5:
                        train_info['destination'] = train_info['destination'][5:]

            except Exception as ex:
                print(f"Error while getting destination for {agency} {vehicle.vehicle.id}: {str(ex)}")
                #train_info['destination'] = ''
                pass

            ## resolve ROUTE NAME from routes table
            #try:
            #    # because trip_id is numeric, pandas stores it as integer
            #    route_id = int(vehicle.trip.route_id) if isinstance(vehicle.trip.route_id, int) else vehicle.trip.route_id

            #    condition = (routes['route_id'] == route_id)

            #    route_long_name = routes.loc[condition, 'route_long_name'].values
            #    if len(route_long_name) > 0:
            #        train_info['route_name'] = str(route_long_name[0])
            #    else:
            #        train_info['route_name'] = ''

            #except:
            #    pass

            try:
                # extract route id
                try:
                    route_id = vehicle.trip.route_id if isinstance(vehicle.trip.route_id, int) else int(vehicle.trip.route_id)  # fixed
                except:
                    route_id = vehicle.trip.route_id

                #get the route name from GTFS "routes" table as "route_long_name"
                if isinstance(routes, bool):
                    route_long_name = ''
                else:
                    condition = (routes['route_id'] == route_id)
                    route_long_name = routes.loc[condition, 'route_long_name'].values

                if len(route_long_name) > 0:
                    train_info['route_name'] = str(route_long_name[0])
                else:
                    print(f'read_trains: could not find matching "route_long_name" (route name) in {agency} routes for route_id "{route_id}" and vehicle "{vehicle.vehicle.id}".')
                    #train_info['route_name'] = ''

            except Exception as ex:
                print(f"Error while getting route name for {agency} {vehicle.vehicle.id}: {str(ex)}")
                #train_info['route_name'] = ''
                pass






            train_info['latitude'] = vehicle.position.latitude
            train_info['longitude'] = vehicle.position.longitude
            train_info['speed'] = str(round(vehicle.position.speed, 1))

            print("Processed train: ", train_info)

            train_list.append(train_info)

    return train_list


def get_all_trains():

    exo_trains = get_trains(feed_url = 'https://opendata.exo.quebec/ServiceGTFSR/VehiclePosition.pb?token=9VZEYEXS3C&agency=TRAINS', agency = 'exo')
    via_trains = get_trains(feed_url = 'https://asm-backend.transitdocs.com/gtfs/via', agency = 'via')
    go_trains = get_trains(feed_url = 'https://api.openmetrolinx.com/OpenDataAPI/api/V1/Gtfs.proto/Feed/VehiclePosition?key=30024739', agency = 'go')

    # combine exo and via trains
    train_list = exo_trains + via_trains + go_trains

    # Convert the dictionary to JSON
    json_data = json.dumps(train_list)

    return json_data


@app.get("/api/train-positions")
def get_train_positions():
    # Assuming train_positions is your dictionary
    json_data = get_all_trains()
    #print(json_data)
    return JSONResponse(content=json_data)

@app.get("/map")
async def show_map(request: Request):
    #get_exo_trains()
    return templates.TemplateResponse("map.html", {"request": request})

def main():
    pass

if __name__ == '__main__':
    main()

#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      georg
#
# Created:     24-11-2024
# Copyright:   (c) georg 2024
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sqlite3, os, logging
from typing import List, Optional
from datetime import time


# Initialize FastAPI
app = FastAPI()

# Root folder where data is stored
root_folder = 'C:\\Users\\georg\\Documents\\Rail\\TrainDetector\\results'

# Path to the folder containing your media files
media_folder = os.path.join(root_folder, 'media')  # This should match your media folder

database_file = 'train_sightings.db'

# Set up templates
templates = Jinja2Templates(directory="templates")

# Serve the static files (e.g., CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount the media folder to be accessible via a URL
app.mount("/images", StaticFiles(directory=media_folder), name="images")

# Enable logging for debugging
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level=logging.DEBUG
)

logging.info("td_browser started.")

@app.on_event("startup")
def startup_event():
    logging.info("Application started. Registered routes:")
    for route in app.routes:
        logging.info(route)

# SQLite database connection
def get_db_connection():
    conn = sqlite3.connect(os.path.join(root_folder, database_file))
    conn.row_factory = sqlite3.Row  # To return rows as dictionaries
    return conn

# Root endpoint to serve the HTML page

@app.get("/")
async def root(request: Request):
    logging.info("Root endpoint called.")
    logging.debug(f"Request details: {request}")
    logging.info(f"Headers received: {request.headers}")
    return templates.TemplateResponse("td_browser.html", {"request": request})


#@app.get("/")
#def root():
#    logging.info(f"Main page loaded.")
#    return templates.TemplateResponse("td_browser.html")

# API endpoint to get items from the database
@app.get("/api/items")
def get_items(
    offset: int = 0,
    limit: int = 200,
    camera: List[int] = Query(None),
    start_date: str = Query(None),
    end_date: str = Query(None),
    weekday: List[int] = Query(None),  # multiple values possible
    start_time: str = Query(None),
    end_time: str = Query(None),
    caption: str = Query(None),
    direction: List[str] = Query(None),  # multiple values possible
):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            SELECT camera, date, caption, fb_url, file_path, file_name, weekday, time, direction
            FROM trains
            WHERE fb_url IS NOT NULL AND fb_url != ''
        """
        params = []

        if camera:
            placeholders = ",".join("?" for _ in camera)
            query += f" AND camera IN ({placeholders})"
            params.extend(camera)
        if caption:
            query += " AND caption LIKE ?"
            params.append(f"%{caption}%")
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        if weekday:
            placeholders = ",".join("?" for _ in weekday)
            query += f" AND weekday IN ({placeholders})"
            params.extend(weekday)
        if start_time:
            query += " AND time >= ?"
            params.append(start_time)
        if end_time:
            query += " AND time <= ?"
            params.append(end_time)
        if direction:
            placeholders = ",".join("?" for _ in direction)
            query += f" AND direction IN ({placeholders})"
            params.extend(direction)

        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        logging.debug(f"Executing query: {query} with params: {params}")
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        conn.close()

        result = []
        for row in rows:
            row_dict = dict(row)
            if row_dict['file_path']:
                relative_path = row_dict['file_path'].replace(str(root_folder) + '\\media\\', '')
                row_dict['file_path'] = relative_path
            result.append(row_dict)

        return result
    except sqlite3.Error as e:
        logging.error(f"SQLite Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Database query error")
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server Error: {e}")
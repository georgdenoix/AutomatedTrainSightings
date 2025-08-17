#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      georg
#
# Created:     19-05-2025
# Copyright:   (c) georg 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date

# Load your data
@st.cache_data
def load_data():
    conn = sqlite3.connect("C:/Users/georg/Documents/Rail/TrainDetector/results/train_sightings.db")
    df = pd.read_sql_query("SELECT * FROM trains WHERE fb_url IS NOT NULL AND fb_url != ''", conn)
    conn.close()
    return df

df = load_data()

st.title("Train Sightings Dashboard")

# Filter widgets
camera_ids = st.multiselect("Camera IDs", sorted(df['camera'].dropna().unique()))
directions = st.multiselect("Direction", sorted(df['direction'].dropna().unique()))
weekday_filter = st.multiselect("Weekday", sorted(df['weekday'].dropna().unique()))
time_range = st.slider("Time of day", 0, 23, (0, 23))
# Default range: current year
start_date = st.date_input("Start Date", value=date(2025, 1, 1), format="YYYY-MM-DD")
end_date = st.date_input("End Date", value=date(2025, 12, 31), format="YYYY-MM-DD")

# Apply filters
if camera_ids:
    df = df[df['camera'].isin(camera_ids)]
if directions:
    df = df[df['direction'].isin(directions)]
if weekday_filter:
    df = df[df['weekday'].isin(weekday_filter)]
df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce').dt.hour
df = df[df['hour'].between(*time_range)]
# Ensure 'date' is a datetime column
df['date'] = pd.to_datetime(df['date'])
# Filter by start and end
df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

# Create direction groups
df['direction_group'] = df['direction'].map(lambda d: 'Eastbound' if d in ['East', 'South']
                                             else 'Westbound' if d in ['West', 'North']
                                             else 'Other')


# Density plot
st.subheader("Sightings by Hour")
fig, ax = plt.subplots()

if len(df) >= 2:
    sns.histplot(
        data=df,
        x='hour',
        hue='direction_group',
        bins=24,
        kde=True,
        multiple='layer',
        common_norm=False,
        palette={'Eastbound': 'red', 'Westbound': 'blue', "Other": 'grey'},
        ax=ax
    )

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Sightings")
    ax.set_xticks(range(0, 24))
    ax.legend(title="Direction: East/South: red West/North: blue")
    st.pyplot(fig)


# Ensure weekday is numeric
df['weekday'] = df['weekday'].astype(int)

# Force categorical type with fixed ordering
weekday_order = list(range(7))
df['weekday'] = pd.Categorical(df['weekday'], categories=weekday_order, ordered=True)

# Plot
st.subheader("Sightings by Weekday")
fig, ax = plt.subplots()

sns.histplot(
    df['weekday'],
    bins=len(weekday_order),
    discrete=True,
    kde=False,
    ax=ax
)

ax.set_xticks(weekday_order)
ax.set_xlabel("Weekday (0 = Sunday, 6 = Saturday)")
ax.set_ylabel("Number of Sightings")

st.pyplot(fig)
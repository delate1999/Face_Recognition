import cv2
import sqlite3
import numpy as np


def get_profile(id):
    conn = sqlite3.connect('FaceBase.db')
    cmd = 'SELECT * FROM People WHERE ID=' + str(id)
    cursor = conn.execute(cmd)
    profile = 0
    for row in cursor:
        profile = row
    conn.close()
    return profile

def insert(ID,Name,Age,Gender):
    conn = sqlite3.connect('FaceBase.db')
    cmd = 'SELECT * FROM People WHERE ID=' + str(ID)
    cursor = conn.execute(cmd)
    # Nie mam pojecia po co jest ten fragment ale bez tego sie wysypuje
    is_existing = 0
    for row in cursor:
        is_existing = 1
    if is_existing == 0:
        # az do tad
        cmd = 'INSERT INTO People(ID,Name,Age,Gender) Values('+str(ID)+','+str(Name)+ ',' + str(Age) +',' + str(Gender) + ')'
    conn.execute(cmd)
    conn.commit()
    conn.close()
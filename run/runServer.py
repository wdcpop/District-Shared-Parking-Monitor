
# I/O + system imports
import time
import datetime as dt 
import os
import json
import array
import socket

# DB imports
import mysql.connector

# Image processing imports
# import math
# import numpy as np 
import cv2
from markerdetectsrv import *

host = ''
port = 5560
sock = 0
conn = 0

class parking_entry():
	def __init__(self, space = None, location = None, 
		pixelarea = None, available = None):
		self.space 		= space
		self.location 	= eval(str(location))
		self.pixelarea 	= pixelarea
		self.available 	= available

	def __repr__(self):
		return "[" + str(self.space) + ", " + str(self.location) + ", " + \
			str(self.pixelarea) + ", " + str(self.available) + "]"

def print_welcome():
	print ""
	print " ----------------------"
	print "| EE475 Parking System |"
	print " ----------------------"
	print ""

# Returns whether or not the user is using an existing parking
# database
# Returns: 1 if yes, 0 if no
def get_db_status():
	print "Are you using an existing database? (y/n)"
	while True:
		user_input = raw_input('>')
		if (user_input == 'y' or user_input == 'Y'):
			return 1
		elif (user_input == 'n' or user_input == 'N'):
			return 0
		else: 
			print "Please enter (y/n)"

# Initialize DB connection
def init_db():
	cnx = mysql.connector.connect(user='root', password='raspberry',
		host='localhost')
	return [cnx, cnx.cursor()]

# Commit any changes to the DB connection and close cursor and connection
def close_db(cnx, cursor):
	cnx.commit()
	cursor.close()
	cnx.close()

# Create a new database
# NOTE: FOR DEMO PURPOSES, WE'LL JUST CREATE AN "EE475" DB+TABLES
# IF THEY DON'T ALREADY EXIST
def create_db():
	[cnx, cursor] = init_db()
	query = ("CREATE DATABASE IF NOT EXISTS EE475;")
	cursor.execute(query)

	query = ("CREATE TABLE IF NOT EXISTS `EE475`.`parkingdata` "
		"( `space` INT NOT NULL , `location` VARCHAR(255) NOT NULL, "
		"`pixelarea` FLOAT NOT NULL, `available` BOOLEAN NOT NULL DEFAULT '1', "
		"PRIMARY KEY (`space`)) ENGINE = InnoDB;")
	cursor.execute(query)

	query = ("CREATE TABLE IF NOT EXISTS `EE475`.`parkingphotos` "
		"( `time` TIMESTAMP NOT NULL , `photo` VARCHAR(255) NOT NULL , "
		"PRIMARY KEY (`time`)) ENGINE = InnoDB;")
	cursor.execute(query)

	close_db(cnx, cursor)

# Generate the INSERT vectors for a given marker array
def generate_marker_insert(markers):
	insert = ""
	for m in markers:
		insert = insert + "(" + str(m.number) + ", \"" + \
			str(m.location) + "\", " + str(m.area) + ", " + str(1) + "), "
	return insert[:-2]

# Generate the INSERT vectors for a give parking data array
def generate_parking_insert(data):
	insert = ""
	for d in data:
		insert = insert + "(" + str(d.space) + ", \"" + str(d.location) + \
			"\", " + str(d.pixelarea) + ", " + str(d.available) + "), "
	return insert[:-2]

# INSERT the given markers into the database based on their assigned numner
# Select the datatype to use with optional parameter
def update_db(data, datatype = "dbdata"):
	[cnx, cursor] = init_db()

	query = ("USE EE475")
	cursor.execute(query)

	if datatype == "dbdata":
		insert_vector = generate_parking_insert(data)
	elif datatype == "markerdata":
		insert_vector = generate_marker_insert(data)

	qstr = "INSERT INTO parkingdata (space, location, pixelarea, available) " + \
		"VALUES " + insert_vector + " " + \
		"ON DUPLICATE KEY UPDATE " + \
		"location = VALUES(location), " + \
		"pixelarea = VALUES(pixelarea), " + \
		"available = VALUES(available)"

	query = (qstr)
	cursor.execute(query)

	close_db(cnx, cursor)

# Query the DB for parking data
# RETURNS: Array of parking_entry types
def get_parking_data():
	parking_data = []

	[cnx, cursor] = init_db()
	
	query = ("USE EE475")
	cursor.execute(query)

	qstr = "SELECT * FROM parkingdata"
	query = (qstr)
	cursor.execute(query)

	for (space, location, pixelarea, available) in cursor:
		p = parking_entry(space, location, pixelarea, available)
		parking_data.append(p)

	return parking_data

# Check if a marker exists at the approximately the given location
# with approximately the same pixel area. If it does, we know that
# space is available (and vice versa)
# RETURNS: 	True/False
# FUTURE: 	make this dynamic based on initial marker distribution to avoid
# 			incorrect space availability assignment!!
def check_marker(location, pixelarea, markers):
	f = False
	for m in markers:
		# If location x & y are within 10%, 
		# and if pixelarea is within 10%
		if 	abs(1.0 - float(m.location[0]) / float(location[0])) <= 0.10 and \
			abs(1.0 - float(m.location[1]) / float(location[1])) <= 0.10 and \
			abs(1.0 - m.area / pixelarea) <= 0.10:
				f = True
	return f


# Run image processing code on an initial photograph to determine
# marker location and occlusion information
def assign_spaces(f):
	# Find the markers in the original image
	[markers, image] = process_image(f)

	# Assign numbers to those markers
	for m in markers:
		print "Assign an identifier to marker " + str(m.number) + " in the image"
		ident = raw_input("> ")
		m.number = ident

	# Confirm all markers and locations
	for m in markers:
		print "Assigned marker " + str(m.number) + " at location " + str(m.location)

	update_db(markers, "markerdata")

# Update the availability of each space based on the OpenCV results
# NOTE: - db_data is formatted as database rows:
#				(space, location, pixelarea, available)
#		- markers is formatted as marker objects:
#				(location, tri, trap, component_angle, skew_angle,
#					skews, number, area)
#
# Markers must be compared to their respective DB number by comparing
# location and pixel area
def update_availability(db_data, markers):
	for d in db_data:
		if check_marker(d.location, d.pixelarea, markers):
			d.available = 1
		else:
			d.available = 0
	update_db(db_data, "dbdata")

# Accept a TCP socket to communicate with the Pi
def init_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port)) # bind local host
        print("Socket created and bound to port " + str(port))
    except socket.error as msg:
        print(msg)
    print 'Server is listening on ' + str(port) + '...'
    sock.listen(1) # Allows one connection at a time. blocking
    conn, address = sock.accept()
    print("Client connected: " + address[0] + ":" + str(address[1]))
    return conn

# Receive a photograph from the Pi
def receive_photo(conn):
	dateStr = dt.datetime.now().strftime("%Y-%b-%d-%H%M%S")
	dir_path = "./received_photos/" + dateStr + "/"
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)
	path = "./received_photos/" + dateStr + "/" + dateStr + ".jpg"
	file = open(path, 'w+')
	pic = conn.recv(1024)
	while True:
		file.write(pic)
		pic = conn.recv(1024)
		if pic == '1':
			print "Received photograph..."
			break
	file.close()
	return dateStr

def main():
	print_welcome()
	db_status = get_db_status()
	if db_status:
		# Using existing database, pick it?
		print "Wait for version 1.0!"
	else:
		# Using new database, create it and take photo from Pi to
		# use for manual space identifier setup
		print "Creating new database..."
		create_db()
		print "New database created."

		# Manually assign spaces...
		print "Setting up TCP socket..."
		conn = init_socket()
		print "TCP socket created."

		print "Taking initial photo..."
		conn.send("one_time")
		f = receive_photo(conn)
		print "Processing initial photo..."
		assign_spaces(f)

		# Continue accepting photo data from the Pi indefinitely
		conn.send("Auto")
		while True:
			f = receive_photo(conn)
			[markers, image] = process_image(f)
			print str(len(markers)) + " markers detected."
			update_availability(get_parking_data(), markers)

if __name__ == '__main__': 
  main()

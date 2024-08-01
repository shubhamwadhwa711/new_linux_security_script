import os
import csv
import configparser
import logging
from datetime import datetime

# Define the log directory and log files
LOG_DIR = 'logs'
LOG_FILES = {
    'h1_tags': 'h1_tags_logs.csv',
    'descriptions': 'descriptions_logs.csv',
    'title': 'title_logs.csv'
}

config = configparser.ConfigParser(interpolation=None)
config.read(os.path.join(os.path.dirname(__file__), "config.ini"))
LOG_FILE = config.get('metadata-01', 'log_file')
h1_max_length = int(config.get('max_length', 'h1_tag'))
title_max_length = int(config.get('max_length', 'title'))
description_max_length = int(config.get('max_length', 'description'))

# Ensure the log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

def log_error(data_dictionary, id, connection, field_id=29):
    from helperfunctions import checkH1Unique, checkTitleUnique
    h1_tag = data_dictionary.get('H1')
    title = data_dictionary.get('Title')
    description = data_dictionary.get('Description')

    error_list = []
    if len(h1_tag) > h1_max_length:
        temp = {
            'file_path': os.path.join(LOG_DIR, LOG_FILES.get('h1_tags')),
            'value': h1_tag,
            'id': id,
            'type': "H1 Tag",
            'reason': "max length exceed"
        }
        error_list.append(temp)

    if not checkH1Unique(connection, h1_tag):
        temp = {
            'file_path': os.path.join(LOG_DIR, LOG_FILES.get('h1_tags')),
            'value': h1_tag,
            'id': id,
            'type': "H1 Tag",
            'reason': "Not Unique"
        }
        error_list.append(temp)
    
    if len(title) > title_max_length:
        temp = {
            'file_path': os.path.join(LOG_DIR, LOG_FILES.get('title')),
            'value': title,
            'id': id,
            'type': "Title",
            'reason': "max length exceed"
        }
        error_list.append(temp)

    if not checkTitleUnique(connection, title, field_id):
        temp = {
            'file_path': os.path.join(LOG_DIR, LOG_FILES.get('title')),
            'value': title,
            'id': id,
            'type': "Title",
            'reason': "Not Unique"
        }
        error_list.append(temp)
    
    if len(description) > description_max_length:
        temp = {
            'file_path': os.path.join(LOG_DIR, LOG_FILES.get('descriptions')),
            'value': description,
            'id': id,
            'type': "Description",
            'reason': "max length exceed"
        }
        error_list.append(temp)

    for error in error_list:
        log_file_path = os.path.join(LOG_DIR, "Operations_Logs.csv")
        log_entry = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Article_ID': error['id'],
            'Error_Type': error['type'],
            'Info': error['reason'],
            'Value': error['value']
        }

        # Write the log entry to the CSV file
        file_exists = os.path.exists(log_file_path)

        with open(log_file_path, 'a', newline='') as log_file:
            fieldnames = ['Timestamp', 'Article_ID', 'Error_Type', 'Info', 'Value']
            writer = csv.DictWriter(log_file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(log_entry)

def setup_logger(logger, log_file_path, log_type):
    # Create a file handler
    file_handler = logging.FileHandler(log_file_path)
    # Create a console handler
    console_handler = logging.StreamHandler()

    # Set log levels for both handlers based on log_type
    level = {
        "info": logging.INFO,
        "error": logging.ERROR,
        "warn": logging.WARNING
    }.get(log_type, logging.INFO)

    file_handler.setLevel(level)
    console_handler.setLevel(level)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def log_info(message, log_type="info"):
    # Create a logger object
    log_file_path = LOG_FILE
    logger = logging.getLogger('log_info_logger')
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of logging

    # Check if handlers already exist to avoid duplicate logs
    if not logger.hasHandlers():
        setup_logger(logger, log_file_path, log_type)

    # Log the provided message with appropriate log level
    if log_type == "error":
        logger.error(message)
    elif log_type == "warn":
        logger.warning(message)
    else:
        logger.info(message)
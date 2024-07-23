import os
import json
import configparser
import logging
from datetime import datetime

# Define the log directory and log files
LOG_DIR = 'logs'
LOG_FILES = {
    'h1_tags': 'h1_tags_logs.json',
    'descriptions': 'descriptions_logs.json',
    'title': 'title_logs.json'
}

config = configparser.ConfigParser(interpolation=None)
config.read(os.path.join(os.path.dirname(__file__), "config.ini"))
LOG_FILE = config.get('metadata-01', 'log_file')
h1_max_length = int(config.get('max_length', 'h1_tag'))
title_max_length = int(config.get('max_length', 'title'))
description_max_length = int(config.get('max_length', 'description'))

# Ensure the log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

def log_error(data_dictionary, id):
    h1_tag = data_dictionary.get('H1')
    title = data_dictionary.get('Title')
    description = data_dictionary.get('Description')

    error_list = []
    if len(h1_tag)>h1_max_length:
        temp = {}
        temp['file_path'] = os.path.join(LOG_DIR, LOG_FILES.get('h1_tags'))
        temp['value'] = h1_tag
        temp['id'] = id
        temp['type'] = "H1 Tag"
        error_list.append(temp)
    
    if len(title)>title_max_length:
        temp = {}
        temp['file_path'] = os.path.join(LOG_DIR, LOG_FILES.get('title'))
        temp['value'] = title
        temp['id'] = id
        temp['type'] = "Title"
        error_list.append(temp)
    
    if len(description)>description_max_length:
        temp = {}
        temp['file_path'] = os.path.join(LOG_DIR, LOG_FILES.get('descriptions'))
        temp['value'] = description
        temp['id'] = id
        temp['type'] = "Description"
        error_list.append(temp)

    for error in error_list:
        log_file_path = error.get('file_path')
        log_entry = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Info': f"Limit of {error['type']} with content ID {error['id']} exceeded.",
            'Value': error['value']
        }

        # Read existing logs
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as log_file:
                try:
                    logs = json.load(log_file)
                except json.JSONDecodeError:
                    logs = []
        else:
            logs = []

        # Append the new log entry
        logs.append(log_entry)

        # Write updated logs back to the file
        with open(log_file_path, 'w') as log_file:
            json.dump(logs, log_file, indent=4)

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
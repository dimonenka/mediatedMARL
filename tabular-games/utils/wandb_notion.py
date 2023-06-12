import requests
import json
from datetime import datetime
from utils.build_json import build_json
import os


def set_environ():
    key = str(input('Enter your Notion API key: \n'))
    os.environ['NOTION_API_KEY'] = key


# def prepare_page_id(page_id):
#     page_id = f'{page_id[:8]}-{page_id[8:12]}-{page_id[12:16]}-{page_id[16:20]}-{page_id[20:]}'
#
#     return page_id


def commit_to_notion(page_id, wb_link, file_path='default.json'):
    if 'NOTION_API_KEY' not in os.environ:
        set_environ()

    url = f'https://api.notion.com/v1/blocks/{page_id}/children'
    payload = build_json(wb_link, file_path)

    headers = {
        "Accept": "application/json",
        "Notion-Version": "2021-08-16",  # one day it may become out-of-date
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ['NOTION_API_KEY']
    }

    response = requests.patch(url, json.dumps(payload), headers=headers)

    if response.status_code == 200:
        print(f'Successfully added experiment to Notion: https://www.notion.so/{page_id}\n')
    else:
        print('Something went wrong! Check the answer from API:')
        print(response.json())
        raise

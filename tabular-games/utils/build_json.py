import json
from datetime import datetime


def build_json(wb_link, path):
    with open(path) as json_file:
        data = json.load(json_file)

    time_now = datetime.now().isoformat()
    data['children'][1]['paragraph']['text'][0]['mention']['date']['start'] = str(time_now)
    data['children'][2]['paragraph']['text'][1]['text']['content'] = f'{wb_link}'
    data['children'][2]['paragraph']['text'][1]['text']['link']['url'] = f'{wb_link}'

    return data

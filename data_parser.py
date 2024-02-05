import datetime
import requests
import threading as td
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from time import sleep
from tqdm import tqdm

multithread = True
active_threads = 0

data = {}

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 \
        (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'
}


months = {
    1: 'января',
    2: 'февраля',
    3: 'марта',
    4: 'апреля',
    5: 'мая',
    6: 'июня',
    7: 'июля',
    8: 'августа',
    9: 'сентября',
    10: 'октября',
    11: 'ноября',
    12: 'декабря'
}

days_in_month = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31
}

wind_directions = ['С', 'ССВ', 'СВ', 'ВСВ', 'В', 'ВЮВ', 'ЮВ',
                   'ЮЮВ', 'Ю', 'ЮЮЗ', 'ЮЗ', 'ЗЮЗ', 'З', 'ЗСЗ', 'СЗ', 'ССЗ']

wind_dir_to_vector = {}

for i, wind_dir in enumerate(wind_directions):
    wind_dir_to_vector[wind_dir] = np.array(
        [np.cos(i * np.pi / 8), np.sin(i * np.pi / 8)])


def parse_water_temp(year, data) -> None:
    global active_threads
    
    active_threads += 1
    url = f'http://blacksea-map.ru/sst/doc/sosnybor_{year}.html'

    response = requests.get(url)

    if response.status_code != 200:
        print(f'Error: {response.status_code}')
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table', class_='mt')

    for month, row in enumerate(table.find_all('tr')[3:-1]):
        for day, temp in enumerate(row.find_all('td')):
            try:
                date = datetime.date(year, month + 1, day + 1)
            except ValueError:
                # day number greater than month length
                continue

            temp = float(temp.text.strip().replace(',', '.')
                         ) if temp.text.strip() else None

            if (date not in data):
                data[date] = {}

            data[date]['water_temp'] = temp
    active_threads -= 1

def parse_weather_gismeteo(year, month, data) -> None:

    url = f'https://www.gismeteo.ru/diary/162477/{year}/{month}/'

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f'Error: {response.status_code}')
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('div', id='data_block').find('tbody')
    if table is None:
        print(f'{year}-{month} has no data')
        return None

    for row in table.find_all('tr'):
        cols = row.find_all('td')
        if len(cols) < 11:
            continue

        try:
            c_day_img_scr = cols[3].find('img')['src']
            if c_day_img_scr.endswith('sun.png'):
                c_day = 0
            elif c_day_img_scr.endswith('sunc.png'):
                c_day = 1
            elif c_day_img_scr.endswith('suncl.png'):
                c_day = 2
            elif c_day_img_scr.endswith('dull.png'):
                c_day = 3
            else:
                c_day = np.nan
        except TypeError:
            c_day = np.nan

        try:
            c_night_img_scr = cols[8].find('img')['src']
            if c_night_img_scr.endswith('sun.png'):
                c_night = 0
            elif c_night_img_scr.endswith('sunc.png'):
                c_night = 1
            elif c_night_img_scr.endswith('suncl.png'):
                c_night = 2
            elif c_night_img_scr.endswith('dull.png'):
                c_night = 3
            else:
                c_night = np.nan
        except TypeError:
            c_night = np.nan

        cols_text = [col.text.strip() if col.text.strip() not in (
            '', '-', '−') else np.nan for col in cols]
        day, t_day, ps_day, _, _, w_day, t_night, ps_night, _, _, w_night = cols_text

        if w_day is not np.nan and w_day != 'Ш':
            w_dir_day, w_speed_day = w_day.split()
            w_speed_day = float(w_speed_day[:-3])
        else:
            w_dir_day, w_speed_day = np.nan, 0

        if w_night is not np.nan and w_night != 'Ш':
            w_dir_night, w_speed_night = w_night.split()
            w_speed_night = float(w_speed_night[:-3])
        else:
            w_dir_night, w_speed_night = np.nan, 0

        if t_day is np.nan:
            mean_temp = t_night
        elif t_night is np.nan:
            mean_temp = t_day
        else:
            mean_temp = (float(t_day) + float(t_night)) / 2

        if ps_day is np.nan:
            mean_pressure = ps_night
        elif ps_night is np.nan:
            mean_pressure = ps_day
        else:
            mean_pressure = (float(ps_day) + float(ps_night)) / 2

        if c_day is np.nan:
            mean_cloudiness = c_night
        elif c_night is np.nan:
            mean_cloudiness = c_day
        else:
            mean_cloudiness = (float(c_day) + float(c_night)) / 2

        vector_by_dir = {
            'С': np.array([0, 1]),
            'СВ': np.array([1, 1]) / np.sqrt(2),
            'В': np.array([1, 0]),
            'ЮВ': np.array([1, -1]) / np.sqrt(2),
            'Ю': np.array([0, -1]),
            'ЮЗ': np.array([-1, -1]) / np.sqrt(2),
            'З': np.array([-1, 0]),
            'СЗ': np.array([-1, 1]) / np.sqrt(2),
            np.nan: np.array([0, 0])
        }

        mean_wind = (
            w_speed_day * vector_by_dir[w_dir_day] + w_speed_night * vector_by_dir[w_dir_night]) / 2
        mean_wind_x, mean_wind_y = mean_wind

        date = datetime.date(year, month, int(day))

        water_temp = data[date]['water_temp'] if date in data else np.nan
        data[date] = {
            'water_temp': water_temp,

            'air_temp_mean': mean_temp,
            'air_temp_day': t_day,
            'air_temp_night': t_night,

            'pressure_mean': mean_pressure,
            'pressure_day': ps_day,
            'pressure_night': ps_night,

            'cloudiness_mean': mean_cloudiness,
            'cloudiness_day': c_day,
            'cloudiness_night': c_night,

            'wind_x_mean': mean_wind_x,
            'wind_y_mean': mean_wind_y,
            'wind_speed_mean': (mean_wind_x**2 + mean_wind_y**2)**0.5,

            'wind_x_day': w_speed_day * vector_by_dir[w_dir_day][0],
            'wind_y_day': w_speed_day * vector_by_dir[w_dir_day][1],
            'wind_speed_day': w_speed_day,

            'wind_x_night': w_speed_night * vector_by_dir[w_dir_night][0],
            'wind_y_night': w_speed_night * vector_by_dir[w_dir_night][1],
            'wind_speed_night': w_speed_night,
        }


def parse_weather_nuipogoda(year, month, data, day=-1) -> None:
    global active_threads
    if day == -1:
        for day in range(1, days_in_month[month] + 1):
            if multithread:
                td.Thread(target=parse_weather_nuipogoda,
                          args=(year, month, data, day)).start()
            else:
                parse_weather_nuipogoda(year, month, data, day)
    else:
        active_threads += 1
        date = datetime.date(year, month, day)

        url = f'https://sosnoviy-bor.nuipogoda.ru/{day}-{months[month]}#{year}'
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f'Error: {response.status_code}')
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', class_='weather').find('tbody')
        if table is None:
            print(f'{year}-{month}-{day} has no data')
            return None

        rows = table.find_all('tr')[::2]

        del response, soup, table

        water_temp = data[date]['water_temp'] if date in data else np.nan
        cur_day_data = {
            'water_temp': water_temp
        }
        for i_row, row in enumerate(rows):
            cols = row.find_all('td')[:-1]
            if i_row == 0:
                cols = cols[1:]
            cloudiness, precipitation = cols[0].next['title'].split(', ')
            hour = int(cols[1].text.strip().split(':')[0])
            temp = float(cols[2].text.strip()[:-1])
            wind_dir = cols[3].find('span', class_='wd').text.strip()
            if wind_dir == 'штиль':
                wind_speed = 0
                wind_dir_x, wind_dir_y = 0, 0
            else:
                wind_speed = float(cols[3].find(
                    'span', class_='ws').text.strip().split()[0])
                wind_dir_x, wind_dir_y = wind_dir_to_vector[wind_dir] * wind_speed
            pressure = float(cols[4].text.strip().split()[0][:-2])

            cur_day_data = {
                **cur_day_data,
                f'cloudiness_{hour}': cloudiness,
                f'precipitation_{hour}': precipitation,
                f'temp_{hour}': temp,
                f'wind_dir_{hour}': wind_dir,
                f'wind_speed_{hour}': wind_speed,
                f'wind_dir_x_{hour}': wind_dir_x,
                f'wind_dir_y_{hour}': wind_dir_y,
                f'pressure_{hour}': pressure
            }

        data[date] = cur_day_data
        active_threads -= 1

print('Parsing water temperature...')
for _year in tqdm(range(2011, 2023), desc='Year'):
    if multithread:
        td.Thread(target=parse_water_temp, args=(_year, data)).start()
        sleep(0.1)
    else:
        parse_water_temp(_year, data)

while active_threads > 0:
    sleep(3)

print('Water temperature parsed successfully!\n\n\n')

print('Parsing weather...')
for _year in tqdm(range(2011, 2023), desc='Year'):
    for _month in tqdm(range(1, 13), desc='Month', leave=False):
        parse_weather_nuipogoda(_year, _month, data)
    while td.active_count() > 64:
        sleep(3)


while active_threads > 0:
    sleep(3)
print(f'Weather parsed successfully!\n\n\n')


df = pd.DataFrame.from_dict(data, orient='index')
df.index.name = 'date'

print("Missed values:")
print(df.isna().sum() / len(df) * 100)

df.to_csv('data.csv')
print(df.head())

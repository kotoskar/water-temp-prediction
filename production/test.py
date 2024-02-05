from sklearn.linear_model import Lasso
from bs4 import BeautifulSoup
import requests
import numpy as np
import pickle
import datetime
import pandas as pd
from astral.sun import sun
from astral import LocationInfo
from pysolar.solar import get_altitude
import os

base_path = os.path.dirname(os.path.abspath(__file__))

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

wind_directions = ['С', 'ССВ', 'СВ', 'ВСВ', 'В', 'ВЮВ', 'ЮВ',
                   'ЮЮВ', 'Ю', 'ЮЮЗ', 'ЮЗ', 'ЗЮЗ', 'З', 'ЗСЗ', 'СЗ', 'ССЗ']

wind_dir_to_vector = {}

for i, wind_dir in enumerate(wind_directions):
    wind_dir_to_vector[wind_dir] = np.array(
        [np.cos(i * np.pi / 8), np.sin(i * np.pi / 8)])

def weather(date):
    day, month, year = date.day, date.month, date.year
    
    url = f'https://sosnoviy-bor.nuipogoda.ru/{day}-{months[month]}#{year}'
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f'Error: {response.status_code}')
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', class_='weather').find('tbody')
    if table is None:
        print(f'Error: {year}-{month}-{day} has no data')
        return None

    rows = table.find_all('tr')[::2]
    del response, soup, table
    
    cur_day_data = {}
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

    return cur_day_data

def today_water_temp():
    url = 'http://russia.pogoda360.ru/526529/water'
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f'Error: {response.status_code}')
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    water_block = soup.find('div', id='sstPanel')
    water_temp = water_block.find('td', class_='temp').text.strip()[1:-2]
    if water_temp == '':
        print(f'Error: water temperature is not available')
        return None
    return float(water_temp)
    

latitude, longitude = 59.5, 29.1
city = LocationInfo("Sosnovy Bor", "Russia", "Europe/Moscow", latitude, longitude)

def get_day_length(x):
    s = sun(city.observer, datetime.date.fromisoformat(x))
    return (s['sunset'] - s['sunrise']).seconds / 3600

def solar_elevation(time):
    s = sun(city.observer, datetime.date.fromisoformat(time))    
    return get_altitude(latitude, longitude, s['noon'])

def prepare(X, day):
    X['date'] = f'{day.year}-{day.month:02d}-{day.day:02d}'
    X['month'] = X['date'].apply(lambda x: int(x.split('-')[1]))
    X['day'] = X['date'].apply(lambda x: int(x.split('-')[2]))
    X['day_length'] = X['date'].apply(get_day_length)
    X['solar_elevation'] = X['date'].apply(solar_elevation)
    X.drop('date', axis=1, inplace=True)
    X.drop(columns=[col for col in X.columns if col.endswith('_24')], inplace=True)
    categorical = [col for col in X.columns if X[col].dtype == 'O'] + ['month', 'day']
    
    encoder = pickle.load(open(base_path + '\\..\\models\\encoder.pkl', 'rb'))
    transformed = encoder.transform(X[categorical])
    X.drop(columns=categorical, axis=1, inplace=True)
    X = pd.concat([X, pd.DataFrame(transformed)], axis=1)
    X.columns = X.columns.astype(str)
    return X

def main():
    model = pickle.load(open(base_path + '\\..\\models\\model.pkl', 'rb'))
    
    today = datetime.date.today()
    X = pd.DataFrame(weather(today), index=[0])
    X = prepare(X, today)
    
    water_temp = today_water_temp()
    
    print(f'Predicted water temperature for the next 14 days:')
    print(f'{today}: {water_temp}')
    for i in range(1, 14):
        cur_day = today + datetime.timedelta(days=i)
        last_temp = pd.DataFrame({'water_temp': water_temp}, index=[0])
        predicted_temp = model.predict(pd.concat([X, last_temp], axis=1).values)
        last_temp = pd.DataFrame({'water_temp': predicted_temp}, index=[0])
        X =pd.DataFrame(weather(cur_day), index=[0])
        X = prepare(X, cur_day)
        print(f'{cur_day}: {predicted_temp[0]:.2f}')
    
if __name__ == '__main__':
    main()
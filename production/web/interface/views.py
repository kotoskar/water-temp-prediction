from django.shortcuts import render
from .parser.parser import get_prediction_for_day, today_water_temp
from .models import Prediction
import datetime

def index(request):
    today = datetime.date.today()
    need_to_update = False
    if Prediction.objects.filter(date=today).exists():
        obj = Prediction.objects.get(date=today)
        if (obj.actual_temperature is None):
            temp = today_water_temp()
            if temp is not None:
                obj.actual_temperature = temp
                obj.save()
                need_to_update = True
            else:
                print('Could not get actual temperature for today')
    else:
        temp = today_water_temp()
        if temp is not None:
            Prediction.objects.create(date=today, actual_temperature=temp)
            need_to_update = True
        else:
            print('Could not get actual temperature for today')
        
    if need_to_update:
        print(f'Updating predictions from {today + datetime.timedelta(days=1)} to {today + datetime.timedelta(days=14)}')
        temp = Prediction.objects.get(date=today).actual_temperature
        for i in range(14):
            cur_day = today + datetime.timedelta(days=i+1)
            temp = get_prediction_for_day(cur_day, temp)
            Prediction.objects.create(date=cur_day, predicted_temperature=temp)
    
    predictions = []
    for i in range(14):
        cur_day = today + datetime.timedelta(days=i + 1)
        obj = Prediction.objects.get(date=cur_day)
        predictions.append((obj.date, round(obj.predicted_temperature, 2)))
            
    data = {'predictions': predictions}
    return render(request, 'interface/index.html', data)

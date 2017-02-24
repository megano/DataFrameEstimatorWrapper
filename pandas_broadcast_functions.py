from datetime import timedelta
import math as m
from datetime import datetime
import pandas as pd
import re as re
#extract numbers from comma separated values

WEEK_DAY = ['MON','TUE','WED','THU','FRI','SAT','SUN']

def broadcast_function(function, np_arr) :
    f = np.vectorize(function)
    return f(np_arr)

def extract_numbers(inp) :
    typ = type(inp)

    if(typ == float) | (typ == int) :
        return inp
    try :
        return float(inp.replace(",",""))
    except :
        return np.NaN

#get integer
def get_integer(inp) :
    inp = extract_numbers(inp)
    if(not inp) | (m.isnan(inp)) | (inp == np.NaN) :
        return np.NaN
    inp = int(inp)
    return inp

#date function
def strip_date(date_str, format) :
    x = datetime.strptime(date_str, format)
    serv = []
    d = x.date()

    date = d

    day = d.day
    month = d.month
    year = d.year
    quarter = (month-1)/3 + 1

    week_day = d.weekday()
    week_number = d.strftime("%V")
    monday_date = x - timedelta(days=week_day)
    day_of_year = d.strftime("%j")

    hour = x.hour
    minutes = x.minute
    seconds = x.second

    return {'day':day,
            'date':d,
           'month':month,
           'year':year,
           'week_day':WEEK_DAY[week_day],
           'week_number':week_number,
           'day_of_year':int(day_of_year),
           'monday_date':monday_date,
           'quarter':quarter,
           'hour':hour,
           'minutes':minutes,
           'seconds':seconds}

#get deiffernce dates
def date_difference(d1, d2) :
    try :
       return (d1 - d2).days
    except:
        print "error in finding date differences"
        return

#
def feet_inches_to_feet(num, feet_symb = "'", inches_symb = '"') :
    typ = type(num)

    if(typ == int) | (typ == float) :
        return num

    spl = num.split(feet_symb)

    feet = float(num[0 : num.index(feet_symb)]) if (feet_symb in num) else 0
    num = num[num.index(feet_symb)+1:] if (feet_symb in num) else num
    inch = float(num[0 : num.index(inches_symb)]) if (inches_symb in num) else 0

    return feet + (inch * 1./12)

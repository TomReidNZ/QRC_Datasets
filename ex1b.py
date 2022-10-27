'''Methods for use in Exercise 1b'''
from datetime import datetime, tzinfo
from math import floor
import pytz

def print_time():
    current_time = datetime.now().strftime("%H:%M:%S")
    print("Current Time =", current_time)


def print_time_and_date():
    print(get_time_in("NZ"))


def say_happy_birthday(name:str):
    # str() below in case they provide a number while exploring
    print("Happy Birthday " + str(name) + ", I hope your day goes well!")
    

def comment_on_age(name:str, age:float):
    if type(name) is not str:
        print('Error! The first argument should be a string. Write it in "quotes"')
        return

    if not ((type(age) is int) or (type(age) is float)):
        print('Error! The second argument should be a number. Do not write it in quotes')
        return
    
    message = "It looks like " + name
    if age < 0:
        message += " has yet to be born"
    elif age <= 12:
        message += " is a child"
    elif age < 20:
        message += " is a teenager"  
    elif age < 100:
        message += " is in their " + str(floor(age / 10) * 10) + "s"
    else:
        message += " is probably someone with a lot of wisdom"
    
    print(message)


def get_time_in_new_york():
    '''Returns datetime.now for NY, USA'''
    return get_time_in("America/New_York")


def get_time_in_adelaide():
    '''Returns datetime.now for Adelaide, Australia'''
    return get_time_in("Australia/Adelaide")


def get_time_in(timezone:str):
    '''Returns datetime.now for a specific time zone'''
    tz = pytz.timezone(timezone)
    return datetime.now(tz)
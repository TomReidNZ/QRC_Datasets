def predict_favourite_fruit(age:int):
    '''Predicts favourite fruit based on someone's age'''
    fruit = ["Apple", "Orange", "Banana", "Watermelon", "Rockmelon", "Strawberry", "Feijoa", "Lemon"]
    return fruit[int(age) % len(fruit)]
import streamlit as st
import pandas as pd
import numpy as np
from copy import deepcopy
from math import *
import warnings
import itertools
from multiprocessing import Pool
from functools import partial
import time
import os

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def HaltonSequence(n, dim):
    n = int(n)
    dim = int(dim)
    prim = np.array(
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
         109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
         233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
         283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421,
         431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541])
    prim = prim[:, np.newaxis]
    hs = np.zeros((n, dim))
    for idim in range(dim):
        b = prim[idim, 0]
        hs[:, idim] = halton(n, b)

    return (hs[10:n, :])


def halton(n, s):
    k = floor(log(n + 1) / log(s))
    phi = np.zeros((1, 1))
    i = 1
    count = 0
    while i <= k:
        count = count + 1
        x = phi
        j = 1
        while j < s:
            y = phi + (j / s ** i)
            x = np.vstack((x, y))
            j += 1

        phi = x
        i += 1

    x = phi
    j = 1
    while (j < s) and (len(x) < (n + 1)):
        y = phi + (j / s ** i)
        x = np.vstack((x, y))
        j += 1

    out = x[1:(n + 1), 0]
    return (out)

row_wise = 1
col_wise = 0
Windows_OS = 1  # 1 for Windows, 0 for Linux
Starting_sample_size = 802
Generate_Draws = 1

current_dir = os.path.dirname(os.path.abspath(__file__))
Data_file_name  = os.path.join(current_dir,'MDC_Final_data_Scenario.csv')  # Name of the data file
Parameter_file  = os.path.join(current_dir,'Logsum_Parameters.csv')
price_file_name  = os.path.join(current_dir,'Price_list.csv')
filter_file_name = os.path.join(current_dir,'Filter_list.csv')
nrep = 500
Consumption_Normalize = 1  # 1 to use expenditure as proportion
Expenditure_Actual = 0

Category_product = {'Burger': [13, 14, 15, 16, 17, 18, 19],
                    'Combos': [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
                    'Drinks': [33, 34, 35, 36, 37, 38],
                    'Fried Chicken': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'Ice cream & Sweets': [26, 27, 28],
                    'Milk Shake': [29, 30, 31, 32],
                    'Rice': [11, 12],
                    'Snacks': [20, 21, 22, 23, 24, 25]}

Category_Wise_Collection = {13: [14, 15, 16, 17, 18, 19],
                            14: [13, 15, 16, 17, 18, 19],
                            15: [13, 14, 16, 17, 18, 19],
                            16: [13, 14, 15, 17, 18, 19],
                            17: [13, 14, 15, 16, 18, 19],
                            18: [13, 14, 15, 16, 17, 19],
                            19: [13, 14, 15, 16, 17, 18],
                            39: [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
                            40: [39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
                            41: [39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
                            42: [39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
                            43: [39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
                            44: [39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53],
                            45: [39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53],
                            46: [39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53],
                            47: [39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53],
                            48: [39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53],
                            49: [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53],
                            50: [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53],
                            51: [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53],
                            52: [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53],
                            53: [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52],
                            33: [34, 35, 36, 37, 38],
                            34: [33, 35, 36, 37, 38],
                            35: [33, 34, 36, 37, 38],
                            36: [33, 34, 35, 37, 38],
                            37: [33, 34, 35, 36, 38],
                            38: [33, 34, 35, 36, 37],
                            1: [2, 3, 4, 5, 6, 7, 8, 9, 10],
                            2: [1, 3, 4, 5, 6, 7, 8, 9, 10],
                            3: [1, 2, 4, 5, 6, 7, 8, 9, 10],
                            4: [1, 2, 3, 5, 6, 7, 8, 9, 10],
                            5: [1, 2, 3, 4, 6, 7, 8, 9, 10],
                            6: [1, 2, 3, 4, 5, 7, 8, 9, 10],
                            7: [1, 2, 3, 4, 5, 6, 8, 9, 10],
                            8: [1, 2, 3, 4, 5, 6, 7, 9, 10],
                            9: [1, 2, 3, 4, 5, 6, 7, 8, 10],
                            10: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                            26: [27, 28],
                            27: [26, 28],
                            28: [26, 27],
                            29: [30, 31, 32],
                            30: [29, 31, 32],
                            31: [29, 30, 32],
                            32: [29, 30, 31],
                            11: [12],
                            12: [11],
                            20: [21, 22, 23, 24, 25],
                            21: [20, 22, 23, 24, 25],
                            22: [20, 21, 23, 24, 25],
                            23: [20, 21, 22, 24, 25],
                            24: [20, 21, 22, 23, 25],
                            25: [20, 21, 22, 23, 24]}

Ala_carte = {
    1: 'Special Bone-in Chicken (original)',
    2: 'Special Bone-in Chicken (spicy)',
    3: 'Devils Claw Chicken Steak',
    4: 'Cheese Chicken Tenders',
    5: 'Cheese Popcorn Chicken',
    6: 'Cajun Spicy Wings',
    7: 'Crispy Fried Chicken',
    8: 'Rolled Chicken Tenders',
    9: 'Bone-in Chicken with Sweet and Spicy Sauce',
    10: 'Cheese Fried Chicken Nuggets',
    11: 'Louisiana Assorted Seafood Rice',
    12: 'Coconut Cajun Crispy Chicken Rice',
    13: 'Low-cal Low-fat Chicken Burger',
    14: 'Crispy Chicken Burger with Sweet and Spicy Sauce',
    15: 'Cheese Chicken Burger (original)',
    16: 'Cheese Chicken Burger (spicy)',
    17: 'Cajun Grilled Chicken Burger',
    18: 'Salsa-inspired Pineapple Cod Burger',
    19: 'Tiny Chicken Burger',
    20: 'Fried Cheese Sticks',
    21: 'Fried Prawns',
    22: 'Cajun Grilled Wings',
    23: 'Cajun French Fries (big)',
    24: 'Cajun French Fries (middle)',
    25: 'Cajun Mashed Potatoes',
    26: 'Sundae (mango orange/salt caramel)',
    27: 'Cone (vanilla/chocolate)',
    28: 'Louisiana Egg Tart',
    29: 'Strawberry Milkshake',
    30: 'Chocolate Milkshake',
    31: 'Mango Orange Milkshake',
    32: 'Grapefruit Longjing Milkshake',
    33: 'Mango Orange Sparkling Drink',
    34: 'Coconut Banana Sparkling Drink',
    35: 'Coke',
    36: 'Coke Without Sugar',
    37: 'Orange Juice',
    38: 'Lemon Tea'}

Combo_Mapping = {
    39: ['Special Bone-in Chicken (original)', 'Cajun Spicy Wings', 'Coke'],
    40: ['Special Bone-in Chicken (original)', 'Cheese Chicken Tenders', 'Cajun Spicy Wings', 'Coke'],
    41: ['Special Bone-in Chicken (original)', 'Cheese Chicken Tenders', 'Cajun Spicy Wings',
         'Bone-in Chicken with Sweet and Spicy Sauce', 'Coke', 'Coke Without Sugar'],
    42: ['Cheese Popcorn Chicken', 'Louisiana Assorted Seafood Rice', 'Coke'],
    43: ['Cajun Spicy Wings', 'Coconut Cajun Srispy Chicken Rice', 'Coke'],
    44: ['Cajun Spicy Wings', 'Cajun Grilled Chicken Burger', 'Coke'],
    45: ['Salsa-inspired Pineapple Cod Burger', 'Fried Prawns', 'Coke'],
    46: ['Salsa-inspired Pineapple Cod Burger', 'Fried Prawns', 'Cajun French fries (middle)', 'Coke'],
    47: ['Cheese Chicken Burger (original)', 'Cajun French Fries (middle)', 'Coke'],
    48: ['Cajun Spicy Wings', 'Cheese Chicken Burger (original)', 'Cajun French Fries (middle)', 'Coke'],
    49: ['Cheese Chicken Tenders', 'Cheese Popcorn Chicken', 'Cajun French Fries (big)', 'Strawberry Milkshake',
         'Chocolate Milkshake'],
    50: ['Cheese Chicken Tenders', 'Fried Prawns', 'Cajun French Fries (big)', 'Mango Orange Milkshake',
         'Grapefruit Longjing Milkshake'],
    51: ['Special Bone-in Chicken (original)', 'Special Bone-in Chicken (spicy)', 'Cheese Chicken Tenders',
         'Cheese Popcorn Chicken', 'Cajun Spicy Wings', 'Crispy Fried Chicken', 'Rolled Chicken Tenders',
         'Cajun Grilled Wings', 'Mango Orange Milkshake', 'Grapefruit Longjing Milkshake',
         'Mango Orange Sparkling Drink', 'Coconut Banana Sparkling Drink', 'Coke', 'Orange Juice'],
    52: ['Cajun Grilled Chicken Burger', 'Tiny Chicken Burger', 'Cajun French Fries (middle)', 'Louisiana Egg Tart',
         'Coke', 'Orange Juice'],
    53: ['Cheese Chicken Burger (original)', 'Cajun Grilled Chicken Burger', 'Tiny Chicken Burger', 'Fried Prawns',
         'Cajun French Fries (big)', 'Louisiana Egg Tart', 'Strawberry Milkshake', 'Coke', 'Orange Juice']}

Mapping_dict = [[1, 39, 40, 41, 51],
                [2, 51],
                [3],
                [4, 40, 41, 49, 50, 51],
                [5, 42, 49, 51],
                [6, 39, 40, 41, 43, 44, 48, 51],
                [7, 51],
                [8, 51],
                [9, 41],
                [10],
                [11, 42],
                [12],
                [13],
                [14],
                [15, 47, 48, 53],
                [16],
                [17, 44, 52, 53],
                [18, 45, 46],
                [19, 52, 53],
                [20],
                [21, 45, 46, 50, 53],
                [22, 51],
                [23, 49, 50, 53],
                [24, 47, 48, 52],
                [25],
                [26],
                [27],
                [28, 52, 53],
                [29, 49, 53],
                [30, 49],
                [31, 50, 51],
                [32, 50, 51],
                [33, 51],
                [34, 51],
                [35, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 51, 52, 53],
                [36, 41],
                [37, 51, 52, 53],
                [38]]

Gamma_List = [
    ['uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast',
     'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast',
     'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast',
     'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast',
     'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast',
     'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast',
     'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast',
     'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast',
     'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast',
     'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast',
     'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone',
     'Family'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone',
     'Family']]

Utility_Spec = [
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_1_PriceC', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_1_Pcomb', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_1_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Burger_logsum', 'Drinks_logsum', 'Milk Shake_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_2_PriceC', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_2_Pcomb', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_2_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Burger_logsum', 'Drinks_logsum', 'Milk Shake_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_3_PriceC',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_3_Pcomb', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_3_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Burger_logsum', 'Drinks_logsum', 'Milk Shake_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_4_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_4_Pcomb',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_4_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'Burger_logsum', 'Drinks_logsum', 'Milk Shake_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_5_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_5_Pcomb',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_5_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Burger_logsum', 'Drinks_logsum', 'Milk Shake_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'Alt_6_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_6_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_6_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'Burger_logsum', 'Drinks_logsum', 'Milk Shake_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'Alt_7_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_7_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_7_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Burger_logsum', 'Drinks_logsum', 'Milk Shake_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Alt_8_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'Alt_8_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_8_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Burger_logsum', 'Drinks_logsum', 'Milk Shake_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Alt_9_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'Alt_9_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_9_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Burger_logsum', 'Drinks_logsum', 'Milk Shake_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_10_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Alt_10_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_10_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Burger_logsum', 'Drinks_logsum', 'Milk Shake_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_11_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Alt_11_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_11_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Snacks_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_12_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_12_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_12_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Snacks_logsum', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_13_PriceC', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_13_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_13_logsum', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Fried Chicken_logsum', 'Drinks_logsum', 'Milk Shake_logsum', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_14_PriceC', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_14_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_14_logsum', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Fried Chicken_logsum', 'Drinks_logsum', 'Milk Shake_logsum', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_15_PriceC', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_15_Pcomb', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_15_logsum', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Fried Chicken_logsum', 'Drinks_logsum', 'Milk Shake_logsum', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_16_PriceC', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_16_Pcomb', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_16_logsum', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Fried Chicken_logsum', 'Drinks_logsum', 'Milk Shake_logsum', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_17_PriceC',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_17_Pcomb', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_17_logsum', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Fried Chicken_logsum', 'Drinks_logsum', 'Milk Shake_logsum', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_18_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_18_Pcomb',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_18_logsum', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Fried Chicken_logsum', 'Drinks_logsum', 'Milk Shake_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_19_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_19_Pcomb',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_19_logsum', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Fried Chicken_logsum', 'Drinks_logsum', 'Milk Shake_logsum', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'Alt_20_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_20_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_20_logsum', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Rice_logsum', 'Ice cream & Sweets_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'Alt_21_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_21_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_21_logsum', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Rice_logsum', 'Ice cream & Sweets_logsum', 'sero', 'sero',
     'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Alt_22_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'Alt_22_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_22_logsum', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Rice_logsum', 'Ice cream & Sweets_logsum', 'sero', 'sero',
     'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Alt_23_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'Alt_23_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_23_logsum', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Rice_logsum', 'Ice cream & Sweets_logsum', 'sero', 'sero',
     'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_24_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Alt_24_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_24_logsum', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Rice_logsum', 'Ice cream & Sweets_logsum', 'sero', 'sero',
     'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_25_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Alt_25_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_25_logsum', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Rice_logsum', 'Ice cream & Sweets_logsum', 'sero', 'sero',
     'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_26_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_26_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_26_logsum', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Snacks_logsum', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_27_PriceC', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_27_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_27_logsum', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Snacks_logsum', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_28_PriceC', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_28_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_28_logsum', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Snacks_logsum', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_29_PriceC', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_29_Pcomb', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_29_logsum', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Burger_logsum',
     'Fried Chicken_logsum', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_30_PriceC', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_30_Pcomb', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_30_logsum', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Burger_logsum',
     'Fried Chicken_logsum', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_31_PriceC',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_31_Pcomb', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_31_logsum', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Burger_logsum',
     'Fried Chicken_logsum', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_32_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_32_Pcomb',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_32_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Burger_logsum',
     'Fried Chicken_logsum', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_33_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_33_Pcomb',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_33_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Burger_logsum',
     'Fried Chicken_logsum'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'Alt_34_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_34_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_34_logsum',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Burger_logsum', 'Fried Chicken_logsum'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'Alt_35_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_35_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_35_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Burger_logsum',
     'Fried Chicken_logsum'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Alt_36_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'Alt_36_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_36_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Burger_logsum',
     'Fried Chicken_logsum'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Alt_37_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'Alt_37_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_37_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Burger_logsum',
     'Fried Chicken_logsum'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_38_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Alt_38_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_38_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Burger_logsum',
     'Fried Chicken_logsum'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_39_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_39_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_40_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_40_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_41_PriceC', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_41_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_42_PriceC', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_42_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_43_PriceC', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_43_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_44_PriceC', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_44_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_45_PriceC',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_45_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_46_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_46_logsum',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_47_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_47_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'Alt_48_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_48_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'Alt_49_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_49_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Alt_50_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_50_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Alt_51_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_51_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_52_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_52_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_53_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_53_logsum', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero']]

Gamma_List_logsum = [
    ['uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast',
     'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast',
     'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast',
     'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast',
     'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast',
     'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast',
     'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast',
     'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast',
     'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast',
     'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast',
     'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30',
     'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone',
     'Family'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone', 'Family', 'sero', 'sero', 'sero', 'sero',
     'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Below_30', 'Breakfast', 'Meal', 'Alone',
     'Family']]

Utility_list_logsum = [
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_1_PriceC', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_1_Pcomb', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_2_PriceC', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_2_Pcomb', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_3_PriceC',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_3_Pcomb', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_4_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_4_Pcomb',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_5_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_5_Pcomb',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'Alt_6_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_6_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'Alt_7_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_7_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Alt_8_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'Alt_8_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Alt_9_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'Alt_9_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_10_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Alt_10_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_11_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Alt_11_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_12_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_12_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_13_PriceC', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_13_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_14_PriceC', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_14_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_15_PriceC', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_15_Pcomb', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_16_PriceC', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_16_Pcomb', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_17_PriceC',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_17_Pcomb', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_18_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_18_Pcomb',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_19_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_19_Pcomb',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'Alt_20_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_20_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'Alt_21_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_21_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Alt_22_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'Alt_22_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Alt_23_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'Alt_23_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_24_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Alt_24_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_25_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Alt_25_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_26_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_26_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_27_PriceC', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_27_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_28_PriceC', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_28_Pcomb', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_29_PriceC', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_29_Pcomb', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_30_PriceC', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_30_Pcomb', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_31_PriceC',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_31_Pcomb', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_32_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_32_Pcomb',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_33_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_33_Pcomb',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'Alt_34_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_34_Pcomb', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'Alt_35_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_35_Pcomb', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Alt_36_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'Alt_36_Pcomb', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Alt_37_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'Alt_37_Pcomb', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_38_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Alt_38_Pcomb'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_39_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_40_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_41_PriceC', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_42_PriceC', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_43_PriceC', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_44_PriceC', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_45_PriceC',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_46_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'Alt_47_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'Alt_48_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'Alt_49_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'Alt_50_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'Alt_51_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_52_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero'],
    ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'Alt_53_PriceC', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
     'sero', 'sero', 'sero', 'sero']]

nvarm_logsum   = len(Utility_list_logsum[0])  # number of variables in baseline utility
nvargam_logsum = len(Gamma_List_logsum[0])  # number of variables in translation

nvarm = len(Utility_Spec[0])  # number of variables in baseline utility
nvargam = len(Gamma_List[0])  # number of variables in translation

Logsum_total_param = nvarm_logsum + nvargam_logsum + 1
Model_total_parm = nvarm + nvargam + 1

ivg_logsum = list(itertools.chain.from_iterable(Gamma_List_logsum))
ivm_logsum = list(itertools.chain.from_iterable(Utility_list_logsum))

ivg = list(itertools.chain.from_iterable(Gamma_List))
ivm = list(itertools.chain.from_iterable(Utility_Spec))

Total_products = 53
nc = Total_products
Consumption_col = [f'Alt_{iprod + 1}_Consumption' for iprod in range(Total_products)]
Avalability_col = [f'Alt_{iprod + 1}_Avalability' for iprod in range(Total_products)]
Price_col       = [f'Alt_{iprod + 1}_PriceC' for iprod in range(Total_products)]

Price_data = pd.read_csv(price_file_name)
Filter_data = pd.read_csv(filter_file_name)

Demographic_Variable = ['Below_30', 'Above_30','Breakfast','Meal','Snack', 'Alone', 'Family', 'Friend']
Demographic_Variable_aggregate = ['Age','Time of day','Party size']

st.set_page_config(layout='wide', page_title="RBI Dashboard")

def Get_Exp_XB(dta, parm):
    smallb = parm[0:nvarm_logsum]
    xgam = parm[nvarm_logsum:nvarm_logsum + nvargam_logsum]
    xsigm = parm[nvarm_logsum + nvargam_logsum:nvarm_logsum + nvargam_logsum + 1]

    v2 = (np.kron(np.ones((nc, 1)), smallb)) * (dta.loc[:, ivm_logsum].values.T)
    u2 = (np.kron(np.ones((nc, 1)), xgam)) * (dta.loc[:, ivg_logsum].values.T)

    if xsigm <= 0:
        xsigm = 1

    v = np.empty(shape=(nobs, nc), dtype=float)
    u = np.empty(shape=(nobs, nc), dtype=float)
    for i in range(nc):
        j = i + 1
        v[:, i] = np.sum(v2[(j - 1) * nvarm_logsum:(j * nvarm_logsum), :], axis=col_wise)
        u[:, i] = np.sum(u2[(j - 1) * nvargam_logsum:(j * nvargam_logsum), :], axis=col_wise)

    del v2, u2

    a = np.ones((nobs, nc))  # a is (1-Alpha)

    f = np.exp(u)
    b = (dta.loc[:, flagchm].values > 0).astype(float)
    m = np.sum(b, axis=row_wise)
    try:
        temp1 = m.shape[1]
    except:
        m = m[:, np.newaxis]

    c = (a * b) / (dta.loc[:, flagchm].values + f)
    c = c / (dta.loc[:, flagprcm].values)
    np.place(c, b == 0, 1)  # Replacing 0's with 1s
    e = (1 / c) * b
    d = np.sum(e, axis=row_wise)
    c = np.prod(c, axis=row_wise)

    v = v - a * np.log((dta.loc[:, flagchm].values + f) / f) - np.log(dta.loc[:, flagprcm].values)
    ut = v / xsigm
    p1 = np.exp(ut)
    p2 = (p1 * dta.loc[:, flagavm].values)
    return p2


def Get_Logsum(Data, Param):
    All_XB = Get_Exp_XB(Data, Param)
    Product_col_name = []
    Product_Wise_logsum = np.zeros((nobs, nc))
    for curr_item, substitute_list in Category_Wise_Collection.items():
        substitute_list = [x - 1 for x in substitute_list]
        temp = np.sum(All_XB[:, substitute_list], axis=row_wise)
        temp[temp == 0] = 0.00001
        Product_Wise_logsum[:, curr_item - 1] = temp
        Product_col_name.append(f'Alt_{curr_item}_logsum')

    Category_col_name = []
    Category_wise_logsum = np.zeros((nobs, len(Category_product)))
    count = -1
    for curr_category, product_list in Category_product.items():
        product_list = [x - 1 for x in product_list]
        count += 1
        temp = np.sum(All_XB[:, product_list], axis=row_wise)
        temp[temp == 0] = 0.00001
        Category_wise_logsum[:, count] = temp
        Category_col_name.append(f'{curr_category}_logsum')

    Both_logsum = np.hstack((Product_Wise_logsum, Category_wise_logsum))
    Both_logsum = np.log(Both_logsum)
    Logsum_df = pd.DataFrame(Both_logsum, index=range(nobs), columns=Product_col_name + Category_col_name)
    Logsum_df = Logsum_df[[(f'Alt_{curr_item}_logsum') for curr_item in range(1, nc + 1)] + Category_col_name]
    return Logsum_df


def Prediction(Parm):
    x = Parm
    if Num_Threads > 1:
        data_list = [iter for iter in range(Num_Threads)]
        pool = Pool(processes=Num_Threads)
        prod_x = partial(lpr_Main, parm=x)
        result_list = pool.map(prod_x, data_list)

        pool.close()
        pool.join()
        a_temp = list(itertools.chain.from_iterable(result_list))
        atemp_array = np.asarray(a_temp)
    else:
        atemp_array = lpr_Main(0, x)

    return atemp_array


def lpr_Main(iter, parm):
    if (iter == 0 and iter + 1 == Num_Threads):
        st_iter = int(0)
        end_iter = int(nobs - 1)
    elif (iter == 0 and iter + 1 != Num_Threads):
        st_iter = int(Data_Split[iter, 0])
        end_iter = int(Data_Split[iter, 1])
    else:
        st_iter = int(Data_Split[iter, 0] - 1) + 1
        if (iter + 1 < Num_Threads):
            end_iter = int(Data_Split[iter, 1])
        else:
            end_iter = int(nobs - 1)

    nobs_num = int(end_iter - st_iter + 1)

    smallb = parm[0:nvarm]
    xgam = parm[nvarm:nvarm + nvargam]
    xsigm = parm[nvarm + nvargam:nvarm + nvargam + 1]

    v2 = (np.kron(np.ones((nc, 1)), smallb)) * (Main_data.loc[st_iter:end_iter, ivm].values.T)
    u2 = (np.kron(np.ones((nc, 1)), xgam)) * (Main_data.loc[st_iter:end_iter, ivg].values.T)

    if xsigm <= 0:
        xsigm = 1

    v = np.empty(shape=(nobs_num, nc), dtype=float)
    u = np.empty(shape=(nobs_num, nc), dtype=float)
    for i in range(nc):
        j = i + 1
        v[:, i] = np.sum(v2[(j - 1) * nvarm:(j * nvarm), :], axis=col_wise)
        u[:, i] = np.sum(u2[(j - 1) * nvargam:(j * nvargam), :], axis=col_wise)

    del v2, u2

    a = np.ones((nobs_num, nc))  # a is (1-Alpha)

    f1 = np.exp(u)
    v = np.exp(v)
    v = np.kron(v, np.ones((nrep, 1)))
    alts = np.arange(1, nc + 1, 1)

    if iter == 0:
        draw_st_row = int(st_iter * nrep)
        draw_end_row = int(nobs_num * nrep)
    else:
        draw_st_row = int(st_iter * nrep)
        draw_end_row = int(nobs_num * nrep) + draw_st_row
    logistic_draw = -np.log(-np.log(Halton_draws[draw_st_row:draw_end_row, :]))

    v = v * np.exp(logistic_draw)

    expenditure = Main_data.loc[st_iter:end_iter, 'Total_Consumption'].values
    Avalability = Main_data.loc[st_iter:end_iter, Avalability_col].values

    All_consumption = np.zeros((nobs_num*nrep, nc))
    count = -1
    pbar = st.empty()
    for i in range(nobs_num):
        variable_output = f'Progress : {round(((i+1)/nobs_num)*100)}%'
        html_str = f"""<style>p.a{{font-size:26px;font-family:sans serif;color:blue; text-align:center}}</style><p class="a">{variable_output}</p>"""
        pbar.markdown(html_str, unsafe_allow_html=True)
        for r_rep in range(nrep):
            count += 1
            fc = np.zeros((1, nc))
            vqr = np.zeros((5, nc))
            vqr[0, :] = alts
            vqr[1, :] = v[count, :]
            vqr[2, :] = np.ones(((nc)))
            vqr[3, :] = f1[i, :]
            vqr[4, :] = Scenario_price

            vqr = vqr.T
            curr_budget = expenditure[i]
            budget_enough = (Scenario_price.T <= curr_budget).astype(int)
            budget_enough = np.sum(budget_enough)
            if budget_enough > 0:
                # Removing items with price higher than the budget or not available
                vqr = vqr[(Avalability[i, :] == 1), :]
                vqr = vqr[(vqr[:, 4] <= curr_budget), :]
                vqr = sorted(vqr, key=lambda x: x[1], reverse=True)
                vqr = np.array(vqr)
                vqr = vqr.T

                nc_local = vqr.shape[1]
                m = 1
                k = -1
                N = (vqr[3, 0] * vqr[2, 0]) * (pow(vqr[1, 0], (1 / a[i, 0])))
                if Expenditure_Actual == 1:
                    D = expenditure[i] + (vqr[3, 0] * vqr[2, 0])
                else:
                    D = 1 + (vqr[3, 0] * vqr[2, 0])
                Lagrang_lambda = pow((D / N), -1)

                if nc_local > 1:
                    if (vqr[1, 1] < Lagrang_lambda):
                        predicted_proportion = ((vqr[1, 0] / Lagrang_lambda) * vqr[3, 0]) - vqr[3, 0]
                        if Expenditure_Actual != 1:
                            item_price = vqr[4, 0]
                            implied_prop, actual_price_spend = Round_proportion(predicted_proportion, expenditure[i],
                                                                                item_price)
                            fc[0, int(vqr[0, 0]) - 1] = implied_prop
                        else:
                            fc[0, int(vqr[0, 0]) - 1] = predicted_proportion

                    else:
                        while k != m:
                            m += 1
                            if (m == nc_local):
                                temp = ((vqr[1, :] / Lagrang_lambda) - np.ones(nc_local)) * (vqr[3, :])
                                temp = temp / np.sum(temp)
                                for ialt_index in range(temp.shape[0]):
                                    fc[0, int(vqr[0, ialt_index]) - 1] = temp[ialt_index]
                                k = m
                            elif (m < nc_local):
                                N = N + (vqr[3, m - 1] * vqr[2, m - 1] * vqr[1, m - 1])
                                D = D + (vqr[3, m - 1] * vqr[2, m - 1])
                                Lagrang_lambda = (N / D)
                                if (vqr[1, m] < Lagrang_lambda):
                                    temp = ((vqr[1, 0:m] / Lagrang_lambda) - np.ones((m))) * (vqr[3, 0:m])
                                    for ialt_index in range(temp.shape[0]):
                                        fc[0, int(vqr[0, ialt_index]) - 1] = temp[ialt_index]
                                    k = m
                else:
                    if Expenditure_Actual == 1:
                        fc[0, int(vqr[0, 0]) - 1] = expenditure[i]
                    else:
                        fc[0, int(vqr[0, 0]) - 1] = 1

                if np.sum(fc) != 1:
                    fc = fc / np.sum(fc)
                #temp_consumption[r_rep, :] = fc[0,:]

        #Curr_avg_consumption = np.mean(temp_consumption, axis=col_wise)
            All_consumption[count, :] = fc[0,:]

    return All_consumption


def Round_proportion(predicted_proportion, budget, item_price):
    implied_total_price = round(predicted_proportion * budget)
    if (implied_total_price < item_price):
        Current_spend = item_price
    else:
        Current_spend = round(implied_total_price)
        if Current_spend > budget:
            Current_spend = budget
        item_quantity = round(Current_spend / item_price)
        if item_quantity * item_price > budget:
            Current_spend = (item_quantity - 1) * item_price

    implied_proportion = Current_spend / budget
    return implied_proportion, Current_spend


def Prepare_Specification(Item_Config,Demographic_Available):
    global nobs, nind, flagchm, flagprcm, flagavm, Data_Split, Num_Threads
    global Halton_draws, Main_data,Scenario_price


    dp_progress = st.empty()
    message_formatted1 = '<p style="font-size:26px;font-family:sans serif;color:blue; text-align:center">Reading files</p>'
    message_formatted2 = '<p style="font-size:26px;font-family:sans serif;color:blue; text-align:center">Creating choice set based variables</p>'
    message_formatted3 = '<p style="font-size:26px;font-family:sans serif;color:blue; text-align:center">Creating subtitution & complementarity variables</p>'
    message_formatted4 = '<p style="font-size:26px;font-family:sans serif;color:blue; text-align:center">Predicting expenditure across items</p>'

    Num_Threads = 300  # No. of threads to be used for parallel processing
    if Windows_OS == 1:
        Num_Threads = 1

    dp_progress.markdown(message_formatted1, unsafe_allow_html=True)
    Main_data      = pd.read_csv(Data_file_name)

    Main_data = Main_data[Main_data['Total_Consumption'] > 0]  # Dropping data with no choices
    if np.sum(Demographic_Available) != len(Demographic_Variable):
        conditions_list = [[(Demographic_Variable_aggregate[0], Demographic_Available[0]),
                            (Demographic_Variable_aggregate[0], Demographic_Available[1] + 1 * Demographic_Available[1])],
                           [(Demographic_Variable_aggregate[1], Demographic_Available[2]),
                            (Demographic_Variable_aggregate[1], Demographic_Available[3] + 1 * Demographic_Available[3]),
                            (Demographic_Variable_aggregate[1], Demographic_Available[4] + 2 * Demographic_Available[4])],
                           [(Demographic_Variable_aggregate[2], Demographic_Available[5]),
                            (Demographic_Variable_aggregate[2], Demographic_Available[6] + 1 * Demographic_Available[6]),
                            (Demographic_Variable_aggregate[2], Demographic_Available[7] + 2 * Demographic_Available[7])]]

        masks = [Main_data.apply(lambda row: any(row[col] == value for col, value in condition), axis=1) for condition
                 in conditions_list]
        final_mask = pd.concat(masks, axis=1).all(axis=1)
        Main_data = Main_data[final_mask]

    if len(Main_data) > 0:
        Main_data.reset_index(inplace=True)
        Parameter_data = pd.read_csv(Parameter_file)

        Logsum_betas = Parameter_data['Logsum'].values
        Utility_betas = Parameter_data['Utility'].values
        try:
            temp = Logsum_betas.shape[1]
        except:
            Logsum_betas = Logsum_betas[:, np.newaxis]

        try:
            temp = Utility_betas.shape[1]
        except:
            Utility_betas = Utility_betas[:, np.newaxis]

        Main_data[Avalability_col] = Item_Config[:, 1]
        Main_data[Price_col] = Item_Config[:, 2]

        Scenario_price = Item_Config[:, 2]

        if Consumption_Normalize == 1:
            subset_df = Main_data[Consumption_col]
            row_sums = subset_df.sum(axis=row_wise)
            normalized_df = subset_df.div(row_sums, axis=col_wise)
            Main_data[Consumption_col] = normalized_df
        else:
            Main_data[Consumption_col] = Main_data[Consumption_col] / 100
            Main_data['Alt_51_Consumption'] = Main_data['Alt_51_Consumption'] / 10
            Main_data['Alt_53_Consumption'] = Main_data['Alt_53_Consumption'] / 10

        Main_data['uno'] = 1  # Creating a column of ones
        Main_data['sero'] = 0  # Creating a column of zeros
        _price = 0  # 1 if there is price variation across goods, 0 otherwise
        nc = len(Consumption_col)  # Number of alternatives

        nind = int(Main_data.shape[0])
        nobs = nind

        if Generate_Draws == 1:
            Halton_draws = HaltonSequence((nobs * nrep)+100, Total_products)
            Halton_draws = Halton_draws[0:int(nobs * nrep), 0:nc]

        Logsum_betas = Logsum_betas[0:Logsum_total_param]
        Utility_betas = Utility_betas[0:Model_total_parm]

        flagchm = Consumption_col
        flagavm = Avalability_col
        flagprcm = ['uno'] * nc

        Data_Split = np.zeros((Num_Threads, 2))
        for i in range(1, Num_Threads + 1):
            Data_Split[i - 1, 0] = int(ceil((i - 1) * ((nind - 1) / Num_Threads)) + 1)
            if (i != Num_Threads):
                Data_Split[i - 1, 1] = int(ceil(i * ((nind - 1) / Num_Threads)))
            else:
                Data_Split[i - 1, 1] = nind
        Data_Split = Data_Split - 1

        dp_progress.markdown(message_formatted2, unsafe_allow_html=True)
        All_store = np.zeros((nobs, len(Ala_carte)))
        for items in Mapping_dict:
            curr = [0] * len(Combo_Mapping)
            if len(items) > 1:
                for icr in range(1, len(items)):
                    curr[items[icr] - 39] = 1
            curr = (np.array(curr)[:, np.newaxis]).T
            curr = np.tile(curr, (nobs, 1))
            curr_ava_data = Main_data[Avalability_col[len(Ala_carte):]].values
            Mult_values = np.multiply(curr, curr_ava_data)
            Mult_values = np.sum(Mult_values, axis=row_wise)
            Mult_values = (Mult_values > 0).astype(int)
            All_store[:, items[0] - 1] = Mult_values

        All_store_df = pd.DataFrame(All_store, index=range(All_store.shape[0]),
                                    columns=[f'Alt_{i + 1}_Pcomb' for i in range(len(Ala_carte))])
        Main_data = pd.concat([Main_data, All_store_df], axis=row_wise)

        dp_progress.markdown(message_formatted3, unsafe_allow_html=True)
        Main_data.loc[:, Price_col] = Main_data.loc[:, Price_col].replace(0, 1)
        Main_data[Price_col] = -1*(np.log(Main_data[Price_col].values))
        Curr_logsum_df = Get_Logsum(Main_data, Logsum_betas)
        Main_data = pd.concat([Main_data, Curr_logsum_df], axis=row_wise)

        dp_progress.markdown(message_formatted4, unsafe_allow_html=True)
        Curr_probability = Prediction(Utility_betas)
        Output_metrix = np.zeros((nc, 3))
        for i in range(nc):
            if Item_Config[i, 1] == 1:
                temp = Curr_probability[:, i]
                Share = np.sum(temp > 0) / temp.shape[0]
                percentage_share = round(Share * 100, 2)
                Output_metrix[i, 0] = percentage_share
                Output_metrix[i, 1] = round(percentage_share*Item_Config[i, 2], 0)

        Price_Share_normalized = Output_metrix[:,1] / np.sum(Output_metrix[:,1])
        Price_Share_normalized = np.round(Price_Share_normalized * 100,2)
        Output_metrix[:,2] = Price_Share_normalized
        used_sample_size = Main_data['ID'].unique().shape[0]
    else:
        Output_metrix = np.zeros((Total_products, 3))
        used_sample_size = 0

    return (Output_metrix,used_sample_size)

def write_df_to_csv_with_gap(df, csv_file):
    gap_df = pd.DataFrame({'': [''] * len(df)})
    df_with_gap = pd.concat([df, gap_df], axis=1)
    df_with_gap.to_csv(csv_file, index=False)

def run():
    st.session_state.run = True
def main():
    table1, table2, table3 = st.columns([2.25,0.7,0.6])

    if "Item_table" not in st.session_state:
        col_names = ['Product Number', 'Product Name', 'Price Range (￥)', 'Current Price (￥)',
                     'Availability', 'Product Share (%)', 'Revenue (￥)', 'Revenue Share (%)']

        # Create a DataFrame with 53 rows and 4 columns
        df1 = pd.DataFrame(index=range(53), columns=col_names)

        # Pre-populate the 'Name' column
        df1['Product Number'] = Price_data['Product_ID'].values
        df1['Product Name'] = Price_data['Label'].values
        df1['Current Price (￥)'] = Price_data['P2'].values
        df1['Price Range (￥)'] = Price_data['Price_range'].values

        df1['Availability'] = True
        df1.fillna("", inplace=True)
        st.session_state["Item_table"] = df1

    with table1:
        st.write('<p style="font-size:26px;font-family:sans serif;color:red; text-align:center">Menu items</p>', unsafe_allow_html=True)
        edited_df1 = st.data_editor(st.session_state["Item_table"], hide_index=True, use_container_width=False,
                                    disabled=['Product Number', 'Product Name', 'Price Range (￥)', 'Product Share (%)',
                                              'Revenue (￥)', 'Revenue Share (%)'],
                                    column_config={"Availability": st.column_config.CheckboxColumn(default=True)})

    if "Segment_table" not in st.session_state:
        # Create the second table with 8 rows and 3 columns
        col_names2 = ['Attribute', 'Category', 'Select']
        data2 = [['Age', 'Below_30', True],
                 ['Age', 'Above_30', True],
                 ['Time of day', 'Breakfast', True],
                 ['Time of day', 'Meal', True],
                 ['Time of day', 'Snack', True],
                 ['Party size', 'Alone', True],
                 ['Party size', 'Family', True],
                 ['Party size', 'Friend', True]]
        df2 = pd.DataFrame(data2, columns=col_names2)
        st.session_state["Segment_table"] = df2

    if "Sample size" not in st.session_state:
        temp_data = [int(Starting_sample_size)]
        df3 = pd.DataFrame(temp_data, columns=['Sample size'])
        st.session_state["Sample size"] = df3


    with table2:
        st.write('<p style="font-size:26px;font-family:sans serif;color:red;margin-left:21%">Filter</p>', unsafe_allow_html=True)
        edited_df2 = st.data_editor(st.session_state["Segment_table"], hide_index=True, use_container_width=False,
                                    disabled=['Attribute','Category'],
                                    column_config={"Select": st.column_config.CheckboxColumn(default=True)})
        edited_df3 = st.data_editor(st.session_state["Sample size"], hide_index=True, use_container_width=False,disabled=['Sample size'],)

    with table3:
        with st.expander("⚙️ Instructions ", expanded=False):
            st.write(
                """    
            - In the Menu item table, modify the columns "Availability" and "Current Price (￥)" to simulate a scenario
            - Check all the filter options to simulate for whole sample
            - (Un)check the appropriate attributes in the Filter table to simulate for the selected segments
            - Revenue (￥) is reported per 100 individuals
              """
            )
    if 'run' not in st.session_state:
        st.session_state.run = False

    if st.button("Simulate Scenario", type='primary',on_click=run, disabled=st.session_state.run):
        Avalability_checked = edited_df1['Availability']
        Avalability_numeric = [int(1) if checked == True else 0 for checked in Avalability_checked]
        Price_col = edited_df1['Current Price (￥)'].astype(float)
        Product_ID = edited_df1['Product Number'].astype(int)

        Check_Price_Range = (np.logical_and(Price_data['P1'].values <= Price_col, Price_col <= Price_data['P3'].values)).astype(int)
        Check_Price_Range = Check_Price_Range * np.array(Avalability_numeric)

        if np.any(Check_Price_Range[np.array(Avalability_numeric)==1] == 0):
            Error_message1 = '<p style="font-size:26px;font-family:sans serif;color:blue; text-align:left">One or more prices (in column Current Price (￥)) of table Menu items are not within the price range</p>'
            st.markdown(Error_message1, unsafe_allow_html=True)
            time.sleep(10)
            st.session_state.run = False
            st.experimental_rerun()
        else:
            Joint_metrix = np.zeros((len(Avalability_numeric), 3))
            Joint_metrix[:, 0] = Product_ID
            Joint_metrix[:, 1] = Avalability_numeric
            Joint_metrix[:, 2] = Price_col

            Joint_metrix = sorted(Joint_metrix, key=lambda x: x[0])
            Joint_metrix = np.array(Joint_metrix)

            Filter_selected = edited_df2['Select']
            Filter_selected = [int(1) if checked == True else 0 for checked in Filter_selected]

            Filter_metrix = deepcopy(edited_df2)
            Filter_metrix.drop(['Select'],axis=1,inplace=True)
            Filter_metrix['Select'] = Filter_selected


            matched_rows = Filter_metrix[Filter_metrix['Category'].isin(Demographic_Variable)]
            Segment_Indicator = matched_rows['Select'].values

            if np.sum(Segment_Indicator) == 0:
                Error_message2 = '<p style="font-size:26px;font-family:sans serif;color:blue; text-align:left">Please select at least one category in Filter table</p>'
                st.markdown(Error_message2, unsafe_allow_html=True)
                time.sleep(10)
                st.session_state.run = False
                st.experimental_rerun()
            else:
                Scenario_Results, New_sample_size = Prepare_Specification(Joint_metrix,Segment_Indicator)

                LRresult = deepcopy(edited_df1)
                LRFilter = deepcopy(edited_df2)
                LRSample = pd.DataFrame([New_sample_size], columns=['Sample size'])


                LRresult['Product Number'] = Joint_metrix[:, 0]
                LRresult['Product Name']   = Price_data['Label'].values
                LRresult['Current Price (￥)']  = Joint_metrix[:, 2]
                LRresult['Price Range (￥)']    = Price_data['Price_range'].values
                LRresult['Availability']   = (Joint_metrix[:, 1] == 1).astype(bool)
                LRresult['Product Share (%)'] = Scenario_Results[:,0]
                LRresult['Revenue (￥)']       = Scenario_Results[:,1]
                LRresult['Revenue Share (%)'] = Scenario_Results[:,2]

                LRFilter['Attribute'] = Filter_data['Attribute'].values
                LRFilter['Category']  = Filter_data['Category'].values
                LRFilter['Select']    = (Filter_metrix['Select'].values == 1).astype(bool)


                edited_df1.update(LRresult)
                edited_df2.update(LRFilter)
                edited_df3.update(LRSample)

                st.session_state["Item_table"]    = edited_df1
                st.session_state["Segment_table"] = edited_df2
                st.session_state["Sample size"]   = edited_df3

                st.session_state.run = False
                st.experimental_rerun()





if __name__ == "__main__":
    main()


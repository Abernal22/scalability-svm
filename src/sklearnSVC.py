import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import generate_data as gd


def gen_files():
    gd.make_classification(10, 500, save_to_file= True)
    gd.make_classification(10, 1000, save_to_file= True)
    gd.make_classification(10, 5000, save_to_file= True)
    gd.make_classification(10, 10000, save_to_file= True)
    gd.make_classification(10, 100000, save_to_file= True)

    gd.make_classification(50, 500, save_to_file= True)
    gd.make_classification(50, 1000, save_to_file= True)
    gd.make_classification(50, 5000, save_to_file= True)
    gd.make_classification(50, 10000, save_to_file= True)
    gd.make_classification(50, 100000, save_to_file= True)

    gd.make_classification(100, 500, save_to_file= True)
    gd.make_classification(100, 1000, save_to_file= True)
    gd.make_classification(100, 5000, save_to_file= True)
    gd.make_classification(100, 10000, save_to_file= True)
    gd.make_classification(100, 100000, save_to_file= True)

    gd.make_classification(500, 500, save_to_file= True)
    gd.make_classification(500, 1000, save_to_file= True)
    gd.make_classification(500, 5000, save_to_file= True)
    gd.make_classification(500, 10000, save_to_file= True)
    gd.make_classification(500, 100000, save_to_file= True)

    gd.make_classification(1000, 500, save_to_file= True)
    gd.make_classification(1000, 1000, save_to_file= True)
    gd.make_classification(1000, 5000, save_to_file= True)
    gd.make_classification(1000, 10000, save_to_file= True)
    gd.make_classification(1000, 100000, save_to_file= True)


gen_files()    


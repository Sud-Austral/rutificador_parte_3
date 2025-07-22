import pandas as pd
from tqdm import tqdm
import pickle
from fuzzywuzzy import fuzz, process
import numpy as np
import networkx as nx
from fuzzywuzzy import fuzz
from tqdm import tqdm
from collections import defaultdict
from unidecode import unidecode

tqdm.pandas()  # habilita barras en apply

def GLOBAL():
    print("Hola")
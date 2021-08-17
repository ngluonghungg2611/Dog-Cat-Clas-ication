import pandas as pd

from A_libraries import *
filenames = os.listdir("./train")

categories = []
for f_name in filenames:
    category = f_name.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame(
    {
        'filename': filenames,
        'category': categories
    }
)
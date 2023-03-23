import pandas as pd
import numpy as np
from ..utility.pre_processing import clean

def IE_brand(brand):
  path = "data/Scraped_Car_Review_" + brand + ".csv"
  df = pd.read_csv(path,delimiter=',', nrows = 100)
  df['Review_clean'] = df['Review'].apply(clean)
  df['Review_clean'][2]
  reviews = df['Review_clean'][0:5]
  reviews = np.array(reviews)
  df2 = df["Rating"].mean()
  return (df2/5) * 10

def get_dataframe_head(brand):
  path = "data/Scraped_Car_Review_" + brand + ".csv"
  df = pd.read_csv(path,delimiter=',', nrows = 5)
  
  return df.to_dict()


def get_dataframe_sentence(brand):
  # path = "data/Scraped_Car_Review_" + brand + ".csv"
  # df = pd.read_csv(path,delimiter=',', nrows = 5)
  # df_dict = df.to_dict()
  # review = df_dict["Review"]
  # sentence = review[1]
  sentence = f"I love the {brand} engines sound and specially the mileage it gives.would highly recommend this brand"
  return sentence






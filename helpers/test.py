# from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'
# import argparse
import cv2
# import os
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import csv
# import nltk
import re
# from nltk.tokenize import word_tokenize
# from nltk.corpus import wordnet 



image=cv2.imread('bill3.png',0)
#convert it into text
text=(pytesseract.image_to_string(image)).lower()
# print(text)
match=re.findall(r'\d+[/.-]\d+[/.-]\d+', text)

st=" "
st=st.join(match)
print(st)
#lets find the price of the category.
price=re.findall(r'[\$\£\€](\d+(?:\.\d{1,2})?)',text)
price = list(map(float,price)) 
print(max(price))
x=max(price)  
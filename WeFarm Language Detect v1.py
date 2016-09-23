# -*- coding: utf-8 -*-
"""
WeFarm Data Language Detection First Pass

@author: mike.a.taylor@ accenture.com

Todos
1. detet langauge based on conversation, not just on the question (reasonable to assume that
   Q&A are the same language, any builingual convos?)
2. improve approach with some deterministic rules (e.g. certain words = Englsh)
3. collapse all northern european langauges to english?
4. use other detection libraries, creating an ensemble model?
5. other clever things...

"""

##############################################################################
# Library Imports
##############################################################################

### Standard data manipulation libraries
import pandas as pd
import numpy as np

### Language detection library
from langdetect import detect

##############################################################################
# Main Code
##############################################################################

### Import Data
sampleData = pd.read_csv(r"threaded-data.csv")#, encoding = "ISO-8859-1")

print("DataKind Text Analytics Beta")
print("--------------------------------------------------")
print("File Length - ", len(sampleData))
print("")

### Prep DataFram
sampleData["Detected Language"] = "?"



### Language detect
for i in range(0,len(sampleData)):
    msgBody = sampleData["body"][i]
    try:
        tmpLang = detect(msgBody)
    except:
        tmpLang = "?"
    sampleData.set_value(i,"Detected_Language", tmpLang)
    
### Import the Data
corpusFull = np.asarray(sampleData['body'])

### Save DF to csv
sampleData.to_csv("threaded-data-with-langdetect.csv")




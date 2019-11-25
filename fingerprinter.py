
from scapy.all import *
import re
import os
import numpy as np
import pandas as pd
#from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import scapy

input_pcap = sys.argv[1]


def get_class_value(filename):
    if re.search("^wiki", filename):
        class_value = 'Wikipedia'
    elif re.search("^autolab", filename):
        class_value = 'AutoLab'
    elif re.search("^craigslist", filename):
        class_value = 'Craigslist'
    elif re.search("^torproject", filename):
        class_value = 'TorProject'
    elif re.search("^neverssl", filename):
        class_value = 'NeverSSL'
    elif re.search("^canvas", filename):
        class_value = 'Canvas'
    elif re.search("^bing", filename):
        class_value = 'Bing'
    return class_value

path = os.getcwd()+'/Training/'
features= pd.DataFrame(columns=['length'])
features.shape
label= pd.DataFrame(columns=['filename'])
label.shape

length = []
fname = []
for filename in os.listdir(path):
    pcap = rdpcap(path+filename)
    length.append(len(pcap))
    fname.append(get_class_value(filename))

features['length'] = length 
label['filename'] = fname

gnb = GaussianNB()
gnb.fit(features, label)


test_features= pd.DataFrame(columns=['length'])
test_features.shape
test_label= pd.DataFrame(columns=['filename'])
test_label.shape
test_length=[]
test_fname=[]
pcap = rdpcap( os.getcwd()+"/Test/" +input_pcap)
test_length.append(len(pcap))
test_fname.append(input_pcap)

test_features['length'] = test_length 
test_label['filename'] = test_fname

predict_probs = gnb.predict_proba(test_features)
predict = []


preds = gnb.predict_proba(test_features)
preds_idx = np.argsort(preds, axis=1)[-7:]

final_pred_dict = {

}

for i,d in enumerate(test_features):
    for p in preds_idx[i]:
        final_pred_dict[gnb.classes_[p]] = preds[i][p]
        #print(gnb.classes_[p],"(",preds[i][p],")")
for i in sorted (final_pred_dict, key=final_pred_dict.get, reverse=True) :  
     print(i)
     print("") 


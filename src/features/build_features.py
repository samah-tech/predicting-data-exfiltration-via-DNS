#!python pip install tldextract
import tldextract
import pandas as pd
import re 
import tldextract
from collections import Counter
import tldextract
import numpy as np
#otal count of characters in FQDN
def FQDN (df):
  FQDN=[]
  for i in range(len(df)):
    dot_count= re.findall(r"[.]", df.iloc[:,0][i])
    FQDN.append(len(df.iloc[:,0][i] ) - len(dot_count))
  return FQDN

#Count of characters in subdomain
def subdomain_length(df):
  subdomain_length=[]
  for i in range(len(df)):
     ext = tldextract.extract( df.iloc[:,0][i])
     subdomain_ =ext.subdomain
     dot_count= re.findall(r"[.]", subdomain_)
     subdomain_length.append(len(subdomain_)-len(dot_count)) 
  return subdomain_length

#	Count of uppercase characters
def upper (df):
  upper=[]
  for i in range(len(df)):
     UP= re.findall(r"[A-Z]", df.iloc[:,0][i])
     upper.append(len(UP)) 
  return upper

#	Count of lowercase characters
def lower (df):
  lower=[]
  for i in range(len(df)):
     LOW= re.findall(r"[a-z]", df.iloc[:,0][i])
     lower.append(len(LOW)) 
  return lower

#Count of numerical characters
def numeric(df):
  numeric=[]
  for i in range(len(df)):
     num= re.findall(r"[0-9]", df.iloc[:,0][i])
     numeric.append(len(num)) 
  return numeric

#calculating the entropy 
def calcEntropy(df):
  entropy = []
  for i in range(len(df)):
    p, lens = Counter(df.iloc[:,0][i]), float(len(df.iloc[:,0][i]))
    entropy.append(-sum( count/lens * np.log2(count/lens) for count in p.values()))
  return entropy

#	Number of special characters
def special(df):
  special=[]
  for i in range(len(df)):
     spec_char= re.findall(r"[-%_=  .]", df.iloc[:,0][i])
     special.append(len(spec_char)) 
  return special

#calculating Number of labels
def labels(df):
  labels=[]
  for i in range(len(df)):
     label= re.findall(r"[.]", df.iloc[:,0][i])
     labels.append(len(label)+1) 
  return labels

#	Maximum label length
def labels_max(df):
  labels_max=[]
  for i in range(len(df)):
    result = df.iloc[:,0][i].split('.')
    labels_max.append(len(max(result,key=len))) 
  return labels_max

#Average label length
def labels_average(df):
  labels_average=[]
  sum=0
  for i in range(len(df)):
   result = df.iloc[:,0][i].split('.')
   for k in(result):
     sum+=len(k)
   labels_average.append(sum/(len(result)+1))
  return labels_average

#Longest meaningful word over domain length average
def longest_word(df):
  longest_word=[]
  for i in range(len(df)):
    string = df.iloc[:,0][i].split('.')
    longest_word.append(max(string,key=len))
  return longest_word

#Second level domain
def sld(df):
  sld=[]
  for i in range(len(df)):
     ext = tldextract.extract( df.iloc[:,0][i])
     sld.append(ext.domain)
  return sld 

#Length of domain and subdomain
def len_gth(df):
  len_gth=[]
  for i in range(len(df)):
     ext = tldextract.extract( df.iloc[:,0][i])
     len_gth.append(len(ext.domain)+len(ext.subdomain))
  return len_gth 

#subdomain
def subdomain(df):
  subdomain=[]
  for i in range(len(df)):
     ext = tldextract.extract( df.iloc[:,0][i])
     sub_D = ext.subdomain
     subdomain.append(1) if sub_D else subdomain.append(0)
  return subdomain

def pass_data(df):
 
  from sklearn.preprocessing import LabelEncoder 
  DataFrame = pd.DataFrame(FQDN(df),columns=['FQDN'])
  DataFrame['subdomain_length']=subdomain_length(df)
  DataFrame['upper']=upper(df)
  DataFrame['lower']=lower(df)
  DataFrame['numeric']=numeric(df)
  DataFrame['calcEntropy']=calcEntropy(df)
  DataFrame['special']=special(df)
  DataFrame['labels']=labels(df)
  DataFrame['labels_max']=labels_max(df)
  DataFrame['labels_average']=labels_average(df)
  DataFrame['longest_word']=longest_word(df)
  DataFrame['sld']=sld(df)
  DataFrame['len']=len_gth(df)
  DataFrame['subdomain']=subdomain(df)

  
  return(DataFrame)



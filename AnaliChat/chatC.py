import pandas as pd
import numpy as np
import datetime as dt
import re
import regex 
import emoji
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sbn  
import matplotlib.pyplot as plt


def startLine(s):
    #print(s)
    # Ex: '02/21/2021 11:27 a. m. - ... '
    patt = r'^([0-9]|1[0-2])(\/)([1-9]|1[0-9]|2[0-9]|3[0-1])(\/)([0-9][0-9]), ([0-9][0-9]+):([0-9][0-9]) - '
    res = re.match(patt, s)  #Check if each line match with the pattern
    if res:
        return True
    return False
  
# to find the Persons of the chat
def findPerson(s):
    
    patts = [
        r'([\w]+):',                                    # Name
        r'([\w]+[\s]+[\(]+[\w]+[\)]+):',      # Name (Nickname)
        r'([\w]+[\s]+[\w]+):',                    # Name + Last Name
        r'([\w]+[\s]+[\w]+[\s]+)[\u263a-\U0001f999]:', # Name1 + Name2 + Emoji
        r'([\w]+[\s]+[\w]+[\s]+[\w]+):',    # Name 1 + Name 2 + Las Name
        r'([+]\d{2} \d{3} \d{3} \d{3}):',     # Phone Number
        r'([\w]+)[\u263a-\U0001f999]+:', # Name + Emoji            
    ]
    patt = '^' + '|'.join(patts)     
    res = re.match(patt, s)   #Check if each line match with the pattern
    if res:
        return True
    return False
  
# SPlit each part(Date,Time,Name,SMS)
def getParts(line):   
    # Ex: '02/21/23, 11:27 - WOW: Eooooo'
    splitLine = line.split(' - ') 
    DateTime = splitLine[0]                     # '02/21/23 11:27 a. m.'
    splitDateTime = DateTime.split(' ')   
    date = splitDateTime[0]                    # '02/21/23'
    time = ' '.join(splitDateTime[1:])          # '11:27 a. m.'
    sms = ' '.join(splitLine[1:])             # 'WOW: Eooooo'
    if findPerson(sms): 
        splitSMS = sms.split(': ')      
        nameP = splitSMS[0]               # 'WOW' 
        sms = ' '.join(splitSMS[1:])    # 'Eooooo'
    else:
        nameP = None
    return date, time, nameP, sms

def getEmojis(line):
    emoList= ' '
    car1= regex.findall(r'\X',line)
    for caracter in car1:
        if any(c in emoji.EMOJI_DATA for c in caracter):
            emoList= [w.replace(' ',caracter) for w in emoList]
        pass
    return emoList                     
        
def makeDf(list_r):
    ret_list= []
    df_list= []
    for x in list_r:
        date,time,nameP,sms = None,None,None,None
        for i in range(40113): # for in with the size of documment , you can check the size with #print(i)
            #print(i)
            line= x.readline() #read each line
            if not line:
                pass
            if startLine(line.strip()):
                date,time,nameP,sms= getParts(line.strip())
                date= date.replace(',','')
                date
                ret_list.append([date,time,str(nameP),str(sms)])
                
        df_list.append(pd.DataFrame(ret_list, columns= ['Date','Time','Name','SMS']))
    #print(startLine(line.strip()))
    
    if len(df_list) > 1:
        df = pd.concat(df_list, ignore_index=True)
    else: 
        df = df_list[0]
        
          
    df['DayOfWeek']= pd.to_datetime(df['Date'], errors= 'coerce').dt.strftime('%A')
    df['Date']= pd.to_datetime(df['Date']).dt.strftime('%m/%d/%Y')
    df['Time']= pd.to_datetime(df['Time'],errors= 'coerce').dt.strftime('%H:%M')
    df['URLs'] = df['SMS'].apply(lambda x: re.findall(r'(https?://\S+)',x)).str.len()
    df['Emojis']= (df['SMS'].apply(getEmojis)) #to add the Column Emoji with the emojis
        
    return df
    
        

pathCht1= "./data.txt"
pathCht2= "./data2.txt"


f= open(pathCht1,"r",encoding='utf-8')
f1= open(pathCht2,"r",encoding='utf-8')



data= makeDf([f,f1])


##### Getting stats 

totalSMS= data['SMS'].shape[0]    # Total of sms in the chat

totalMedia= data[data['SMS'] == "<Media omitted>"].shape[0] # Total Media in the chat

totlLink= len(data['SMS'].apply(lambda x: re.findall(r'(https?://\S+)',x)).sum()) #total Links in the chat

totalEmoji= data[data['Emojis'].apply(lambda x: ' ' not in x)].shape[0] #total emojis in the chat

dic_sta= {'Type':["Total SMS","Total Media","Total Emojis","Total Link"],
          'Quantity':[totalSMS,totalMedia,totalEmoji,totlLink]
          }


sta = pd.DataFrame(dic_sta,columns= ["Type","Quantity"] )

#Plot the stats



plt.figure(figsize= (10,5))
sbn.barplot(x= sta['Type'],y=sta['Quantity'])
plt.show()

#Analisys of the Emojis
#the most common



emoCommon= (Counter((data['Emojis']).sum())).items()


emojiDF= pd.DataFrame(emoCommon,columns= ["Emoji","Quantity"]).sort_values(by="Quantity",ascending=False)
emojiDF= emojiDF[emojiDF['Emoji'].apply(lambda x : ' ' not in x)]
#print(emojiDF)

####Plot the top 5 emojis.
plt.figure(figsize= (10,5))
sbn.barplot(x= emojiDF['Emoji'].head(),y=emojiDF['Quantity'].head())
plt.show()

    #how to know who send more SMS
moreSMS= data.groupby("Name")['SMS'].count().sort_values(ascending=False).to_frame() 


# Divide sms with media and without

multiData= data[data['SMS'] == "<Media omitted>"] #SMS with media

justSMS= data.drop(multiData.index) #SMS without Media

justSMS['Leter']= justSMS['SMS'].apply(lambda x : len(x.replace(" ",""))) #Add Colummn Letter for stadistic

justSMS['Word']= justSMS['SMS'].apply(lambda x : len(x.split(' '))) #Add Column Word for stadistics


#Stadistics of the chat

#Mosts actives days

actDays= data['Date'].value_counts().head(10) #the top 10 of most actives days

#graph

#actDays.plot.bar()

#plt.show()

#Words per day mean

wPerD= int(data['Date'].value_counts().sum()/len(data['Date'].value_counts()))

# top 10 of most actives hours

actHours= data['Time'].value_counts().head(10)

#actHours.plot()
#actHours.plot.bar()

'''

stopwords = STOPWORDS.update(['siempre','dije','mismo','ella','bien','vas','dice','bueno','','q','d','cuando','ver','ir','ver','vi','ve','quien','cosa','iba','porque','oye','cosas','soy','ha','p9r','mal','iba','ahora','voy','vez','nada','pa','xq','okok','ok','ni','fue','ta','ti','ay',
                              'da','x','tan','van','da','ño','sé','que', 'qué', 'con', 'de', 'te', 'en', 'la', 'lo', 'le', 'el', 'las', 'los', 'les', 'por', 'es', 
                                                         'son', 'se', 'para', 'un', 'una', 'chicos', 'su', 'si', 'chic', 'nos', 'ya', 'hay', 'esta', 
                                                         'pero', 'del', 'mas', 'más', 'eso', 'este', 'como', 'así', 'todo', 'https', 'tu', 'y', 'al',
                                                         'mi', 'tus', 'esa', 'o', 'sus', 'tiene', 'también', 'tambien', 'sea', 'esos', 'esto', 'ese',
                                                         'uno', 'ahi', 'ahí', 'va', 'está', 'yo', 'tenga', 'ser', 'otra', 'otro', 'mis', 'han'])


ListCloud= ""

for i in justSMS['SMS']:
    words= str(i).lower().split()
    for j in words:
        ListCloud= ListCloud + j + ' '
        
worldcloud= WordCloud(width= 1000, height= 800,
                      background_color='white',stopwords=stopwords,min_font_size=10).generate(ListCloud)

worldcloud.to_image()

'''
#how many sms send each one

eSMS= data.groupby(['Name'])['SMS'].count()

data['#SMS']= 1

dateD= data.groupby('Date').sum()
dateD.reset_index(inplace=True)

#data['Time']= pd.to_datetime(data['Time'], format= '%H:%M' , errors= 'coerce' , utc= True).dt.strftime('%H:%M')
#data['rat']= data['Time'].apply(lambda x: x.time)

#data.to_csv("dataC1", index= False)
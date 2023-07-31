
###Scripts which should be initially automated:
####1. Combining csv files of In time and Out time
####2. Total Hours spent at institution(Only possible if csv's are merged)
##Total number of analytics done - 10 types(Includes student wise,dept wise,daywise,monthwise,mentorwise, total hours, latecomers,early leavers, students who spent less hours etc)
dayy=input('Enter date in DD-MM-YYYY format:')
monthh=input('Enter month number ranging from 1-12:')
if int(monthh)<1 or int(monthh)>12:
  print("Please enter Valid Month")
  exit()
deptt=input('Enter department:')
deptss=["CSE","ECE","MECH","CIVIL","EEE"]
if deptt not in deptss:
  print("Please enter valid department")
  exit()
import pandas as pd
import os
from datetime import datetime
outname = datetime.now().date()
outname=str(outname)
#outname = now.strftime("%H-%M-%S")
df=pd.read_csv('H:\Major Project\Backend\Intime_sample.csv')
df2=pd.read_csv('H:\Major Project\Backend\Outtime_sample.csv')
path = 'H:\Major Project\Backend\Reports'
df2=df2[['Out time', 'Roll Number','Early Leave']]
df3=pd.merge(df, df2, 
                   on='Roll Number', 
                   how='inner')
def latecomers():
  path1 = os.path.join(path,outname+'_latecomers_report.csv')
  df.loc[df['Late']=='Yes'].to_csv(path1, index=False)

def Intimers():
  path1 = os.path.join(path,outname+'_Intimers_report.csv')
  df.loc[df['Late'] == 'No'].to_csv(path1, index=False)

def Earlyleavers():
  path1 = os.path.join(path,outname+'_Earlyleavers_report.csv')
  df3.loc[df3['Early Leave']=='Yes'].to_csv(path1, index=False)



#DEPARTMENT WISE TYPES OF REPORTS
def deptwise(dept):
  path1 = os.path.join(path,outname+'_deptwise_report.csv')
  df.loc[df['Department']==dept].to_csv(path1, index=False)

def deptwiselate(dept):
  df_dept=df.loc[df['Department']==dept]
  df_deptlate=df_dept.loc[df['Late']=='Yes']
  path1 = os.path.join(path,outname+'_deptwiselate_report.csv')
  df_deptlate.to_csv(path1, index=False)


def deptwiseearly(dept):
  df_dept=df.loc[df['Department']==dept]
  path1 = os.path.join(path,outname+'_deptwiseearly_report.csv')
  df_dept.loc[df['Late']=='No'].to_csv(path1, index=False)



#Day and Monthwise
def daywise(day):
  path1 = os.path.join(path,outname+'_daywise_report.csv')
  df.loc[df['Date[MM//DD//YY]']==day].to_csv(path1, index=False)

def monthwise(month):
  month_df=pd.DataFrame()
  present=[]
  for i in df['Date[MM//DD//YY]']:
    if str(i)[3:5]==month and i not in present:
        present.append(i)
        month_df=month_df.append(df.loc[df['Date[MM//DD//YY]']==i],ignore_index=True)
  path1 = os.path.join(path,outname+ '_monthwise_report.csv')
  month_df.to_csv(path1, index=False)


#total hours of students and students who attended low hours
from datetime import datetime
def total_hours():
  FMT ='%H:%M:%S'
  df3['Total Hours']=''
  norows=df3.shape[0]
  for i in range(norows):
    df3.iloc[i:,len(df3.columns)-1]=datetime.strptime(df3.iloc[i,7], FMT) - datetime.strptime(df3.iloc[i,0], FMT)
  path1 = os.path.join(path,outname+ '_totalhours_report.csv')
  df3.to_csv(path1, index=False)

def low_hrs_attended():
  low_df=pd.DataFrame()
  present=[]
  for i in df3['Total Hours']:
    k=str(i)
    if int(k[0])<4 and k not in present:
      present.append(k)
      low_df=low_df.append(df3.loc[df3['Total Hours']==i],ignore_index=True)
  path1 = os.path.join(path,outname+'_low_hrs_attended_report.csv')
  low_df.to_csv(path1, index=False)

def mentor_wise():
  mentors=df3['Mentor'].unique()
  for i in mentors:
    k=df3.loc[df3['Mentor']==i]
    path1 = os.path.join(path, outname+'_'+str(i)+'_mentor_report.csv')
    k.to_csv(path1, index=False)

latecomers()
Intimers()
Earlyleavers()
deptwise(deptt)
deptwiselate(deptt)
deptwiseearly(deptt)
daywise(dayy)
monthwise(monthh)
total_hours()
low_hrs_attended()
mentor_wise()


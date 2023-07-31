# Real-time-Monitoring-System-using-CNN
Introduced a deep learning model using a combination of 2 different architectures(YoloV3 & Dlib), used 4 extracted coordinates from the YolovV3 pipeline in Dlib pipeline. Monitors 100+ students of more than 5 departments with 98% accuracy in 1 frame. Evaluates  students' faces inside campus and generates 8 types of analysis sheets that reduce work of 100+ of mentors in monitoring 1000+ of students daily.

## DATASET
### FOR IMAGES:
Use face focussed images of the people on whom you want to apply this model. Name the images with their respective names. Add them in the directory
### FOR USERS DETAILS
Create dictionaries using the details(Name, Roll number, Department etc) of the people whose images were used in the pipeline. Add them in ingate.py and outgate.py in mentioned areas
## OUTPUT
Run manually or use automated script to execute mp_report_analysis using the outputs of both ingate.py and outgate.py as inputs for mentioned executable file. 8 types of detailed analysis sheets will be produced as the output of mp_report_analysis which reduces the human efforts to a significant extent in monitoring.


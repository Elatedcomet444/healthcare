import csv  
    
# opening the CSV file  
with open('static/encodings.csv', mode ='r')as file:
  
      
  # reading the CSV file  
  csvFile = csv.reader(file)  
  print("lines:")
  # displaying the contents of the CSV file  
  for lines in csvFile:  
        print('lines:',lines)  
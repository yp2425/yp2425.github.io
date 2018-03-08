import pandas as pd

def getobject():
    data=pd.read_json('sample_conversations.json')
    data_length=len(data)
    customerdata=[]
    servicedata=[]
    
    for i in range(data_length):
        temp = data.Issues[i]['Messages']
        for j in range(len(temp)):
            if  temp[j]['IsFromCustomer']==True:
                customerdata.append(temp[j]['Text'])
            else:
                servicedata.append(temp[j]['Text'])
            
    train_customer=[]
    train_service=[]
    for line in customerdata:
        train_customer.append(line.strip())
    for line in servicedata:
        train_service.append(line.strip())
    
    #return train_customer, train_service
    return servicedata
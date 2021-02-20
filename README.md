# Human Fall Decection
This region shows the UP-Fall Detection dataset and depicts the technique of its acquiring, pre-dealing with, joining and storaging. Besides, one potential part extraction process is moreover point by point. 

## Data extraction
As setting careful sensors, we presented six infrared sensors as a system 0.40 m over the floor of the room, to check the alterations in impedance of the optical devices, where 0 infers obstruction and 1 no impedance. At last, two Microsoft LifeCam Cinema cameras were arranged at 1.82 m over the floor

![alt text](/images/3.png)

![alt text](/images/4.png)

## Description of the Dataset:

We present a tremendous dataset generally for fall revelation, specifically UP-Fall Detection, that joins 11 activities and 3 fundamentals for each activity. Subjects performed six direct human step by step practices similarly as five remarkable sorts of human falls. These data were assembled in excess of 17 strong young adults without shortcoming using a multimodal approach, i.e., wearable sensors, incorporating sensors and vision contraptions. The consolidated dataset (812 GB), similarly as, the part dataset (171 GB) are openly available 

## Labels in the Dataset

![alt text](/1.png?raw=true)

## Dataset(csv)
![alt text](/images/6.png)

## Dataset(image)
![alt text](/images/2.png)

## Mapping the datasets
Here we are mapping the CSV and IMAGE datasets by using a timestamp, the reason why we are using timestamp is image are stored in folders with names like -timestamp.png
image Folder structure
![alt text](/images/5.png)

## code
def map_image__csv(ts,activity):
  ts=ts.replace(':','_')
  image_path=f"Activity{activity}/{ts}.png"
  return image_path
def rearrange_dataframe(df):
  df["id"]=df.apply(lambda x:map_image__csv(x['TimeStamps'], x['Activity']),axis=1)
  return df
  
  Dataset with image path
  ![alt text](/images/7.png)
  
## Model
  RESnet
  restnet=models.resnet34(pretrained=True, progress=True)
  restnet_updated = nn.Sequential(*list(restnet.children())[:-1])-this sequential layer is used because of timesires data
  
  Bringing to low level features from High level
  
  class LsfmNet(nn.Module):
    def __init__(self,init_net,no_sensor_features,classes):
        super(LsfmNet, self).__init__()
        self.restnet = init_net
        self.fc1 = nn.Linear(512, 120)
        self.fc2 = nn.Linear(120+no_sensor_features, 60)
        self.fc3 = nn.Linear(60, classes)

   def forward(self, image,data):
        image=image.type(torch.float32)
        rest_out = self.restnet(image)
        x1=self.fc1(rest_out.squeeze())
        if len(x1.size())==1:x1=torch.unsqueeze(x1, dim=0)
        x = torch.cat((x1, data), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x)


lsfmnet = LsfmNet(restnet_updated,30,11)

## Confusion Matrix
![alt text](/images/8.png)

### Model accrucury is 89%
  


  
  
  
  

# PNEUMONIA_DETECTION
## Pneumoniua detection by CNN model with the help of keras. 
In which it takes the chest X-ray and take lung opacity as feature vector 
### Pneumonia classes Normal, No Lung Opacity / Not Normal, Lung Opacity
![IMAGE](https://github.com/Gulshan-gaur/PNEUMONIA_DETECTION/blob/master/images/Screenshot%20from%202019-11-11%2011-57-56.png)
### Structured of Neural network 
![IMAGE](https://github.com/Gulshan-gaur/PNEUMONIA_DETECTION/blob/master/images/Screenshot%20from%202019-11-11%2011-58-16.png)

### ROC Curve For Lung Opacity
![IMAGE](https://github.com/Gulshan-gaur/PNEUMONIA_DETECTION/blob/master/images/Screenshot%20from%202019-11-11%2012-00-12.png)

#### My work includes self laid neural network which was repeatedly tuned for one of the best hyperparameters and used variety of utility function of keras like callbacks for learning rate reduction and checkpointing. Other metrics like precision , recall and f1 score using confusion matrix were taken off special care. The other part included a brief introduction of transfer learning via Xception 

## Requirements
```
1. On GPU                          
  Requirement for loading model 
               1. CUDA capable GPU
               2. CUDA  10.0 Tool kit (  ubuntu 18.04)
               3. cudnn7  library (need account on nvidia developer )
               4. Anaconda3 
               5. Tensorflow-gpu
               6. Pydicom(For reading DICOM files)
               7. sklearn(for labeling )
               8. matplotlib
 2. On CPU
   Requirement for loading model:
               1. Anaconda3
               2. Tensorflow=1.11.0
               3. Pydicom
               4. sklearn
               5. matplotlib
```  
 
### Data provided by RSNA

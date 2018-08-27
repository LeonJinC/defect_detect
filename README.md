# defect_detect

![image](https://github.com/LeonJinC/defect_detect/blob/master/result.jpg)

**Environment:  Win10+VS2015+Opencv2.4.9**

The whole algorithm adopts online detection framework and target update method based on Kalman filtering.

The traditional defect detection method based on edge detection is difficult to enhance the subtle and complicated defects.

Using 360 degrees dynamic lighting solutions for detection and edge detector based on the Hessian matrix of image preprocessing, 

threshold in sequence image, the minimum target defects outsourcing rectangular two-dimensional point sets modeling object, 

through the model initialization vector, the adaptive change model, update the estimation error covariance, establish dynamic change the defects of objects, real-time tracking the position of the defect, 

so as to achieve the goal of detection and identification of defects.




**See the blog for the full implementation idea**  https://blog.csdn.net/jin739738709






#include"DefectFunctionLibrary.h"
#include"Ctracker.h"
#include "Blob.h"
#include<windows.h>



int main()
{

	double dur;
	clock_t start, end;
	start = clock();


	char *fn = "./defect_0001.avi";
	CTracker tracker(0.2, 0.5, 50.0, 10, 10);//自定义CTracker.h

	vector<Point2f> centers;//used to store points detected.
	vector<Blob> b;
	int widthStep, nChannel;
	uchar *pSrc;

	char EscKeyCheck = 0;



	/*CREATE BACKGROUND*/
	VideoCapture capVid(fn);//实例化VideoCapture的capVid1(fn)





	Mat grayFrame;
	Mat imgFrame;
	capVid.read(imgFrame); // read a frame| 读取第一帧图像


	VideoWriter writer;
	int isColor = 1;
	//double frame_fps = capVid.get(CV_CAP_PROP_FPS);
	double frame_fps = 2;//一秒一帧放慢速度
	int frame_width = imgFrame.size().width;//capVid.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height = imgFrame.size().height;//*2+10;//capVid.get(CV_CAP_PROP_FRAME_HEIGHT);
											  /*int frame_width = capVid.get(CV_CAP_PROP_FRAME_WIDTH);
											  int frame_height = capVid.get(CV_CAP_PROP_FRAME_HEIGHT);*/
	string video_name = "./result.avi";
	writer = VideoWriter(video_name, CV_FOURCC('F', 'L', 'V', '1'), frame_fps, Size(frame_width, frame_height), isColor);


	while (capVid.isOpened())
	{

		//cvtColor(imgFrame, grayFrame, CV_RGB2GRAY);

		grayFrame = surface_detect(imgFrame);

		widthStep = grayFrame.step[0];
		nChannel = grayFrame.step[1];//图片通道数

		for (int i = 0; i < grayFrame.rows; i++)
		{
			pSrc = grayFrame.ptr<uchar>(i);
			for (int j = 0; j < widthStep; j += nChannel)
			{
				if (pSrc[j] == 255)
				{
					bool isIn = false;
					for (int k = 0; k < b.size(); k++)
					{
						if (b[k].isInBlob(Point(j, i)))
						{
							b[k].Add(Point(j, i));
							isIn = true;
							break;
						}
					}
					if (!isIn)
					{
						Blob n;
						n.Add(Point(j, i));
						b.push_back(n);
					}
				}
			}
		}
		//去除互相包含的区域
		for (int i = 0; i < b.size(); i++)
		{
			for (int j = i + 1; j < b.size(); j++)
			{
				// blob j contains blob i
				if ((b[i].xmax <= b[j].xmax && b[i].xmin >= b[j].xmin
					&& b[i].ymax <= b[j].ymax && b[i].ymin >= b[j].ymin))
				{
					b.erase(b.begin() + i, b.begin() + i + 1);
					i--;
					break;
				}

				// blob i contains blob 
				if (b[j].xmax <= b[i].xmax && b[j].xmin >= b[i].xmin
					&& b[j].ymax <= b[i].ymax && b[j].ymin >= b[i].ymin)
				{
					b.erase(b.begin() + j, b.begin() + j + 1);
					break;
				}
			}
		}


		centers.clear();
		cvtColor(grayFrame, grayFrame, CV_GRAY2BGR);
		for (int i = 0; i < b.size(); i++)
		{
			if (b[i].size < 2)
				continue;
			rectangle
			(
				grayFrame,
				Rect
				(
					b[i].xmin - 20,
					b[i].ymin - 20,
					b[i].xmax - b[i].xmin + 40,
					b[i].ymax - b[i].ymin + 40
				),
				Scalar(0, 0, 255),
				2
			);
			Point center = Point(b[i].xcenter, b[i].ymax);
			centers.push_back(center);
		}
		b.clear();



		if (centers.size() > 0)
		{
			tracker.Update(centers);
			for (int i = 0; i < tracker.tracks.size(); i++)
			{
				if (tracker.tracks[i]->trace.size() > 1)
				{
					for (int j = 0; j < tracker.tracks[i]->trace.size() - 1; j++)
					{
						Point p1 = Point(tracker.tracks[i]->trace[j].x, tracker.tracks[i]->trace[j].y);
						Point p2 = Point(tracker.tracks[i]->trace[j + 1].x, tracker.tracks[i]->trace[j + 1].y);
						/*line
						(
						grayFrame,
						p1,
						p2,
						Colors[tracker.tracks[i]->track_id % 8],
						10,
						CV_AA
						);*/
						//circle(imgFrame, p1, 40, Colors[tracker.tracks[i]->track_id % 8], 3);
						circle(imgFrame, p1, 40, Scalar(0, 0, 255), 3);

					}
				}
			}
		}
		//circle(imgFrame, Point(600, 400), 40, Scalar(0, 0, 255), 3);


		Mat orishow = imgFrame.clone();
		Mat grayshow = grayFrame.clone();
		//Mat mergeshow;
		//mergeImg(mergeshow, orishow, grayshow);


		//cout << "hello world" << endl;
		//writer.write(imgFrame);



		Mat mergeshow;
		mergeImg(mergeshow, orishow, grayshow);
		//cout << mergeshow.size() << endl;s
		Mat showImage = mergeshow.clone();
		resize(showImage, showImage, Size(0, 0), 0.30, 0.30);
		//namedWindow("output", 0);
		imshow("output", showImage);
		waitKey(30);// 30/1000s=0.03s








					//break;

		if ((capVid.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVid.get(CV_CAP_PROP_FRAME_COUNT))
		{
			capVid.read(imgFrame);//读取下一帧图像

		}
		else
		{
			std::cout << "end of video\n";
			break;
		}




	}//end of while (capVid.isOpened() && EscKeyCheck != 27)
	if (EscKeyCheck != 27)
	{               // if the user did not press esc (i.e. we reached the end of the video)
		cv::waitKey(0);                         // hold the windows open to allow the "end of video" message to show
	}

	end = clock();
	dur = (double)(end - start) / CLOCKS_PER_SEC;
	dur = dur / 60;
	cout << "detect using time:" << dur << " min" << endl;

	return 0;
}

#ifndef __DEFECTFUNCTIONLIBRARY_H__
#define __DEFECTFUNCTIONLIBRARY_H__

#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
#include<io.h>
#include<time.h>

using namespace std;
using namespace cv;


Mat getPaddedImage(Mat &image);

void FFT(Mat image, Mat &complexImg);

void getBandPass(Mat &padded, Mat &bandpass);

void calculateSnP(Mat &Real, Mat &Imaginary);

void removeMargin(Mat &Image, int margin);

void reduceTheImage(Mat &dialatedImage, Mat &reducedImage);

void preProcessImage(Mat &oringinImage, Mat &reducedImage);

bool search8neighbor(Mat &L, cv::Point searchpoint, int max);

void StegerLine(Mat &img0, Mat &img_result);

void lines_gauss(Mat &reducedImage, Mat &logLines, int size, int max, int min);

void RemoveSmallRegion(Mat &Src, Mat &Dst, int AreaLimit, int CheckMode, int NeihborMode, int mod);

void mergeImg(Mat &dst, Mat &src1, Mat &src2);

Mat surface_detect(Mat &color);

#endif // !__DEFECTFUNCTIONLIBRARY_H__



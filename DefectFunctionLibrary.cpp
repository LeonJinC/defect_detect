#include"DefectFunctionLibrary.h"

Mat getPaddedImage(Mat &image)
{
	int w = getOptimalDFTSize(image.cols);
	int h = getOptimalDFTSize(image.rows);
	Mat padded;
	copyMakeBorder(image, padded, 0, h - image.rows, 0, w - image.cols, BORDER_CONSTANT, Scalar::all(0));
	padded.convertTo(padded, CV_32FC1);
	return padded;
}

void FFT(Mat image, Mat &complexImg)
{
	int w = getOptimalDFTSize(image.cols);
	int h = getOptimalDFTSize(image.rows);
	Mat padded;
	copyMakeBorder(image, padded, 0, h - image.rows, 0, w - image.cols, BORDER_CONSTANT, Scalar::all(0));
	padded.convertTo(padded, CV_32FC1);
	//imshow("padded", padded);
	for (int i = 0; i < padded.rows; i++)
	{
		float *ptr = padded.ptr<float>(i);
		for (int j = 0; j < padded.cols; j++)   ptr[j] *= pow(-1, i + j);
	}
	Mat plane[] = { padded, Mat::zeros(padded.size(), CV_32F) };

	merge(plane, 2, complexImg);
	dft(complexImg, complexImg);

}


void getBandPass(Mat &padded, Mat &bandpass)
{


	float D0 = 50;
	float W = 200;
	for (int i = 0; i < padded.rows; i++)
	{
		float*p = bandpass.ptr<float>(i);

		for (int j = 0; j < padded.cols; j++)
		{
			float D_pow = pow(i - padded.rows / 2, 2) + pow(j - padded.cols / 2, 2);
			float D_sqrt = sqrtf(D_pow);
			float D0_pow = pow(D0, 2);
			p[2 * j] = expf(-pow(((D_pow - D0_pow) / (D_sqrt*W)), 2));
			p[2 * j + 1] = expf(-pow(((D_pow - D0_pow) / (D_sqrt*W)), 2));
		}
	}

}

void calculateSnP(Mat &Real, Mat &Imaginary)
{ //Ƶ�׺���λ�׾�δlog����͹�һ������
	magnitude(Real, Imaginary, Real);
	phase(Real, Imaginary, Imaginary);
}


void removeMargin(Mat &Image, int margin)
{
	for (int i = 0; i < Image.rows; i++)
		for (int j = 0; j < Image.cols; j++)
		{
			if (i >= 0 && i <= margin)
				Image.at<uchar>(i, j) = 0;
			if (i >= Image.rows - margin && i < Image.rows)
				Image.at<uchar>(i, j) = 0;
			if (j >= 0 && j <= margin)
				Image.at<uchar>(i, j) = 0;
			if (j >= Image.cols - margin && j < Image.cols)
				Image.at<uchar>(i, j) = 0;
		}
}


void reduceTheImage(Mat &dialatedImage, Mat &reducedImage)
{
	//GaussianBlur(reducedImage, reducedImage, Size(3, 3), 2, 2);
	medianBlur(reducedImage, reducedImage, 5);
	for (int i = 0; i < reducedImage.rows; i++)
		for (int j = 0; j < reducedImage.cols; j++)
		{
			if (dialatedImage.at<uchar>(i, j) < 10)
				reducedImage.at<uchar>(i, j) = 0;
		}
}

void preProcessImage(Mat &oringinImage, Mat &reducedImage)
{

	int m = oringinImage.rows;
	int n = oringinImage.cols;
	Mat invertedImage = ~oringinImage;

	Mat padded = getPaddedImage(invertedImage);
	Mat plane[] = { padded, Mat::zeros(padded.size(), CV_32F) };

	Mat complexImg;
	FFT(invertedImage, complexImg);

	Mat bandpass(padded.size(), CV_32FC2);//��ͨ��
	getBandPass(padded, bandpass);

	Mat	convolImage;
	multiply(complexImg, bandpass, convolImage);

	Mat shiftedfilteredImage;
	idft(convolImage, shiftedfilteredImage);
	split(shiftedfilteredImage, plane);
	calculateSnP(plane[0], plane[1]);
	normalize(plane[0], plane[0], 0, 1, CV_MINMAX);

	Mat segImage = plane[0](Rect(0, 0, n, m));

	segImage.convertTo(segImage, CV_8UC1, 255.0);

	int blockSize = 17;
	int constValue = 10;
	Mat local = Mat::zeros(segImage.rows, segImage.cols, CV_8U);
	adaptiveThreshold(segImage, local, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);

	removeMargin(local, 50);

	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(20, 20));
	Mat dilatedImage;
	dilate(local, dilatedImage, element);

	reduceTheImage(dilatedImage, reducedImage);
	//imwrite("reducedImage.bmp", reducedImage);


}

bool search8neighbor(Mat &L, cv::Point searchpoint, int max)
{
	vector<Point> neighbor_points = { Point(-1,-1)	,	Point(0,-1)	,	Point(1,-1)	,	Point(1,0),
		Point(1,1)	,   Point(0,1)	,	Point(-1,1)	,	Point(-1,0) };
	for (int i = 0; i < neighbor_points.size(); i++)
	{
		Point tmppoint = searchpoint + neighbor_points[i];
		if (L.at<double>(tmppoint) > max)
		{
			return true;
		}
	}
	return false;


}


void StegerLine(Mat &img0, Mat &img_result)
{
	//Mat img0 = imread("logLines.png", 1);

	//cvtColor(img0, img0, CV_BGR2GRAY);



	Mat img;
	img = img0.clone();

	//��˹�˲�
	img.convertTo(img, CV_32FC1);
	GaussianBlur(img, img, Size(0, 0), 5, 5);

	//һ��ƫ����
	Mat m1, m2;
	m1 = (Mat_<float>(1, 2) << 1, -1);  //xƫ��
	m2 = (Mat_<float>(2, 1) << 1, -1);  //yƫ��

	Mat dx, dy;
	filter2D(img, dx, CV_32FC1, m1);
	filter2D(img, dy, CV_32FC1, m2);

	//����ƫ����
	Mat m3, m4, m5;
	m3 = (Mat_<float>(1, 3) << 1, -2, 1);   //����xƫ��
	m4 = (Mat_<float>(3, 1) << 1, -2, 1);   //����yƫ��
	m5 = (Mat_<float>(2, 2) << 1, -1, -1, 1);   //����xyƫ��

	Mat dxx, dyy, dxy;
	filter2D(img, dxx, CV_32FC1, m3);
	filter2D(img, dyy, CV_32FC1, m4);
	filter2D(img, dxy, CV_32FC1, m5);

	//hessian����
	double maxD = -1;
	int imgcol = img.cols;
	int imgrow = img.rows;
	vector<double> Pt;
	for (int i = 0; i < imgcol; i++)
	{
		for (int j = 0; j < imgrow; j++)
		{
			if (img0.at<uchar>(j, i) > 200)
			{
				//����2*2��hessian����
				Mat hessian(2, 2, CV_32FC1);
				hessian.at<float>(0, 0) = dxx.at<float>(j, i);
				hessian.at<float>(0, 1) = dxy.at<float>(j, i);
				hessian.at<float>(1, 0) = dxy.at<float>(j, i);
				hessian.at<float>(1, 1) = dyy.at<float>(j, i);


				Mat eValue;
				Mat eVectors;
				//eigen����hessian���������ֵ����������
				eigen(hessian, eValue, eVectors);

				double nx, ny;
				double fmaxD = 0;
				if (fabs(eValue.at<float>(0, 0)) >= fabs(eValue.at<float>(1, 0)))  //������ֵ���ʱ��Ӧ����������
				{
					nx = eVectors.at<float>(0, 0);
					ny = eVectors.at<float>(0, 1);
					fmaxD = eValue.at<float>(0, 0);
				}
				else
				{
					nx = eVectors.at<float>(1, 0);
					ny = eVectors.at<float>(1, 1);
					fmaxD = eValue.at<float>(1, 0);
				}

				double t = -(nx*dx.at<float>(j, i) + ny*dy.at<float>(j, i)) / (nx*nx*dxx.at<float>(j, i) + 2 * nx*ny*dxy.at<float>(j, i) + ny*ny*dyy.at<float>(j, i));

				if (fabs(t*nx) <= 0.5 && fabs(t*ny) <= 0.5)//fabs �������ľ���ֵ
				{
					Pt.push_back(i);
					Pt.push_back(j);
				}
			}
		}
	}

	for (int k = 0; k < Pt.size() / 2; k++)
	{
		Point rpt;
		rpt.x = Pt[2 * k + 0];
		rpt.y = Pt[2 * k + 1];
		//circle(img0, rpt, 1, Scalar(0, 0, 255));
		circle(img_result, rpt, 0.5, Scalar(255, 255, 255), 0.5);
	}
	//imshow("oringinal", img0);
	//imshow("stegerLine", img_result);
	//imwrite("stegerLine.png", img_result);
	//waitKey(0);
}


void lines_gauss(Mat &reducedImage, Mat &logLines, int size, int max, int min)
{
	//����ƫ����+�ͺ���ֵ��	+	stegerϸ����Ե���������׶���ƫ��������������ֵ����������+���ߵķ������Ͻ����̩�չ�ʽ
	if (size == 5)
	{
		double logFilter[5][5] =
		{
			{ 0.0448 ,	 0.0468	 ,	 0.0564	 ,	 0.0468	 ,	 0.0448 },

			{ 0.0468 ,   0.3167  ,   0.7146  ,   0.3167  ,   0.0468 },

			{ 0.0564 ,   0.7146  ,  -4.9048  ,   0.7146  ,   0.0564 },

			{ 0.0468 ,   0.3167  ,   0.7146  ,   0.3167  ,   0.0468 },

			{ 0.0448 ,   0.0468  ,   0.0564  ,   0.0468  ,   0.0448 }
		};
		Mat L = Mat::zeros(reducedImage.rows, reducedImage.cols, CV_64FC1);
		for (int j = 2; j < reducedImage.rows - 2; j++)
			for (int k = 2; k < reducedImage.cols - 2; k++)
			{
				double T[25] =
				{
					(logFilter[0][0])*reducedImage.at<uchar>(j - 2,k - 2)    ,     (logFilter[0][1])*reducedImage.at<uchar>(j - 2,k - 1)  ,    (logFilter[0][2])*reducedImage.at<uchar>(j - 2,k)   ,    (logFilter[0][3])*reducedImage.at<uchar>(j - 2,k + 1)  ,   (logFilter[0][4])*reducedImage.at<uchar>(j - 2,k + 2) ,

					(logFilter[1][0])*reducedImage.at<uchar>(j - 1,k - 2)    ,		(logFilter[1][1])*reducedImage.at<uchar>(j - 1,k - 1)  ,    (logFilter[1][2])*reducedImage.at<uchar>(j - 1,k)   ,    (logFilter[1][3])*reducedImage.at<uchar>(j - 1,k + 1)  ,   (logFilter[1][4])*reducedImage.at<uchar>(j - 1,k + 2) ,

					(logFilter[2][0])*reducedImage.at<uchar>(j    ,k - 2)	 ,	    (logFilter[2][1])*reducedImage.at<uchar>(j    ,k - 1)  ,    (logFilter[2][2])*reducedImage.at<uchar>(j    ,k)   ,    (logFilter[2][3])*reducedImage.at<uchar>(j    ,k + 1)  ,   (logFilter[2][4])*reducedImage.at<uchar>(j    ,k + 2) ,

					(logFilter[3][0])*reducedImage.at<uchar>(j + 1,k - 2)	 ,		(logFilter[3][1])*reducedImage.at<uchar>(j + 1,k - 1)  ,    (logFilter[3][2])*reducedImage.at<uchar>(j + 1,k)   ,    (logFilter[3][3])*reducedImage.at<uchar>(j + 1,k + 1)  ,   (logFilter[3][4])*reducedImage.at<uchar>(j + 1,k + 2) ,

					(logFilter[4][0])*reducedImage.at<uchar>(j + 2,k - 2)	 ,		(logFilter[4][1])*reducedImage.at<uchar>(j + 2,k - 1)  ,    (logFilter[4][2])*reducedImage.at<uchar>(j + 2,k)   ,    (logFilter[4][3])*reducedImage.at<uchar>(j + 2,k + 1)  ,   (logFilter[4][4])*reducedImage.at<uchar>(j + 2,k + 2)
				};
				double median[25] =
				{
					reducedImage.at<uchar>(j - 2,k - 2)     ,     reducedImage.at<uchar>(j - 2,k - 1)  ,    reducedImage.at<uchar>(j - 2,k)   ,    reducedImage.at<uchar>(j - 2,k + 1)  ,   reducedImage.at<uchar>(j - 2,k + 2) ,

					reducedImage.at<uchar>(j - 1,k - 2)     ,	  reducedImage.at<uchar>(j - 1,k - 1)  ,    reducedImage.at<uchar>(j - 1,k)   ,    reducedImage.at<uchar>(j - 1,k + 1)  ,   reducedImage.at<uchar>(j - 1,k + 2) ,

					reducedImage.at<uchar>(j    ,k - 2)		,	  reducedImage.at<uchar>(j    ,k - 1)  ,    reducedImage.at<uchar>(j    ,k)   ,    reducedImage.at<uchar>(j    ,k + 1)  ,   reducedImage.at<uchar>(j    ,k + 2) ,

					reducedImage.at<uchar>(j + 1,k - 2)		,	  reducedImage.at<uchar>(j + 1,k - 1)  ,    reducedImage.at<uchar>(j + 1,k)   ,    reducedImage.at<uchar>(j + 1,k + 1)  ,   reducedImage.at<uchar>(j + 1,k + 2) ,

					reducedImage.at<uchar>(j + 2,k - 2)		,	  reducedImage.at<uchar>(j + 2,k - 1)  ,    reducedImage.at<uchar>(j + 2,k)   ,    reducedImage.at<uchar>(j + 2,k + 1)  ,   reducedImage.at<uchar>(j + 2,k + 2)
				};
				double m = 0;
				double prod = 1;
				double sum = 0;
				for (int i = 0; i < 25; i++)
				{
					prod *= T[i];
					sum += T[i];
					m += median[i];
				}
				m = m / 25;

				if (prod == 0)
				{

					L.at<double>(j, k) = 0;

				}
				else {
					if (m > 130)
					{
						//cout << "hello world!" << endl;
						L.at<double>(j, k) = fabs(sum) - 30;
					}
					else
					{
						L.at<double>(j, k) = fabs(sum);
					}
					//L.at<double>(j, k) = fabs(sum);

				}

			}

		for (int i = 0; i < L.rows; i++)
			for (int j = 0; j < L.cols; j++)
			{
				if (L.at<double>(i, j) > max)
				{
					logLines.at<uchar>(i, j) = 255;
				}
				else {
					if (L.at<double>(i, j) > min && L.at<double>(i, j) <= max)
					{
						cv::Point searchpoint = cv::Point(j, i);
						if (search8neighbor(L, searchpoint, max))
						{
							logLines.at<uchar>(i, j) = 255;
						}
					}
					else {
						logLines.at<uchar>(i, j) = 0;
					}
				}
			}

	}

	if (size == 9)
	{
		double logFilter[9][9] =
		{
			{ 0.0138    ,0.0138    ,0.0138    ,0.0138    ,0.0138    ,0.0138    ,0.0138    ,0.0138    ,0.0138 },

			{ 0.0138    ,0.0138    ,0.0138    ,0.0138    ,0.0138    ,0.0138    ,0.0138    ,0.0138    ,0.0138 },

			{ 0.0138    ,0.0138    ,0.0138    ,0.0158    ,0.0254    ,0.0158    ,0.0138    ,0.0138    ,0.0138 },

			{ 0.0138    ,0.0138    ,0.0158    ,0.2858    ,0.6837    ,0.2858    ,0.0158    ,0.0138    ,0.0138 },

			{ 0.0138    ,0.0138    ,0.0254    ,0.6837    ,-4.9357   ,0.6837    ,0.0254    ,0.0138    ,0.0138 },

			{ 0.0138    ,0.0138    ,0.0158    ,0.2858    ,0.6837    ,0.2858    ,0.0158    ,0.0138    ,0.0138 },

			{ 0.0138    ,0.0138    ,0.0138    ,0.0158    ,0.0254    ,0.0158    ,0.0138    ,0.0138    ,0.0138 },

			{ 0.0138    ,0.0138    ,0.0138    ,0.0138    ,0.0138    ,0.0138    ,0.0138    ,0.0138    ,0.0138 },

			{ 0.0138    ,0.0138    ,0.0138    ,0.0138    ,0.0138    ,0.0138    ,0.0138    ,0.0138    ,0.0138 }
		};

		Mat L = Mat::zeros(reducedImage.rows, reducedImage.cols, CV_64FC1);
		for (int j = 4; j < reducedImage.rows - 4; j++)
			for (int k = 4; k < reducedImage.cols - 4; k++)
			{
				double T[81] =
				{
					(logFilter[0][0])*reducedImage.at<uchar>(j - 4,k - 4)	 ,(logFilter[0][1])*reducedImage.at<uchar>(j - 4,k - 3)		 ,(logFilter[0][2])*reducedImage.at<uchar>(j - 4,k - 2)		 ,(logFilter[0][3])*reducedImage.at<uchar>(j - 4,k - 1)  ,    (logFilter[0][4])*reducedImage.at<uchar>(j - 4,k)   ,    (logFilter[0][5])*reducedImage.at<uchar>(j - 4,k + 1)  ,   (logFilter[0][6])*reducedImage.at<uchar>(j - 4,k + 2) ,    (logFilter[0][7])*reducedImage.at<uchar>(j - 4,k + 3)  ,   (logFilter[0][8])*reducedImage.at<uchar>(j - 4,k + 4),

					(logFilter[1][0])*reducedImage.at<uchar>(j - 3,k - 4)	 ,(logFilter[1][1])*reducedImage.at<uchar>(j - 3,k - 3)		 ,(logFilter[1][2])*reducedImage.at<uchar>(j - 3,k - 2)		 ,(logFilter[1][3])*reducedImage.at<uchar>(j - 3,k - 1)  ,    (logFilter[1][4])*reducedImage.at<uchar>(j - 3,k)   ,    (logFilter[1][5])*reducedImage.at<uchar>(j - 3,k + 1)  ,   (logFilter[1][6])*reducedImage.at<uchar>(j - 3,k + 2) ,    (logFilter[1][7])*reducedImage.at<uchar>(j - 3,k + 3)  ,   (logFilter[1][8])*reducedImage.at<uchar>(j - 3,k + 4),

					(logFilter[2][0])*reducedImage.at<uchar>(j - 2,k - 4)	 ,(logFilter[2][1])*reducedImage.at<uchar>(j - 2,k - 3)		 ,(logFilter[2][2])*reducedImage.at<uchar>(j - 2,k - 2)      ,(logFilter[2][3])*reducedImage.at<uchar>(j - 2,k - 1)  ,    (logFilter[2][4])*reducedImage.at<uchar>(j - 2,k)   ,    (logFilter[2][5])*reducedImage.at<uchar>(j - 2,k + 1)  ,   (logFilter[2][6])*reducedImage.at<uchar>(j - 2,k + 2)  ,    (logFilter[2][7])*reducedImage.at<uchar>(j - 2,k + 3)  ,   (logFilter[2][8])*reducedImage.at<uchar>(j - 2,k + 4),

					(logFilter[3][0])*reducedImage.at<uchar>(j - 1,k - 4)	 ,(logFilter[3][1])*reducedImage.at<uchar>(j - 1,k - 3)		 ,(logFilter[3][2])*reducedImage.at<uchar>(j - 1,k - 2)      ,(logFilter[3][3])*reducedImage.at<uchar>(j - 1,k - 1)  ,    (logFilter[3][4])*reducedImage.at<uchar>(j - 1,k)   ,    (logFilter[3][5])*reducedImage.at<uchar>(j - 1,k + 1)  ,   (logFilter[3][6])*reducedImage.at<uchar>(j - 1,k + 2) ,    (logFilter[3][7])*reducedImage.at<uchar>(j - 1,k + 3)  ,   (logFilter[3][8])*reducedImage.at<uchar>(j - 1,k + 4),

					(logFilter[4][0])*reducedImage.at<uchar>(j	  ,k - 4)	 ,(logFilter[4][1])*reducedImage.at<uchar>(j    ,k - 3)		 ,(logFilter[4][2])*reducedImage.at<uchar>(j    ,k - 2)	     ,(logFilter[4][3])*reducedImage.at<uchar>(j    ,k - 1)  ,    (logFilter[4][4])*reducedImage.at<uchar>(j    ,k)   ,    (logFilter[4][5])*reducedImage.at<uchar>(j    ,k + 1)  ,   (logFilter[4][6])*reducedImage.at<uchar>(j    ,k + 2)  ,    (logFilter[4][7])*reducedImage.at<uchar>(j   ,k + 3)  ,   (logFilter[4][8])*reducedImage.at<uchar>(j    ,k + 4),

					(logFilter[5][0])*reducedImage.at<uchar>(j + 1,k - 4)	 ,(logFilter[5][1])*reducedImage.at<uchar>(j + 1,k - 3)		 ,(logFilter[5][2])*reducedImage.at<uchar>(j + 1,k - 2)	     ,(logFilter[5][3])*reducedImage.at<uchar>(j + 1,k - 1)  ,    (logFilter[5][4])*reducedImage.at<uchar>(j + 1,k)   ,    (logFilter[5][5])*reducedImage.at<uchar>(j + 1,k + 1)  ,   (logFilter[5][6])*reducedImage.at<uchar>(j + 1,k + 2)  ,    (logFilter[5][7])*reducedImage.at<uchar>(j + 1,k + 3)  ,   (logFilter[5][8])*reducedImage.at<uchar>(j + 1,k + 4),

					(logFilter[6][0])*reducedImage.at<uchar>(j + 2,k - 4)	 ,(logFilter[6][1])*reducedImage.at<uchar>(j + 2,k - 3)		 ,(logFilter[6][2])*reducedImage.at<uchar>(j + 2,k - 2)	     ,(logFilter[6][3])*reducedImage.at<uchar>(j + 2,k - 1)  ,    (logFilter[6][4])*reducedImage.at<uchar>(j + 2,k)   ,    (logFilter[6][5])*reducedImage.at<uchar>(j + 2,k + 1)  ,   (logFilter[6][6])*reducedImage.at<uchar>(j + 2,k + 2) ,    (logFilter[6][7])*reducedImage.at<uchar>(j + 2,k + 3)  ,   (logFilter[6][8])*reducedImage.at<uchar>(j + 2,k + 4),

					(logFilter[7][0])*reducedImage.at<uchar>(j + 3,k - 4)	 ,(logFilter[7][1])*reducedImage.at<uchar>(j + 3,k - 3)		 ,(logFilter[7][2])*reducedImage.at<uchar>(j + 3,k - 2)	     ,(logFilter[7][3])*reducedImage.at<uchar>(j + 3,k - 1)  ,    (logFilter[7][4])*reducedImage.at<uchar>(j + 3,k)   ,    (logFilter[7][5])*reducedImage.at<uchar>(j + 3,k + 1)  ,   (logFilter[7][6])*reducedImage.at<uchar>(j + 3,k + 2) ,    (logFilter[7][7])*reducedImage.at<uchar>(j + 3,k + 3)  ,   (logFilter[7][8])*reducedImage.at<uchar>(j + 3,k + 4),

					(logFilter[8][0])*reducedImage.at<uchar>(j + 4,k - 4)	 ,(logFilter[8][1])*reducedImage.at<uchar>(j + 4,k - 3)		 ,(logFilter[8][2])*reducedImage.at<uchar>(j + 4,k - 2)	     ,(logFilter[8][3])*reducedImage.at<uchar>(j + 4,k - 1)  ,    (logFilter[8][4])*reducedImage.at<uchar>(j + 4,k)   ,    (logFilter[8][5])*reducedImage.at<uchar>(j + 4,k + 1)  ,   (logFilter[8][6])*reducedImage.at<uchar>(j + 4,k + 2) ,    (logFilter[8][7])*reducedImage.at<uchar>(j + 4 ,k + 3)  ,   (logFilter[8][8])*reducedImage.at<uchar>(j + 4,k + 4)
				};
				double median[81] =
				{
					reducedImage.at<uchar>(j - 4,k - 4)		 ,reducedImage.at<uchar>(j - 4,k - 3)		 ,reducedImage.at<uchar>(j - 4,k - 2)		 ,reducedImage.at<uchar>(j - 4,k - 1)  ,    reducedImage.at<uchar>(j - 4,k)   ,    reducedImage.at<uchar>(j - 4,k + 1)  ,   reducedImage.at<uchar>(j - 4,k + 2)  ,    reducedImage.at<uchar>(j - 4,k + 3)   ,   reducedImage.at<uchar>(j - 4,k + 4),

					reducedImage.at<uchar>(j - 3,k - 4)		 ,reducedImage.at<uchar>(j - 3,k - 3)		 ,reducedImage.at<uchar>(j - 3,k - 2)		 ,reducedImage.at<uchar>(j - 3,k - 1)  ,    reducedImage.at<uchar>(j - 3,k)   ,    reducedImage.at<uchar>(j - 3,k + 1)  ,   reducedImage.at<uchar>(j - 3,k + 2)  ,    reducedImage.at<uchar>(j - 3,k + 3)   ,   reducedImage.at<uchar>(j - 3,k + 4),

					reducedImage.at<uchar>(j - 2,k - 4)		 ,reducedImage.at<uchar>(j - 2,k - 3)		 ,reducedImage.at<uchar>(j - 2,k - 2)        ,reducedImage.at<uchar>(j - 2,k - 1)  ,    reducedImage.at<uchar>(j - 2,k)   ,    reducedImage.at<uchar>(j - 2,k + 1)  ,   reducedImage.at<uchar>(j - 2,k + 2)  ,    reducedImage.at<uchar>(j - 2,k + 3)   ,   reducedImage.at<uchar>(j - 2,k + 4),

					reducedImage.at<uchar>(j - 1,k - 4)		 ,reducedImage.at<uchar>(j - 1,k - 3)		 ,reducedImage.at<uchar>(j - 1,k - 2)        ,reducedImage.at<uchar>(j - 1,k - 1)  ,    reducedImage.at<uchar>(j - 1,k)   ,    reducedImage.at<uchar>(j - 1,k + 1)  ,   reducedImage.at<uchar>(j - 1,k + 2)  ,    reducedImage.at<uchar>(j - 1,k + 3)   ,   reducedImage.at<uchar>(j - 1,k + 4),

					reducedImage.at<uchar>(j	,k - 4)	     ,reducedImage.at<uchar>(j    ,k - 3)		 ,reducedImage.at<uchar>(j    ,k - 2)	     ,reducedImage.at<uchar>(j    ,k - 1)  ,    reducedImage.at<uchar>(j    ,k)   ,    reducedImage.at<uchar>(j    ,k + 1)  ,   reducedImage.at<uchar>(j    ,k + 2)  ,    reducedImage.at<uchar>(j    ,k + 3)   ,   reducedImage.at<uchar>(j    ,k + 4),

					reducedImage.at<uchar>(j + 1,k - 4)	     ,reducedImage.at<uchar>(j + 1,k - 3)		 ,reducedImage.at<uchar>(j + 1,k - 2)	     ,reducedImage.at<uchar>(j + 1,k - 1)  ,    reducedImage.at<uchar>(j + 1,k)   ,    reducedImage.at<uchar>(j + 1,k + 1)  ,   reducedImage.at<uchar>(j + 1,k + 2)  ,    reducedImage.at<uchar>(j + 1,k + 3)   ,   reducedImage.at<uchar>(j + 1,k + 4),

					reducedImage.at<uchar>(j + 2,k - 4)  	 ,reducedImage.at<uchar>(j + 2,k - 3)		 ,reducedImage.at<uchar>(j + 2,k - 2)	     ,reducedImage.at<uchar>(j + 2,k - 1)  ,    reducedImage.at<uchar>(j + 2,k)   ,    reducedImage.at<uchar>(j + 2,k + 1)  ,   reducedImage.at<uchar>(j + 2,k + 2)  ,    reducedImage.at<uchar>(j + 2,k + 3)   ,   reducedImage.at<uchar>(j + 2,k + 4),

					reducedImage.at<uchar>(j + 3,k - 4)	     ,reducedImage.at<uchar>(j + 3,k - 3)		 ,reducedImage.at<uchar>(j + 3,k - 2)	     ,reducedImage.at<uchar>(j + 3,k - 1)  ,    reducedImage.at<uchar>(j + 3,k)   ,    reducedImage.at<uchar>(j + 3,k + 1)  ,   reducedImage.at<uchar>(j + 3,k + 2)  ,    reducedImage.at<uchar>(j + 3,k + 3)   ,   reducedImage.at<uchar>(j + 3,k + 4),

					reducedImage.at<uchar>(j + 4,k - 4)  	 ,reducedImage.at<uchar>(j + 4,k - 3)		 ,reducedImage.at<uchar>(j + 4,k - 2)	     ,reducedImage.at<uchar>(j + 4,k - 1)  ,    reducedImage.at<uchar>(j + 4,k)   ,    reducedImage.at<uchar>(j + 4,k + 1)  ,   reducedImage.at<uchar>(j + 4,k + 2)  ,    reducedImage.at<uchar>(j + 4 ,k + 3)  ,   reducedImage.at<uchar>(j + 4,k + 4)
				};
				double m = 0;
				double prod = 1;
				double sum = 0;
				for (int i = 0; i < 81; i++)
				{
					prod *= T[i];
					sum += T[i];
					m += median[i];
				}
				m = m / 81;
				if (prod == 0)
				{

					L.at<double>(j, k) = 0;

				}
				else {
					if (m > 130)
					{

						L.at<double>(j, k) = fabs(sum) - 30;
					}
					else
					{
						L.at<double>(j, k) = fabs(sum);
					}

				}

			}

		for (int i = 0; i < L.rows; i++)
			for (int j = 0; j < L.cols; j++)
			{
				if (L.at<double>(i, j) > max)
				{
					logLines.at<uchar>(i, j) = 255;
				}
				else {
					if (L.at<double>(i, j) > min && L.at<double>(i, j) <= max)
					{
						cv::Point searchpoint = cv::Point(j, i);
						if (search8neighbor(L, searchpoint, max))
						{
							logLines.at<uchar>(i, j) = 255;
						}
					}
					else {
						logLines.at<uchar>(i, j) = 0;
					}
				}
			}

	}

	int margin = 10;

	for (int i = 0; i < logLines.rows; i++)
		for (int j = 0; j < logLines.cols; j++)
		{
			if (i >= 0 && i <= margin)
				logLines.at<uchar>(i, j) = 0;
			if (i >= logLines.rows - margin && i <= logLines.rows)
				logLines.at<uchar>(i, j) = 0;
			if (j >= 0 && j <= margin)
				logLines.at<uchar>(i, j) = 0;
			if (j >= logLines.cols - margin && j <= logLines.cols)
				logLines.at<uchar>(i, j) = 0;
		}


	int m = logLines.rows;
	int n = logLines.cols;
	Mat stegerLine = Mat::zeros(m, n, CV_8UC1);
	StegerLine(logLines, stegerLine);
	logLines = stegerLine.clone();
}

void RemoveSmallRegion(Mat &Src, Mat &Dst, int AreaLimit, int CheckMode, int NeihborMode, int mod)
{
	/*
	RemoveCount
	PointLabel����¼ÿ�����ص��״̬��0����δ��飬1�������ڼ��,2�����鲻�ϸ���Ҫ��ת��ɫ����3������ϸ������
	CheckMode��ֵΪ1����ȥ���׵㣬ֵΪ0����ȥ���ڵ�
	NeihborPos:��ά���飬��¼Χ��ÿ����������ص�ļ���λ������
	NeihborMode����鷶Χģ�ͣ�ֵΪ1��ʾ8����ֵΪ0��ʾ4����
	NeihborCount����¼�����С
	CurrX����̬��ʾͼ�����x���������ͼ�������
	CurrY����̬��ʾͼ�����x���������ͼ��������
	GrowBuffer����¼������ͨ���������С
	CheckResult����¼�������ֵΪ2��ʾ�������������ֵΪ1��ʾδ�������������
	δ���������������ͨ������ڽ��������Ϊ���ϸ�����
	���ӣ�
	Mat t1, t2;
	t2 = t1.clone();
	RemoveSmallRegion(t1, t2, 40, 0, 1);

	*/
	int RemoveCount = 0;
	//�½�һ����ǩͼ���ʼ��Ϊ0���ص㣬Ϊ�˼�¼ÿ�����ص����״̬�ı�ǩ��   
	Mat PointLabel = Mat::zeros(Src.size(), CV_8UC1);//��ʼ����ͼ��ȫ��Ϊ0��δ���
	if (CheckMode == 1)//ȥ��С��ͨ����İ�ɫ��  
	{
		//cout << "ȥ��С��ͨ��.";
		for (int i = 0; i < Src.rows; i++)
		{
			for (int j = 0; j < Src.cols; j++)
			{
				if (Src.at<uchar>(i, j) < 10)
				{
					PointLabel.at<uchar>(i, j) = 3;//���Ҷ�С��10�ĺ�ɫ����Ϊ�ϸ񣬱�ǩ3Ϊ��ȫ  
				}
			}
		}
	}
	else//ȥ���׶�����ɫ������  
	{

		for (int i = 0; i < Src.rows; i++)
		{
			for (int j = 0; j < Src.cols; j++)
			{
				if (Src.at<uchar>(i, j) > 10)
				{
					PointLabel.at<uchar>(i, j) = 3;//���ԭͼ�ǰ�ɫ���򣬱��Ϊ�ϸ�����Ϊ3  
				}
			}
		}
	}
	vector<Point2i>NeihborPos;//������ѹ������  
	NeihborPos.push_back(Point2i(-1, 0));
	NeihborPos.push_back(Point2i(1, 0));
	NeihborPos.push_back(Point2i(0, -1));
	NeihborPos.push_back(Point2i(0, 1));
	if (NeihborMode == 1)
	{
		//cout << "Neighbor mode: 8����." << endl;
		NeihborPos.push_back(Point2i(-1, -1));
		NeihborPos.push_back(Point2i(1, 1));
		NeihborPos.push_back(Point2i(1, -1));
		NeihborPos.push_back(Point2i(-1, 1));
	}
	//else cout << "Neighbor mode: 4����." << endl;
	int NeihborCount = 4 + 4 * NeihborMode;
	int CurrX = 0, CurrY = 0;
	//��ʼ���  
	for (int i = 0; i < Src.rows; i++)
	{
		for (int j = 0; j < Src.cols; j++)
		{
			if (PointLabel.at<uchar>(i, j) == 0)//��ǩͼ�����ص�Ϊ0����ʾ��δ���Ĳ��ϸ��  
			{   //��ʼ���  
				vector<Point2i>GrowBuffer;//��¼������ص�ĸ���  
				GrowBuffer.push_back(Point2i(j, i));
				PointLabel.at<uchar>(i, j) = 1;//���Ϊ���ڼ��  
				int CheckResult = 0;


				for (int z = 0; z < GrowBuffer.size(); z++)
				{
					for (int q = 0; q < NeihborCount; q++)
					{
						CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;//�б���j
						CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;//�б���i
						if (CurrX >= 0 && CurrX<Src.cols&&CurrY >= 0 && CurrY<Src.rows)//��ֹԽ��    
						{
							if (PointLabel.at<uchar>(CurrY, CurrX) == 0)
							{
								GrowBuffer.push_back(Point2i(CurrX, CurrY));//��������buffer �����������������˶�̬�仯GrowBuffer��С������ѭ������
								PointLabel.at<uchar>(CurrY, CurrX) = 1;     //���������ļ���ǩ�������ظ����    
							}
						}
					}
				}
				//cout << "GrowBuffer.size()=" << GrowBuffer.size()<<endl;
				if (mod == 1)
				{
					if (GrowBuffer.size() > AreaLimit) //�жϽ�����Ƿ񳬳��޶��Ĵ�С����1Ϊδ������2Ϊ����    
						CheckResult = 2;
					else
					{
						CheckResult = 1;
						RemoveCount++;//��¼�ж�������ȥ��  
					}
				}
				else
				{
					if (GrowBuffer.size() < AreaLimit) //�жϽ�����Ƿ񳬳��޶��Ĵ�С����1Ϊδ������2Ϊ����    
						CheckResult = 2;
					else
					{
						CheckResult = 1;
						RemoveCount++;//��¼�ж�������ȥ��  
					}
				}



				for (int z = 0; z < GrowBuffer.size(); z++)
				{
					CurrX = GrowBuffer.at(z).x;//�б���j
					CurrY = GrowBuffer.at(z).y;//�б���i
					PointLabel.at<uchar>(CurrY, CurrX) += CheckResult;//��ǲ��ϸ�����ص㣬����ֵΪ2  
				}



			}
		}
	}
	CheckMode = 255 * (1 - CheckMode);//��ʼ��ת�����С������    
	for (int i = 0; i < Src.rows; ++i)
	{
		for (int j = 0; j < Src.cols; ++j)
		{
			if (PointLabel.at<uchar>(i, j) == 2)
			{
				Dst.at<uchar>(i, j) = CheckMode;

			}
			else if (PointLabel.at<uchar>(i, j) == 3)
			{
				Dst.at<uchar>(i, j) = Src.at<uchar>(i, j);
			}
		}
	}

}

void mergeImg(Mat &dst, Mat &src1, Mat &src2)
{

	//int rows = src1.rows > src2.rows ? src1.rows : src2.rows;//�ϳ�ͼ�������

	//int cols = src1.cols + 10 + src2.cols;  //�ϳ�ͼ�������

	int cols = src1.cols > src2.cols ? src1.cols : src2.cols;//�ϳ�ͼ�������

	int rows = src1.rows + 10 + src2.rows;  //�ϳ�ͼ�������

											//CV_Assert(src1.type() == src2.type());

	cv::Mat zeroMat = cv::Mat::ones(rows, cols, src1.type());
	zeroMat = zeroMat.mul(255);
	zeroMat.copyTo(dst);

	src1.copyTo(dst(cv::Rect(0, 0, src1.cols, src1.rows)));

	//src2.copyTo(dst(cv::Rect(src1.cols + 10, 0, src2.cols, src2.rows)));//����ͼ��֮�����20������
	src2.copyTo(dst(cv::Rect(0, src1.rows + 10, src2.cols, src2.rows)));


}



Mat surface_detect(Mat &color)
{
	Mat oringinImage;//8U
	cvtColor(color, oringinImage, CV_RGB2GRAY);
	int m = oringinImage.rows;
	int n = oringinImage.cols;

	Mat reducedImage = oringinImage.clone();
	preProcessImage(oringinImage, reducedImage);


	Mat logLine = reducedImage.clone();
	lines_gauss(reducedImage, logLine, 9, 25, 20);//8U
												  //lines_gauss(reducedImage, logLine, 5, 20, 18);
												  //removelogo(color, logLine);

	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));//�����ʵ���С
	Mat dilatedImage;
	dilate(logLine, dilatedImage, element);

	Mat selectedImage = dilatedImage.clone();
	RemoveSmallRegion(dilatedImage, selectedImage, 80, 1, 1, 1);

	return selectedImage;
}
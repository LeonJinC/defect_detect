

#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>

#include "Kalman.h"
#include "Hungarian.h"

using namespace cv;
using namespace std;

class CTrack
{
public:
	vector<Point2d> trace;
	static size_t NextTrackID;
	size_t track_id;
	size_t skipped_frames; 
	Point2f prediction;
	TKalmanFilter* KF;
	CTrack(Point2f p, float dt, float Accel_noise_mag);
	~CTrack();
};


class CTracker
{
public:
	float dt;//1 
	float Accel_noise_mag;//2
	double dist_thres;//3
	int maximum_allowed_skipped_frames;//4
	int max_trace_length;//5
	vector<CTrack*> tracks;
	void Update(vector<Point2f>& detections);
	CTracker
	(
		float _dt,
		float _Accel_noise_mag,
		double _dist_thres=60, 
		int _maximum_allowed_skipped_frames=10,
		int _max_trace_length=10
	);
	~CTracker(void);
};


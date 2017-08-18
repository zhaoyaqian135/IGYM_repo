#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <pthread.h>

using namespace cv;
using namespace std;

//static parameter
const int R = 255;
const int G = 0;
const int B = 255;
const int WAITTIME = 1; // ms
const int MAXOBJ = 10;
const int input_width = 1280;
const int input_height = 720;
const int disp_width = 600*2;
const int disp_height = 800;
const int disp_offset = 2;  // line offset
const int img_width = 670;
const int img_height = 500;
const int marginal_ws = 79;
const int marginal_hs = 79;
const int marginal_we = 70;
const int marginal_he = 70;
const int bound_ws = marginal_ws;
const int bound_we = img_width - marginal_we;
const int bound_hs = marginal_hs;
const int bound_he = img_height - marginal_he;
const int bound_offset = 3;
const int kick_tolerance = 5;
int img_center_x = 328;
int img_center_y = 227;
const float ratio_x = ((float)disp_width)/((float)(img_width - marginal_ws - marginal_we));
const float ratio_y = ((float)disp_height)/((float)(img_height - marginal_hs- marginal_he));
int dilateSize = 5;
int erosionSize = 7;//if noise large, should larger
int dilateSize2 = 13;//if person wear white, should larger
int thresholdvalue = 60;//if person wear white, should lower
int perspective_x = 17; //percent
int perspective_y = 17; //percent
int perspective_r = 10; //percent
const float radius_speed = 1;
const float person_speed_ratio = 1;
const float max_speed = 15;
const float min_speed = 2;
int radiusmin = 25;
const int radiusmax = 150;

// oscillation correction parameter //yaqian
int radius_threshold = 100;
int center_threshold = 110;

// pred filter parameter //yaqian
int ratio_predFilter_x = 0;
int ratio_predFilter_y = 0;
int ratio_predFilter_r = 0;

//friction parameter
const float FRICTION_COEFF = 0.03;

//mutex parameter
pthread_mutex_t framelocker;
pthread_cond_t framecond;
bool frame_buffer_prepared = 0;
int irets[2];

//camera input
VideoCapture cap;
Mat frame;

//Calibrate parameter
Mat remap_image;
Mat rmap[2][2];
int x, y, width, height;

int frame_number = 0;
float person_sx, person_sy;
Mat result;
Mat black, output1, output2;
int object_init_freeze = 0;

void friction (float &x_speed, float &y_speed);
void readIntrinsics(string filename, Mat cameraMatrix[2], Mat distCoeffs[2]);
void readExtrinsics(string filename, Mat& R, Mat& T, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q);
void StereoCalib(Size imageSize, Rect validRoi[2], Mat rmap[2][2]);
void rotate(Mat& src, double angle, Mat &dst);
void *UpdateFrame(void *);
void speedminmaxcontrol(float &ball_sx, float &ball_sy);
void detect(Mat& crop, Mat& background, Mat& result);
void perspectivetransform(Point2f& center, float& radius);
void speedcontrol(const Point2f& centerpre, const float& radiuspre, const Point2f& center, const float& radius, float& ball_x, float& ball_y, float& ball_radius, float& ball_sx, float& ball_sy);
float norm(float x, float y);
void splitDisp(Mat& input, Mat& output1, Mat& output2);

int main()
{
	//Initialization of bouncing ball
	float ball_x = img_center_x;
	float ball_y = img_center_y;
	float ball_radius = 12;
	float ball_x_speed = 0.0f;
	float ball_y_speed = min_speed;

	//player score
	int player_1 = 0;
	int player_2 = 0;
	int score_flag = 0;

	//person data
	vector<Point2f> center_current(MAXOBJ);
	vector<Point2f> center_previous(MAXOBJ);
	vector<float> radius_current(MAXOBJ);
	vector<float> radius_previous(MAXOBJ);
	vector<float> radius_predicted(MAXOBJ,0); //yaqian's change: prediction filter
	const int AVGFILTERSIZE = 4;
	int ring_idx = 0;
	float ring_center_x[MAXOBJ][AVGFILTERSIZE];
	float ring_center_y[MAXOBJ][AVGFILTERSIZE];
	float ring_radius[MAXOBJ][AVGFILTERSIZE];
	for (int i = 0; i < MAXOBJ; i++)
		for (int j = 0; j < AVGFILTERSIZE; j++)
		{
			ring_center_x[i][j] = 0;
			ring_center_y[i][j] = 0;
			ring_radius[i][j] = 0;
		}

	int object_number = 0;

	//Window setup
	namedWindow("display", WINDOW_NORMAL);
	moveWindow("display", 0, 100);
	//setWindowProperty("display", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	namedWindow("control", WINDOW_NORMAL);
	moveWindow("control", 0, 200);
	createTrackbar("threshold","control", &thresholdvalue, 255);
	//createTrackbar("dilateSize","control", &dilateSize, 20);
	//createTrackbar("erosionSize","control", &erosionSize, 20);
	//createTrackbar("dilateSize2","control", &dilateSize2, 20);
	//createTrackbar("radiusmin","control", &radiusmin, (int)radiusmax);
	//createTrackbar("perspective_x %","control", &perspective_x, 100);
	//createTrackbar("perspective_y %","control", &perspective_y, 100);
	//createTrackbar("perspective_r %","control", &perspective_r, 100);
	//createTrackbar("imgctr_x","control", &img_center_x, img_width);
	//createTrackbar("imgctr_y","control", &img_center_y, img_height);
	createTrackbar("radius_thresh","control", &radius_threshold, 1000);
	createTrackbar("center_thresh","control", &center_threshold, 1000);
	createTrackbar("pred_x","control", &ratio_predFilter_x, 200);
	createTrackbar("pred_y","control", &ratio_predFilter_y, 200);
	createTrackbar("pred_r","control", &ratio_predFilter_r, 200);
	
	/*ofstream myfile;
	myfile.open("object_track.txt");
	myfile << "radius center_x center_y"<<endl;*/
	
	namedWindow("display1", WINDOW_NORMAL);
	moveWindow("display1", 0, 800);
	setWindowProperty("display1", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	namedWindow("display2", WINDOW_NORMAL);
	moveWindow("display2", 0, 1600);
	setWindowProperty("display2", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	
	//Input
	cap.open(1);
	cap.set(CAP_PROP_FRAME_WIDTH, input_width);
	cap.set(CAP_PROP_FRAME_HEIGHT, input_height);
	
	//Calibration
	Size imageSize(input_width, input_height); 
	Rect validRoi[2];
	StereoCalib(imageSize, validRoi, rmap);
	x = MAX(validRoi[0].x, validRoi[1].x);
	y = MAX(validRoi[0].y, validRoi[1].y);
	width = MIN(validRoi[0].x+validRoi[0].width, validRoi[1].x+validRoi[1].width) - x;
	height = MIN(validRoi[0].y+validRoi[0].height, validRoi[1].y+validRoi[1].height) - y;
	cout << "After Calibration: x:"<<x<<" y:"<<y<<" width:"<<width<<" height:"<<height<<endl;
	
	//mutex parallel initial
	framelocker = PTHREAD_MUTEX_INITIALIZER;
	framecond = PTHREAD_COND_INITIALIZER;
	frame_buffer_prepared = 0;
	pthread_t threads_1;
	irets[1] = pthread_create(&threads_1,NULL,&UpdateFrame,NULL);

	//image data
	Mat background=imread("./image/current.png"); 
	Mat result;
	Mat input;

	//projector initial
	black = imread("black.png");
	resize(black, black, Size(disp_width, disp_height), INTER_LINEAR);
	flip(black,black,1);
	Mat blackraw = black.clone();
	splitDisp(black, output1, output2);
	imshow("display1", output1);
	imshow("display2", output2);
	waitKey(WAITTIME);
	
	cout << ratio_x << " " << ratio_y <<endl;
	cout<<"Entering loop!"<<endl;
	
	time_t start,end;
	time(&start);
	while(1)
	{
		//update frame from mutex
		pthread_mutex_lock(&framelocker);
		input = remap_image.clone();
		pthread_mutex_unlock(&framelocker);
		
		if(!input.empty())
		{
			frame_number ++;
			if (frame_number < 100) continue;
			//if (frame_number % 10 == 0) cout << "\n Frame No."<< frame_number<<"----->";
			//canvas update
			black = blackraw.clone();
			
			//detection
			detect(input, background, result);

			//find contour
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			findContours(result, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
			vector<vector<Point> > contours_poly(contours.size());
			vector<Point2f>center(contours.size());
			vector<float>radius(contours.size());
			for(int i = 0; i < contours.size(); i++ )
			{ 
				approxPolyDP(Mat(contours[i]), contours_poly[i], 0, true );
		       		minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i] );
			}
		
			//perspective correct
			for(int	i = 0; i < contours.size(); i++)
			{
				perspectivetransform(center[i], radius[i]);
			}
			
			int contour_number = 0;
			for(int i = 0; i < contours.size(); i++)
			{	
				if(radius[i]>radiusmin && radius[i]<radiusmax) // count how many contours can we use
					contour_number++;
			}

			//get the inital player position and parameter
			if (contour_number != object_number)
			//if(object_init_freeze != 0 || contour_number != object_number)
			{
				//if (object_init_freeze == 0) object_init_freeze = 10;
				//object_init_freeze--;
				object_number = 0;
				for(int	i = 0; i < contours.size(); i++)
				{
					if(radius[i]>radiusmin && radius[i]<radiusmax){
						center_previous[object_number] = center_current[object_number] = center[i];
						radius_previous[object_number] = radius_current[object_number] = radius[i];
						object_number++;
					}
				}
				// initialize ring average filter
				//if (object_init_freeze == 1)
				{
					for (int i = 0; i < object_number; i++)
						for(int j = 0; j < AVGFILTERSIZE; j++)
						{
							ring_center_x[i][j] = center_current[i].x;
							ring_center_y[i][j] = center_current[i].y;
							ring_radius[i][j] = radius_current[i];
						}
					ring_idx = 0;
				}
				
			}

			//player position update
			else
			{
				//get current one at right order
				for(int i = 0; i < object_number; i++)
				{
					for(int j = 0; j < contours.size(); j++)
					{
						//yaqian's changing: oscllation_correction_threshold;
						if((radius[j]>radiusmin && radius[j]<radiusmax) && (norm(center_current[i].x-center[j].x, center_current[i].y-center[j].y) < radius_current[i]))
						{	
							bool flag_oscil_center = (abs(center[j].x-center_current[i].x) > center_threshold/100.0)
													| (abs(center[j].y-center_current[i].y) > center_threshold/100.0);
							//cout << "flag_oscil_center: " << flag_oscil_center<<endl;
							if (flag_oscil_center)
							{
								center_previous[i] = center_current[i];
								center_current[i] = center[j];
							}
							bool flag_oscil_radius = abs(radius[j] - radius_current[i]) > radius_threshold/100.0;
							//cout<<"flag_oscil_radius: "<< flag_oscil_radius << endl;
							if ( flag_oscil_radius )
							{
								radius_previous[i] = radius_current[i];
								radius_current[i] = radius[j];
							}
							break;
						}
					}
				}
				
				// do filtering on person position and radius
				for (int i = 0; i < object_number; i++)
				{
					ring_idx = (ring_idx + 1) % AVGFILTERSIZE;
					float temp;
					temp = center_current[i].x;
					float x_backward_average = (center_previous[i].x * AVGFILTERSIZE - ring_center_x[i][ring_idx] + temp) / AVGFILTERSIZE;
					center_current[i].x = (temp - x_backward_average) * ratio_predFilter_x/100.0  + temp;
					
					
					ring_center_x[i][ring_idx] = temp;
					temp = center_current[i].y;
					float y_backward_average = (center_previous[i].y * AVGFILTERSIZE - ring_center_y[i][ring_idx] + temp) / AVGFILTERSIZE;
					center_current[i].y = (temp - y_backward_average) * ratio_predFilter_y/100.0  + temp;
					
					
					ring_center_y[i][ring_idx] = temp;
					temp = radius_current[i];
					float r_backward_average = (radius_previous[i] * AVGFILTERSIZE - ring_radius[i][ring_idx] + temp) / AVGFILTERSIZE;
					radius_current[i] = (temp - r_backward_average) * ratio_predFilter_r/100.0  + temp;
					
					ring_radius[i][ring_idx] = temp;
				}
			}
			
			//draw player
			for(int i = 0; i < object_number; i++)
			{	
				//yaqian's change: prediction filter, |old_predicted_radius ||middle |current_radius ||new_predicted_radius
				//circle(black, Point((center_current[i].x-marginal_ws)*ratio_x, (center_current[i].y-marginal_hs)*ratio_y), (radius_predicted[i] + radius_current[i])/2*ratio_y, Scalar(255,255,255), 2, 8, 0);
				circle(black, Point((center_current[i].x-marginal_ws)*ratio_x, (center_current[i].y-marginal_hs)*ratio_y), radius_current[i]*ratio_y, Scalar(R,G,B), 2, 8, 0);
				//myfile << radius_current[i] << " " << center_current[i].x << " " << center_current[i].y<<endl;
				//radius_predicted[i] = (radius_current[i] + (radius_current[i] - radius_predicted[i])/2);
				//circle(black, Point((center_current[i].x-marginal_ws)*ratio_x, (center_current[i].y-marginal_hs)*ratio_y), (radius_predicted[i])*ratio_y, Scalar(255,255,255), 2, 8, 0);
			}
			
			// ----------------------------- object detection done. deal with ball-object interaction --------------------------
			//bouncing position correct
			for(int i = 0; i < object_number; i++){
				float xd = ball_x - center_current[i].x;
				float yd = ball_y - center_current[i].y;
				float distance = norm(xd, yd); if (distance == 0) continue;
				if (distance < (radius_current[i] + ball_radius - kick_tolerance) ) // ball is inside object circle, needs correction
				{
					float cosc = xd/distance;
					float sinc = yd/distance;
					ball_x = center_current[i].x + (radius_current[i] + ball_radius - kick_tolerance) * cosc;
					ball_y = center_current[i].y + (radius_current[i] + ball_radius - kick_tolerance) * sinc;
				}
			}
			
			//ball speed update
			for(int i = 0; i < object_number; i++)
			{
				speedcontrol(center_previous[i], radius_previous[i], center_current[i], radius_current[i], 
							 ball_x, ball_y, ball_radius, ball_x_speed, ball_y_speed);
			}
			friction(ball_x_speed, ball_y_speed);
			
			//boundary position correct
			if (ball_x + ball_radius > bound_we - bound_offset) ball_x = - ball_radius + bound_we - bound_offset;
			else if (ball_x - ball_radius <= bound_ws + bound_offset) ball_x = ball_radius + bound_ws + bound_offset;
			else if (ball_y + ball_radius >= bound_he - bound_offset) ball_y = - ball_radius + bound_he - bound_offset;
			else if (ball_y - ball_radius <= bound_hs + bound_offset) ball_y = ball_radius + bound_hs + bound_offset;
			
			//boundary bouncing
			if( (ball_x + ball_radius >= bound_we - bound_offset) || (ball_x - ball_radius <= bound_ws + bound_offset) )
			{
				ball_x_speed = -ball_x_speed;
				// count score
				if (ball_x + ball_radius >= bound_we - bound_offset)
				{
					if (score_flag == 0)
					{
						score_flag = 1;
						player_1 ++; 
					}
				}
				else if (ball_x - ball_radius <= bound_ws + bound_offset)
				{
					if (score_flag == 0)
					{
						score_flag = 1;
						player_2 ++; 
					}
				}
			}
			else score_flag = 0;
			
			if( (ball_y + ball_radius >= bound_he - bound_offset) || (ball_y - ball_radius <= bound_hs + bound_offset) )
			{
				ball_y_speed = -ball_y_speed;
			}
			
			//draw ball
			circle(black, Point((int)((ball_x-marginal_ws)*ratio_x), (int)((ball_y-marginal_hs)*ratio_y)), ball_radius*ratio_y, Scalar(R,G,B),-1,8,0);
			
			//update ball position
			//speedminmaxcontrol(ball_x_speed, ball_y_speed);
			ball_x += ball_x_speed;
			ball_y += ball_y_speed;
			
			if (frame_number % 20 == 0) 
			{
				cout << "Ball coordinate: x: "<< ball_x << ", y: " << ball_y
					 <<". Ball speed: x: "<<ball_x_speed<<", y: "<<ball_y_speed<<". "<<endl;
			}
			
			//draw boundary
			int line_offset = 2;
			line(black, Point(line_offset,line_offset),Point(line_offset, disp_height - line_offset), Scalar(R,G,B),5,8,0);
			line(black, Point(line_offset,line_offset),Point(disp_width - line_offset,line_offset), Scalar(R,G,B),5,8,0);
			line(black, Point(disp_width - line_offset,line_offset),Point(disp_width - line_offset,disp_height - line_offset), Scalar(R,G,B),5,8,0);
			line(black, Point(line_offset,disp_height - line_offset),Point(disp_width - line_offset,disp_height - line_offset), Scalar(R,G,B),5,8,0);	
		
			//score display
			rotate(black, 90, black);
			putText(black, format("%d", player_1), Point(0,170), CV_FONT_HERSHEY_PLAIN, 10.0, cvScalar(R,G,B), 5, CV_AA);
			rotate(black, 180, black);
			putText(black, format("%d", player_2), Point(0,170), CV_FONT_HERSHEY_PLAIN, 10.0, cvScalar(R,G,B), 5, CV_AA);
			rotate(black, 90, black);

			//result display
			flip(black,black,-1);
			splitDisp(black, output1, output2);
			imshow("display1", output1);
			imshow("display2", output2);
			if(waitKey(WAITTIME)==27) break;
			
			if(frame_number % 600 == 0)
			{
				time(&end);
				double seconds = difftime(end,start);
				double fps = 600/seconds;
				cout<<"fps: "<<fps<<endl;
				time(&start);
			}
		}
		
	}
	
	return 0;						
}

void friction (float &x_speed, float &y_speed)
{
	float length = norm(x_speed, y_speed);
	if (!length || length < min_speed || length<= FRICTION_COEFF)
	{
	}
	else
	{
		int x_sign, y_sign;
		if (x_speed > 0) x_sign = 1; else x_sign = -1;
		if (y_speed > 0) y_sign = 1; else y_sign = -1;
		x_speed = ((float)x_sign)*(abs(x_speed) - abs(x_speed)/(float)length*FRICTION_COEFF);
		y_speed = ((float)y_sign)*(abs(y_speed) - abs(y_speed)/(float)length*FRICTION_COEFF);
	}
	
	speedminmaxcontrol(x_speed, y_speed);
}

void speedminmaxcontrol(float &ball_sx, float &ball_sy)
{
	// speed min max control
	float ball_speed = norm(ball_sx, ball_sy);
	cout << "using speed minmax control: " << ball_sx << " "<<ball_sy<<endl;
	if (!ball_speed)
	{
		ball_sx = ball_sy = min_speed / sqrt(2);
		cout << "speed zero!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! "<<endl;
	}
	else if (ball_speed > max_speed)
	{
		ball_sx = ball_sx / ball_speed * max_speed;
		ball_sy = ball_sy / ball_speed * max_speed;
	}
	else if (ball_speed < min_speed)
	{
		ball_sx = ball_sx / ball_speed * min_speed;
		ball_sy = ball_sy / ball_speed * min_speed;
	}
}

void readIntrinsics(string filename, Mat cameraMatrix[2], Mat distCoeffs[2])
{
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        cout << filename << " open failed!" << endl;

    FileNode n = fs["M1"];
    n >> cameraMatrix[0];
    n = fs["D1"];
    n >> distCoeffs[0];
    n = fs["M2"];
    n >> cameraMatrix[1];
    n = fs["D2"];
    n >> distCoeffs[1];
    fs.release();
}

void readExtrinsics(string filename, Mat& R, Mat& T, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q)
{
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        cout << filename << " open failed!" << endl;
    
    FileNode n = fs["R"];
    n >> R;
    n = fs["T"];
    n >> T;
    n = fs["R1"];
    n >> R1;
    n = fs["R2"];
    n >> R2;
    n = fs["P1"];
    n >> P1;
    n = fs["P2"];
    n >> P2;
    n = fs["Q"];
    n >> Q;
    
    fs.release();
   
}

void StereoCalib(Size imageSize, Rect validRoi[2], Mat rmap[2][2])
{   
    int i, j, k;
    Mat cameraMatrix[2], distCoeffs[2];
    string fsIn = "intrinsics.yml";

    readIntrinsics(fsIn, cameraMatrix, distCoeffs);
    
    Mat R, T, R1, R2, P1, P2, Q;
    string fsEx = "extrinsics.yml";
    readExtrinsics(fsEx, R, T, R1, R2, P1, P2, Q);
    stereoRectify(cameraMatrix[0], distCoeffs[0],
                  cameraMatrix[1], distCoeffs[1],
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

}

void rotate(Mat& src, double angle, Mat &dst)
{
	if(angle == 90 || angle == -270){
		transpose(src,dst);
		flip(dst,dst,1);
	}
	if(angle == -90 || angle == 270){
		transpose(src,dst);
		flip(dst,dst,0);
	}
	if(angle == 180 || angle == -180){
		transpose(src,dst);
		flip(dst,dst,1);
		transpose(src,dst);
		flip(dst,dst,1);
	}
}

void *UpdateFrame(void *)//another thread to get input
{
	while(1){
		cap >> frame;
		Mat left(frame, Range::all(), Range(0, 1280));
		remap(left, left, rmap[0][0], rmap[0][1], INTER_LINEAR);
		//Mat calibrate(left, Range(y, y+height),Range(x,x+width));
		int cut_start_x = 310;
		int cut_start_y = 120;
		Mat current(left, Rect(cut_start_x, cut_start_y, img_width, img_height));

		pthread_mutex_lock(&framelocker);
		remap_image = current.clone();
		pthread_mutex_unlock(&framelocker);
		
	}
}

void detect(Mat& crop, Mat& background, Mat& result)
{
	absdiff(crop, background, result);
	cvtColor(result, result, CV_BGR2GRAY);
	threshold(result, result, thresholdvalue, 255, THRESH_BINARY);
	//medianBlur(result,result,5);
	Mat elementDilate = getStructuringElement(MORPH_ELLIPSE,
		Size(2*dilateSize + 1, 2*dilateSize+1),
		Point(dilateSize, dilateSize));
	dilate(result, result, elementDilate);
	Mat erosionElement = getStructuringElement(MORPH_ELLIPSE,
		Size(2*erosionSize+1, 2*erosionSize+1),
		Point(erosionSize, erosionSize));
	erode(result, result, erosionElement);
//	imwrite("1.png", result);
	Mat elementDilate2 = getStructuringElement(MORPH_ELLIPSE,
		Size(2*dilateSize2 + 1, 2*dilateSize2+1),
		Point(dilateSize2, dilateSize2));
	dilate(result, result, elementDilate2);
	
	imshow("display", result); waitKey(WAITTIME);
}

void perspectivetransform(Point2f& center, float& radius)
{
	center.x = (img_center_x - center.x)*perspective_x / 100 + center.x;
	center.y = (img_center_y - center.y)*perspective_y /100+ center.y;
	radius = radius - perspective_r/100.0*sqrt((img_center_x - center.x)*(img_center_x - center.x) + (img_center_y - center.y)*(img_center_y - center.y));
	if(radius <=0) radius = 1;
}


void speedcontrol(const Point2f& centerpre, const float& radiuspre, const Point2f& center, const float& radius, float& ball_x, float& ball_y, float& ball_radius, float& ball_sx, float& ball_sy)
{
	// person shift speed
	person_sx = (center.x - centerpre.x) * person_speed_ratio;
	person_sy = (center.y - centerpre.y) * person_speed_ratio;
	
	// obj->ball vector
	float xd = ball_x - center.x;
	float yd = ball_y - center.y;
	float distance = norm(xd, yd);
	if (distance == 0) return;
	
	// person dilate speed
	float dilate_sx, dilate_sy;
	if(radius-radiuspre>0)
	{
		float tmp = (radius-radiuspre)/radiuspre * radius_speed;
		dilate_sx = tmp / distance * xd;
		dilate_sy = tmp / distance * yd;
	}
	
	if(distance <= radius + ball_radius) // ball is kicked
	{
		float ball_speed = norm(ball_sx, ball_sy);
		float cosangle = (person_sx*xd+person_sy*yd); // angle between person vector and ball-obj vector
		if(!ball_speed)
		{//when ball is not moving
			ball_sx = person_sx + dilate_sx;
			ball_sy = person_sy + dilate_sy;
		}
		else
		{
			float ball_reflect_sx, ball_reflect_sy;
			float cosa = (ball_sx*xd+ball_sy*yd)/distance/ball_speed;
			float sina = sqrt(1-cosa*cosa);
			//counter-clockwise rotate ball speed vector and normalize it
			float xcounter = (ball_sx*cosa - ball_sy*sina)/ball_speed;
			float ycounter = (ball_sx*sina + ball_sy*cosa)/ball_speed;
			float diffcounter = sqrt((xcounter-xd/distance)*(xcounter-xd/distance)+(ycounter-yd/distance)*(ycounter-yd/distance));
		
			//clockwise rotate ball speed vector and normalize it
			float xclock = (ball_sx*cosa + ball_sy*sina)/ball_speed;
			float yclock = (-ball_sx*sina + ball_sy*cosa)/ball_speed;
			float diffclock = sqrt((xclock - xd/distance)*(xclock - xd/distance)+(yclock - yd/distance)*(yclock - yd/distance));
			
			//rotate angle
			float cos2a = sina*sina-cosa*cosa;
			float sin2a = -2*sina*cosa;
		
			if(diffclock < diffcounter)
			{//should rotate clockwise
				ball_reflect_sx = ball_sx*cos2a + ball_sy*sin2a;
				ball_reflect_sy = -ball_sx*sin2a + ball_sy*cos2a;
			}
			else
			{//should rotate counter-clockwise
				ball_reflect_sx = ball_sx*cos2a - ball_sy*sin2a;
				ball_reflect_sy = ball_sx*sin2a + ball_sy*cos2a;
			}
			
			// ------------------end computing reflection velocity---------------------------
			if(cosangle >= 0)
			{//when ball hit face by face
				ball_sx = person_sx + ball_reflect_sx + dilate_sx; 
				ball_sy = person_sy + ball_reflect_sy + dilate_sy;
			}
			else
			{
				ball_sx = ball_reflect_sx + dilate_sx;
				ball_sy = ball_reflect_sy + dilate_sy;
			}
		}
		
		//----------------- end updating ball speed, enter minmax control----------------------
		
		// speed min max control
		speedminmaxcontrol(ball_sx, ball_sy);
	}
}			

float norm(float x, float y)
{
	float ans = sqrt(x * x + y * y);
	return ans;
}

void splitDisp(Mat& input, Mat& output1, Mat& output2)
{
	output1 = Mat(input, Rect(0, 0, disp_width/2, disp_height-1));
	rotate(output1,90,output1);
	output2 = Mat(input, Rect(disp_width/2, 0, disp_width/2, disp_height));
	rotate(output2,90,output2);
}


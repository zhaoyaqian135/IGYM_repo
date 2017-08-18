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
const int WAITTIME = 15; // ms
const int MAXOBJ = 10;
const int disp_width = 800;
const int disp_height = 600;
const int disp_offset = 4;
const int img_width = 380;
const int img_height = 288;
const int marginal_ws = 20;
const int marginal_hs = 20;
const int marginal_we = img_width - marginal_ws;
const int marginal_he = img_height - marginal_hs;
const float img_center_x = img_width/2.;
const float img_center_y = img_height/2.;
const float ratio_x = disp_width/(img_width - 2*marginal_ws);
const float ratio_y = disp_height/(img_height - 2*marginal_hs);
const float dilateSize = 1;
const float erosionSize = 1;//if noise large, should larger
const float dilateSize2 = 9;//if person wear white, should larger
const float thresholdvalue = 80;//if person wear white, should lower
const float perspective_x = 0.05;
const float perspective_y = 0.05;
const float perspective_r = 0.03;
const float radius_speed = 3;
const float person_speed_ratio = 3;
const float max_speed = 10;
const float min_speed = 2;
const float radiusmin = 1.0;
const float radiusmax = 150.0;

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
Mat black;
int object_init_freeze = 0;

void friction (float &x_speed, float &y_speed);
void readIntrinsics(string filename, Mat cameraMatrix[2], Mat distCoeffs[2]);
void readExtrinsics(string filename, Mat& R, Mat& T, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q);
void StereoCalib(Size imageSize, Rect validRoi[2], Mat rmap[2][2]);
void rotate(Mat& src, double angle, Mat &dst);
void *UpdateFrame(void *);
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
	const int AVGFILTERSIZE = 5;
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
	moveWindow("display", 20, 20);
	//setWindowProperty("display", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	namedWindow("control", WINDOW_NORMAL);
	moveWindow("display", 0, 0);
	//namedWindow("detect", WINDOW_NORMAL);
	//moveWindow("detect", 1400, 0);
	//setWindowProperty("detect", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	
	//Input
	cap.open("video_input_good.avi");
	
	//image data
	Mat background=imread("./image/current.png"); 
	Mat input;

	//projector initial
	black = imread("black.png");
	resize(black, black, Size(disp_width, disp_height), INTER_LINEAR);
	flip(black,black,1);
	Mat blackraw = black.clone();
	imshow("display",black);
	waitKey(1);
	
	// Video output initial
	//VideoWriter video("simulation_with_filter.avi", CV_FOURCC('M','J','P','G'), 40, Size(800, 600));
	
	for (int i = 0; i < 60 * 23; i++)
		cap >> input;
	
	waitKey(1);
	time_t start,end;
	time(&start);
	while(1)
	{
		//update frame from mutex
		cap >> input;
		
		if(!input.empty())
		{
			frame_number ++;
			if (frame_number % 10 == 0) cout << "\n Frame No."<< frame_number<<"----->";
			//canvas update
			//resize(input, black, Size(disp_width, disp_height), INTER_LINEAR);
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
			if(object_init_freeze != 0 || contour_number != object_number)
			{
				if (object_init_freeze == 0) object_init_freeze = 10;
				object_init_freeze--;
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
				if (object_init_freeze == 1)
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
						if((radius[j]>radiusmin && radius[j]<radiusmax) && (norm(center_current[i].x-center[j].x, center_current[i].y-center[j].y) < radius_current[i]))
						{	
							center_previous[i] = center_current[i];
							radius_previous[i] = radius_current[i];
							center_current[i] = center[j];
							radius_current[i] = radius[j];
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
					center_current[i].x = (center_previous[i].x * AVGFILTERSIZE - ring_center_x[i][ring_idx] + temp) / AVGFILTERSIZE;
					ring_center_x[i][ring_idx] = temp;
					temp = center_current[i].y;
					center_current[i].y = (center_previous[i].y * AVGFILTERSIZE - ring_center_y[i][ring_idx] + temp) / AVGFILTERSIZE;
					ring_center_y[i][ring_idx] = temp;
					temp = radius_current[i];
					radius_current[i] = (radius_previous[i] * AVGFILTERSIZE - ring_radius[i][ring_idx] + temp) / AVGFILTERSIZE;
					ring_radius[i][ring_idx] = temp;
				}
			}
			//draw player
			for(int i = 0; i < object_number; i++)
			{	
				circle(black, Point((center_current[i].x-marginal_ws)*ratio_x, (center_current[i].y-marginal_hs)*ratio_y), radius_current[i]*ratio_y, Scalar(255,255,255), 2, 8, 0);
			}
			
			// ----------------------------- object detection done. deal with ball-object interaction --------------------------
			//bouncing position correct
			for(int i = 0; i < object_number; i++){
				float xd = ball_x - center_current[i].x;
				float yd = ball_y - center_current[i].y;
				float distance = norm(xd, yd); if (distance == 0) continue;
				if (distance < (radius_current[i] + ball_radius) ) // ball is inside object circle, needs correction
				{
					float cosc = xd/distance;
					float sinc = yd/distance;
					ball_x = center_current[i].x + (radius_current[i] + ball_radius) * cosc;
					ball_y = center_current[i].y + (radius_current[i] + ball_radius) * sinc;
				}
			}
			
			//ball speed update
			for(int i = 0; i < object_number; i++)
			{
				if (frame_number < 100) break;
				speedcontrol(center_previous[i], radius_previous[i], center_current[i], radius_current[i], 
							 ball_x, ball_y, ball_radius, ball_x_speed, ball_y_speed);
			}
			friction(ball_x_speed, ball_y_speed);
			
			//boundary bouncing
			if( (ball_x - marginal_ws + ball_radius)*ratio_x >= disp_width-disp_offset || (ball_x - marginal_ws - ball_radius)*ratio_x <= disp_offset)
			{
				ball_x_speed = -ball_x_speed;
				// count score
				if( (ball_x - marginal_ws + ball_radius)*ratio_x >= disp_width-disp_offset )
				{
					if (score_flag == 0)
					{
						score_flag = 1;
						player_1 ++; 
					}
				}
				else if( (ball_x - marginal_ws - ball_radius)*ratio_x <= disp_offset) 
				{
					
					if (score_flag == 0)
					{
						score_flag = 1;
						player_2 ++; 
					}
				}
			}
			else score_flag = 0;
			if( (ball_y - marginal_hs + ball_radius)*ratio_y >= disp_height-disp_offset || (ball_y - marginal_hs - ball_radius)*ratio_y <= disp_offset)
			{
				ball_y_speed = -ball_y_speed;
			}
			
			//update ball position
			ball_x += ball_x_speed;
			ball_y += ball_y_speed;

			//boundary position correct
			if( (ball_x - marginal_ws + ball_radius)*ratio_x > disp_width-disp_offset) ball_x = (disp_width-disp_offset)/ratio_x + marginal_ws - ball_radius;
			if( (ball_x - marginal_ws - ball_radius)*ratio_x < disp_offset) ball_x = disp_offset/ratio_x + marginal_ws + ball_radius;
			if( (ball_y-marginal_hs+ball_radius)*ratio_y > disp_height-disp_offset) ball_y = (disp_height-disp_offset)/ratio_y + marginal_hs - ball_radius;
			if( (ball_y-marginal_hs-ball_radius)*ratio_y < disp_offset) ball_y = disp_offset/ratio_y + marginal_hs + ball_radius;
			
			//draw boundary
			line(black, Point(0,0),Point(0,disp_height - 1), Scalar(255,255,255),5,8,0);
			line(black, Point(0,0),Point(disp_width - 1,0), Scalar(255,255,255),5,8,0);
			line(black, Point(disp_width - 1,0),Point(disp_width - 1,disp_height - 1), Scalar(255,255,255),5,8,0);
			line(black, Point(0,disp_height - 1),Point(disp_width - 1,disp_height - 1), Scalar(255,255,255),5,8,0);

			//draw ball
			circle(black, Point((ball_x-marginal_ws)*ratio_x, (ball_y-marginal_hs)*ratio_y), ball_radius*ratio_y, Scalar(255,255,255),-1,8,0);		
		
			//score display
			rotate(black, 90, black);
			putText(black, format("%d", player_1), Point(0,170), CV_FONT_HERSHEY_PLAIN, 10.0, cvScalar(255,255,255), 5, CV_AA);
			rotate(black, 180, black);
			putText(black, format("%d", player_2), Point(0,170), CV_FONT_HERSHEY_PLAIN, 10.0, cvScalar(255,255,255), 5, CV_AA);
			rotate(black, 90, black);

			//result display
			//flip(result, result, 1);
			//imshow("detect", result);
			//waitKey(1);
			//flip(black,black,1);
			imshow("display", black);
			if(waitKey(WAITTIME)==27) break;
			//video.write(black);
			
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
	if (length == 0 || length<= FRICTION_COEFF)
	{
		return ;
	}
	int x_sign, y_sign;
	if (x_speed > 0) x_sign = 1; else x_sign = -1;
	if (y_speed > 0) y_sign = 1; else y_sign = -1;
	x_speed = ((float)x_sign)*(abs(x_speed) - abs(x_speed)/(float)length*FRICTION_COEFF);
	y_speed = ((float)y_sign)*(abs(y_speed) - abs(y_speed)/(float)length*FRICTION_COEFF);
	// speed min max control
	if (length > max_speed)
	{
		x_speed = x_speed / length * max_speed;
		y_speed = y_speed / length * max_speed;
	}
	else if (length < min_speed)
	{
		x_speed = x_speed / length * min_speed;
		y_speed = y_speed / length * min_speed;
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
	/*int len = max(src.cols, src.rows);
	Point2f pt(len/2., len/2.);
	Mat r = getRotationMatrix2D(pt, angle, 1.0);
	warpAffine(src,dst, r, Size(len,len));*/
}

void *UpdateFrame(void *)//another thread to get input
{
	while(1){
		cap >> frame;
		Mat left(frame, Range::all(), Range(0, 1280));
		remap(left, left, rmap[0][0], rmap[0][1], INTER_LINEAR);
		Mat calibrate(left, Range(y, y+height),Range(x,x+width));
		Mat current(calibrate, Range(125, 413), Range(370, 750));

		pthread_mutex_lock(&framelocker);
		remap_image = current.clone();
		pthread_mutex_unlock(&framelocker);
		
	}
	return NULL;
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
}

void perspectivetransform(Point2f& center, float& radius)
{
	center.x = (img_center_x - center.x)*perspective_x + center.x;
	center.y = (img_center_y - center.y)*perspective_y + center.y;
	radius = radius - perspective_r*sqrt((img_center_x - center.x)*(img_center_x - center.x) + (img_center_y - center.y)*(img_center_y - center.y));
	if(radius <=0) radius = 1;
}

void speedcontrol(const Point2f& centerpre, const float& radiuspre, const Point2f& center, const float& radius, float& ball_x, float& ball_y, float& ball_radius, float& ball_sx, float& ball_sy)
{
	person_sx = (center.x - centerpre.x) * person_speed_ratio;
	person_sy = (center.y - centerpre.y) * person_speed_ratio;
	if(radius-radiuspre>=0)
	{
		float person_speed = norm(person_sx, person_sy); 
		if (person_speed != 0) 
		{
			float cosc = person_sx / person_speed;
			float sinc = person_sy / person_speed;
			person_speed += (radius-radiuspre)/radiuspre * radius_speed;
			person_sx = person_speed * cosc;
			person_sy = person_speed * sinc;
		}
		else
		{
			person_sx = person_sy = (radius-radiuspre)/radiuspre * radius_speed;
		}
	}
		
	float xd = ball_x - center.x;
	float yd = ball_y - center.y;
	float distance = norm(xd, yd);
	if (distance == 0) return;
	if(distance <= radius + ball_radius) // ball is kicked
	{
		float ball_speed = norm(ball_sx, ball_sy);
		float cosangle = (person_sx*xd+person_sy*yd); // angle between person vector and ball-obj vector
		if(ball_speed == 0)
		{//when ball is not moving
			ball_sx = person_sx;
			ball_sy = person_sy;
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
			
			if(cosangle >= 0)
			{//when ball hit face by face
				ball_sx = person_sx + ball_reflect_sx; 
				ball_sy = person_sy + ball_reflect_sy;
			}
			else
			{
				ball_sx = ball_reflect_sx;
				ball_sy = ball_reflect_sy;
			}
		}
		// speed min max control
		ball_speed = norm(ball_sx, ball_sy);
		if (ball_speed == 0)
		{
			ball_sy = min_speed;
			ball_sx = 0;
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
		/*cout << "person speed ("<<person_sx<<", "<<person_sy<<"), ball speed ("<<ball_sx<<", "<<ball_sy<<")";
		float scale = 2;
		line(result, Point(100,100),Point(100 + person_sx*scale,100 + person_sy*scale), Scalar(255,255,255),5,8,0); // person speed vector
		//line(result, Point(100,100),Point(100 + ball_sx*scale,100 + ball_sy*scale), Scalar(255,255,255),5,8,0); // ball speed vector
		imshow("detect", result);
		waitKey(-1);*/
	}
}			

float norm(float x, float y)
{
	float ans = sqrt(x * x + y * y);
	if (ans == 0)
		return 0.01;
	else return ans;
}

void splitDisp(Mat& input, Mat& output1, Mat& output2)
{
	output1 = Mat(input, Rect(0, 0, disp_width/2, disp_height));
	output2 = Mat(input, Rect(disp_width/2, 0, disp_width, disp_height));
}


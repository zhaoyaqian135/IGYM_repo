//modifyyyyyyyyyyyyyyyyyyyyy
#include "header.h"

using namespace cv;
using namespace std;


void friction (float &x_speed, float &y_speed);
void readIntrinsics(string filename, Mat cameraMatrix[2], Mat distCoeffs[2]);
void readExtrinsics(string filename, Mat& R, Mat& T, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q);
void StereoCalib(Size imageSize, Rect validRoi[2], Mat rmap[2][2]);
void rotate(Mat& src, double angle, Mat &dst);
void *UpdateFrame(void *);
void detect(Mat& crop, Mat& background, Mat& result);
void perspectivetransform(Point2f& center, float& radius);
void filterfunc(Point2f& centerpre, float& radiuspre, Point2f& center, float& radius);
void predictfunc(Point2f& centerprepre, float& radiusprepre, Point2f& centerpre, float& radiuspre, Point2f& center, float& radius);
void speedcontrol(Point2f& centerpre, float& radiuspre, Point2f& center, float& radius, float& ball_x, float& ball_y, float& ball_radius, float& ball_sx, float& ball_sy);
float getNorm(float a, float b);
void ballBallCollide(float &ball_x_speedi, float &ball_y_speedi, float &ball_x_speedj, float &ball_y_speedj, float ball_distance_x, float ball_distance_y);

int main()
{
	//Initialization of bouncing ball
	float ball_radius = 12;
	float ball_number = 3;
	vector<float> ball_x(ball_number);
	vector<float> ball_y(ball_number);
	RNG rng;
	for(int i = 0; i < ball_number; i++)
	{
		ball_x[i] = 50;
		ball_y[i] = rng.uniform(50.0f, 200.0f);
	}
	vector<float> ball_x_speed(ball_number);
	vector<float> ball_y_speed(ball_number);
	for(int i = 0; i < ball_number; i++)
	{
		ball_x_speed[i] = 6*rng.uniform(-0.5f, 0.5f);
		ball_y_speed[i] = 6*rng.uniform(-0.5f, 0.5f);
	}

	//player score
	int player_1 = 0;
	int player_2 = 0;

	//person data
	vector<Point2f> center_current(3);
	vector<Point2f> center_previous(3);
	vector<Point2f> center_prepre(3);
	vector<float> radius_current(3);
	vector<float> radius_previous(3);
	vector<float> radius_prepre(3);

	int object_number = 0;
	int frame_number = 0;

	//Window setup
	namedWindow("display", WINDOW_NORMAL);
	moveWindow("display", 0, 0);
	setWindowProperty("display", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	
	//Input
	cap.open(1);
	cap.set(CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CAP_PROP_FRAME_HEIGHT, 720);
	cap.set(CV_CAP_PROP_FPS, 60);
	float fps = cap.get(CV_CAP_PROP_FPS);
	cout<<"Frame rate:"<<fps<<endl;
	
	//Calibration
	Size imageSize(1280, 720); 
	Rect validRoi[2];
	StereoCalib(imageSize, validRoi, rmap);
	x = MAX(validRoi[0].x, validRoi[1].x);
	y = MAX(validRoi[0].y, validRoi[1].y);
	width = MIN(validRoi[0].x+validRoi[0].width, validRoi[1].x+validRoi[1].width) - x;
	height = MIN(validRoi[0].y+validRoi[0].height, validRoi[1].y+validRoi[1].height) - y;
	
	//mutex parallel initial
	framelocker = PTHREAD_MUTEX_INITIALIZER;
	framecond = PTHREAD_COND_INITIALIZER;
	frame_buffer_prepared = 0;
	pthread_t threads_1;
	irets[1] = pthread_create(&threads_1,NULL,&UpdateFrame,NULL);

	//projector initial
	Mat black = imread("black_340_248.png");
	resize(black, black, Size(800, 600), INTER_LINEAR);
	flip(black,black,1);
	imshow("display",black);
	waitKey(1);
	time_t start,end;
	int seconds = 0;
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
			//detection
			detect(input, background, result);
			if (PRINT_DILATE_IMAGE == 1) imwrite(format("./image/%d.png",frame_number),result);

			getCenterAndRadius();

			//get the inital player position and parameter
			if(contours.size() != object_number)
			{
				object_number = 0;
				for(int	i = 0; i < contours.size(); i++)
				{
					if(radius[i]>1&&radius[i]<200){
						center_current[i] = center[i];
						radius_current[i] = radius[i];
						object_number++;
					}
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
						if(abs(center_previous[i].x - center[j].x)*ratio_x < radius_previous[i]*ratio_y && abs(center_previous[i].y - center[j].y) < radius_previous[i])
						{
							center_current[i] = center[j];
							radius_current[i] = radius[i];
						}
					}
				}
				
				for(int i = 0; i < object_number; i++)
				{
					//filter
					filterfunc(center_previous[i], radius_previous[i], center_current[i], radius_current[i]);
					if((center_prepre[i].x!=0)&&(center_prepre[i].y!=0))
					//prediction
						predictfunc(center_prepre[i], radius_prepre[i], center_previous[i], radius_previous[i], center_current[i], radius_current[i]);
				}	
			}
			
			//ball speed update
			for(int i = 0; i < object_number; i++)
			{
				if((center_previous[i].x!=0)&&(center_previous[i].y!=0))
					for(int j = 0; j < ball_number; j++)
						speedcontrol(center_previous[i], radius_previous[i], center_current[i], radius_current[i], ball_x[j], ball_y[j], ball_radius, ball_x_speed[j], ball_y_speed[j]);
			}
			
			// ball & ball collide
			for(int i = 0; i < ball_number-1; i++){
				for(int j = i+1;j < ball_number;j++)
				{
					float ball_distance_x = (ball_x[i]-ball_x[j])*ratio_x;
					float ball_distance_y = (ball_y[i]-ball_y[j])*ratio_y;
					float ball_distance = sqrt(ball_distance_x*ball_distance_x+ball_distance_y*ball_distance_y);
					if(ball_distance <= 2*ball_radius*ratio_y)
					{
						ballBallCollide(ball_x_speed[i], ball_y_speed[i], ball_x_speed[j], ball_y_speed[j], 
														ball_distance_x, ball_distance_y);
					}
				}
			}

			// Forced position correction for ball & ball. 
			for(int i = 0; i < ball_number-1; i++){
				int j = i+1;
				for(;j < ball_number;j++){
					float ball_distance_x = (ball_x[i]-ball_x[j])*ratio_x;
					float ball_distance_y = (ball_y[i]-ball_y[j])*ratio_y;
					float ball_distance = sqrt(ball_distance_x*ball_distance_x+ball_distance_y*ball_distance_y);
					if(ball_distance <= (2*ball_radius-2)*ratio_y){
							ball_x[i] += (2*ball_radius*ratio_y-ball_distance+2)*ball_distance_x/ball_distance/ratio_x;
							ball_y[i] += (2*ball_radius*ratio_y-ball_distance+2)*ball_distance_y/ball_distance/ratio_y;
					}
				}
			}

			//Forced bouncing position correct
			for(int i = 0; i < object_number; i++){
				for(int j = 0; j < ball_number; j++){
					float xd = (ball_x[j] - center_current[i].x)*ratio_x;
					float yd = (ball_y[j] - center_current[i].y)*ratio_y;
					float distance = sqrt(xd*xd+yd*yd);
					if(distance < (radius_current[i] + ball_radius)*ratio_y){
						float cosc = xd/distance;
						float sinc = yd/distance;
						ball_x[j] = ball_x[j] + (radius_current[i]*ratio_y - distance + ball_radius*ratio_y)*cosc/ratio_x;
						ball_y[j] = ball_y[j] + (radius_current[i]*ratio_y - distance + ball_radius*ratio_y)*sinc/ratio_y;
					}
				}
			}

			//boundary bouncing
			for(int j = 0; j < ball_number; j++){
				if(((ball_x[j]-marginal_ws)*ratio_x-ball_radius*ratio_y) <= 5)
				{
					ball_x_speed[j] = -ball_x_speed[j];
					//friction_enable = 1;
				}
				if( (ball_y[j]-marginal_hs+ball_radius)*ratio_y >= 599 || (ball_y[j]-marginal_hs-ball_radius)*ratio_y <= 0)
				{
					ball_y_speed[j] = -ball_y_speed[j];
					//friction_enable = 1;
				}
				if(((ball_x[j]-marginal_ws)*ratio_x+ball_radius*ratio_y) >= 799){
					player_1++;
					ball_x[j] = 50;
					ball_y[j] = rng.uniform(50.0f,200.0f);
					ball_x_speed[j] = 6*rng.uniform(-0.5f,0.5f);
					ball_y_speed[j] = 6*rng.uniform(-0.5f,0.5f);
				}
			}
			//Forced boundary 
			for(int j = 0; j < ball_number; j++){
				if( ((ball_x[j]-marginal_ws)*ratio_x-ball_radius*ratio_y) < 0) ball_x[j] = (0.0+ball_radius*ratio_y)/ratio_x+marginal_ws;
				if( (ball_y[j]-marginal_hs+ball_radius)*ratio_y > 600) ball_y[j] = (599.0-ball_radius*ratio_y)/ratio_y+marginal_hs;
				if( (ball_y[j]-marginal_hs-ball_radius)*ratio_y < 0) ball_y[j] = (0.0+ball_radius*ratio_y)/ratio_y+marginal_hs;
			}
			
			//update ball position
			for(int j = 0; j < ball_number; j++){
				ball_x[j] += ball_x_speed[j];
				ball_y[j] += ball_y_speed[j];
			}
			
			//draw boundary
			line(black, Point(0,0),Point(0,599), Scalar(255,255,255),5,8,0);
			line(black, Point(0,0),Point(799,0), Scalar(255,255,255),5,8,0);
			line(black, Point(799,0),Point(799,599), Scalar(255,255,255),5,8,0);
			line(black, Point(0,599),Point(799,599), Scalar(255,255,255),5,8,0);

			time(&end);
			seconds += difftime(end,start);
			time(&start);
			//score display
			rotate(black, 90, black);
			putText(black, format("lost ball: %d", player_1), Point(0,170), CV_FONT_HERSHEY_PLAIN, 5.0, cvScalar(255,255,255), 5, CV_AA);
			putText(black, format("time: %ds", seconds), Point(0, 240), CV_FONT_HERSHEY_PLAIN, 5.0, cvScalar(255,255,255), 5, CV_AA);
			rotate(black, -90, black);

			//draw ball
			for(int j = 0; j < ball_number; j++){
				circle(black, Point((ball_x[j]-marginal_ws)*ratio_x, (ball_y[j]-marginal_hs)*ratio_y), ball_radius*ratio_y, Scalar(255,255,255),-1,8,0);
			}
			
			//draw player
			for(int i = 0; i < object_number; i++)
			{	
				if(radius_current[i]>1&&radius_current[i]<200)
					if((center_current[i].x-marginal_ws)>=0&&(center_current[i].y-marginal_hs)>=0&&(center_current[i].y-marginal_hs)<=599&&(center_current[i].x-marginal_ws)<=799)
					circle(black, Point((center_current[i].x-marginal_ws)*ratio_x, (center_current[i].y-marginal_hs)*ratio_y), radius_current[i]*ratio_y, Scalar(255,255,255), 2, 8, 0);
			}		
		
			//result display
			flip(black,black,1);
			imshow("display", black);
			if(waitKey(8)==27) break;

			//data update
			black = imread("black_340_248.png");
			resize(black, black, Size(800, 600), INTER_LINEAR);
			
			for(int i = 0; i < object_number; i++)
			{
				if((center_previous[i].x==0)&&(center_previous[i].y==0))
				{
					center_previous[i] = center_current[i];
					radius_previous = radius_current;
				}
				else
				{
					center_prepre[i] = center_previous[i];
					center_previous[i] = center_current[i];
					radius_prepre = radius_previous;
					radius_previous = radius_current;
				}
			}
			
			/*if(frame_number == 600){
				time(&end);
				double seconds = difftime(end,start);
				double fps = frame_number/seconds;
				cout<<"fps"<<fps<<endl;
			}*/
		}
		
	}
	
	return 0;						
}

void friction (float &x_speed, float &y_speed)
{
	float length = sqrt(x_speed*x_speed + y_speed*y_speed);
	if (length <= FRICTION_COEFF)
	{
		x_speed = y_speed = 0;
		return ;
	}
	int x_sign, y_sign;
	if (x_speed > 0) x_sign = 1; else x_sign = -1;
	if (y_speed > 0) y_sign = 1; else y_sign = -1;
	x_speed = ((float)x_sign)*(abs(x_speed) - abs(x_speed)/(float)length*FRICTION_COEFF);
	y_speed = ((float)y_sign)*(abs(y_speed) - abs(y_speed)/(float)length*FRICTION_COEFF);
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

void *UpdateFrame(void *)
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


void filterfunc(Point2f& centerpre, float& radiuspre, Point2f& center, float& radius)
{
	center.x = filter*centerpre.x + (1 - filter)*center.x;
	center.y = filter*centerpre.y + (1 - filter)*center.y;
	radius = filter*radiuspre + (1 - filter)*radius;
}

void predictfunc(Point2f& centerprepre, float& radiusprepre, Point2f& centerpre, float& radiuspre, Point2f& center, float& radius)	
{
	center.x = predict*center.x + (1 - predict)*((centerpre.x - centerprepre.x)+centerpre.x);
	center.y = predict*center.y + (1 - predict)*((centerpre.y - centerprepre.y)+centerpre.y);
	radius = predict*radius + (1 - predict)*((radiuspre - radiusprepre)+radiuspre);
}

void speedcontrol(Point2f& centerpre, float& radiuspre, Point2f& center, float& radius, float& ball_x, float& ball_y, float& ball_radius, float& ball_sx, float& ball_sy)
{
	float person_sx,person_sy;
	if(radius-radiuspre>=0)
	{
		person_sx = center.x - centerpre.x + (radius-radiuspre)/radiuspre * radius_speed;
		person_sy = center.y - centerpre.y + (radius-radiuspre)/radiuspre * radius_speed;
	}
	else
	{
		person_sx = center.x - centerpre.x;
		person_sy = center.y - centerpre.y;
	}
	float xd = (ball_x - center.x)*ratio_x;
	float yd = (ball_y - center.y)*ratio_y;
	float distance = sqrt(xd*xd+yd*yd);
	if(distance <= (radius + ball_radius)*ratio_y)
	{
		float ball_speed = sqrt(ball_sx*ball_sx+ball_sy*ball_sy);
		float cosa = (ball_sx*xd+ball_sy*yd)/distance/ball_speed;
		if(ball_speed == 0 || distance == 0)
		{//when ball is not moving
			ball_sx = person_sx;
			ball_sy = person_sy;
		}
		else{
			float ball_reflect_sx, ball_reflect_sy;
			if(cosa >= 0)
			{//when ball hit face by face
				ball_reflect_sx = ball_sx;
				ball_reflect_sy = ball_sy;
			}
			else
			{
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
			}
			float ball_reflect_speed = sqrt(ball_reflect_sx*ball_reflect_sx+ball_reflect_sy*ball_reflect_sy);
			float person_speed = sqrt(person_sx*person_sx+person_sy*person_sy);
			float cosb = (ball_reflect_sx*person_sx+ball_reflect_sy*person_sy)/ball_reflect_speed/person_speed;
			if(cosb >= 0)
			{
				ball_sx = ball_reflect_sx + person_speed_ratio*person_sx;
				ball_sy = ball_reflect_sy + person_speed_ratio*person_sy;
			}
			else
			{
				ball_sx = ball_reflect_sx;
				ball_sy = ball_reflect_sy;
			}
		}
		if(abs(ball_sx) >= max_speed){
			ball_sy = ball_sy*max_speed/ball_sx;
			ball_sx = max_speed;
		}
		if(abs(ball_sy) >= max_speed){
			ball_sx = ball_sx*max_speed/ball_sy;
			ball_sy = max_speed;
		}
		if(abs(ball_sx) <= min_speed){
			ball_sy = ball_sy*min_speed/ball_sx;
			ball_sx = min_speed;
		}
		if(abs(ball_sy) <= min_speed){
			ball_sx = ball_sx*min_speed/ball_sy;
			ball_sy = min_speed;
		}
	}
	//else{
	//friction enable
		//if(friction_enable == 1){
		//	friction(ball_sx, ball_sy);
		//}
	//}	
}

float getNorm(float a, float b)
{
	return sqrt(a*a + b*b);
}

void ballBallCollide(float &ball_x_speedi, float &ball_y_speedi, 
									 	 float &ball_x_speedj, float &ball_y_speedj, 
									 	 float ball_distance_x, float ball_distance_y)
{
	float ball_distance = getNorm(ball_distance_x, ball_distance_y);
	if (ball_distance == 0) return;
	float cos1 = ball_distance_x / ball_distance;
	float sin1 = ball_distance_y / ball_distance;
	float speedi = getNorm(ball_x_speedi, ball_y_speedi);
	float speedj = getNorm(ball_x_speedj, ball_y_speedj);
	float cos3i = 0, sin3i = 0, cos3j = 0, sin3j = 0;
	if (speedi != 0.0)
	{
		cos3i = ball_x_speedi / speedi;
		sin3i = ball_y_speedi / speedi;
	}
	if (speedj != 0.0)
	{
		cos3j = ball_x_speedj / speedj;
		sin3j = ball_y_speedj / speedj;
	}
	float cos4i = cos3i*cos1 + sin3i*sin1;
	float sin4i = sin3i*cos1 - cos3i*sin1;
	float cos4j = cos3j*cos1 + sin3j*sin1;
	float sin4j = sin3j*cos1 - cos3j*sin1;
	float speedi_vertical = speedi * cos4i;
	float speedi_horizontal = speedi * sin4i;
	float speedj_vertical = speedj * cos4j;
	float speedj_horizontal = speedj * sin4j;
	// super elastic collision, switching vertical speed
	float temp = speedi_vertical;
	speedi_vertical = speedj_vertical;
	speedj_vertical = temp;

	speedi = getNorm(speedi_vertical, speedi_horizontal);
	speedj = getNorm(speedj_vertical, speedj_horizontal);
	cos4i = speedi_vertical / speedi;
	sin4i = speedi_horizontal / speedi;
	cos4j = speedj_vertical / speedj;
	sin4j = speedj_horizontal / speedj;
	cos3i = cos4i*cos1 - sin4i*sin1;
	sin3i = sin4i*cos1 + cos4i*sin1;
	cos3j = cos4j*cos1 - sin4j*sin1;
	sin3j = sin4j*cos1 + cos4j*sin1;
	ball_x_speedi = speedi * cos3i;
	ball_y_speedi = speedi * sin3i;
	ball_x_speedj = speedj * cos3j;
	ball_y_speedj = speedj * sin3j;
}
			

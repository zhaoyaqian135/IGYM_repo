#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <pthread.h>

using namespace cv;
using namespace std;

//camera input
VideoCapture cap;
Mat frame;
const int WIDTH = 1280;
const int HEIGHT = 720; // choices are: 2208x1242@15fps, 
			 //		 1920x1080@30fps,
			 //		 1280x720@60fps
			 //		 672x376@100fps
const int disp_width = 600*2;
const int disp_height = 800;

//Calibrate parameter
Mat remap_image;
Mat rmap[2][2];
int x, y, width, height;

void readIntrinsics(string filename, Mat cameraMatrix[2], Mat distCoeffs[2]);
void readExtrinsics(string filename, Mat& R, Mat& T, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q);
void StereoCalib(Size imageSize, Rect validRoi[2], Mat rmap[2][2]);
void rotate(Mat& src, double angle, Mat &dst);
void splitDisp(Mat& input, Mat& output1, Mat& output2);

int main()
{
	//Window setup
	namedWindow("display", WINDOW_NORMAL);
	moveWindow("display", 0, 0);
	setWindowProperty("display", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	namedWindow("display1", WINDOW_NORMAL);
	moveWindow("display1", 0, 800);
	setWindowProperty("display1", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	namedWindow("display2", WINDOW_NORMAL);
	moveWindow("display2", 0, 1600);
	setWindowProperty("display2", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	//Input
	
	cap.open(1);
	cap.set(CAP_PROP_FRAME_WIDTH, WIDTH);
	cap.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);
	
	//Calibration
	Size imageSize(WIDTH, HEIGHT); 
	Rect validRoi[2];
	StereoCalib(imageSize, validRoi, rmap);
	x = MAX(validRoi[0].x, validRoi[1].x);
	y = MAX(validRoi[0].y, validRoi[1].y);
	width = MIN(validRoi[0].x+validRoi[0].width, validRoi[1].x+validRoi[1].width) - x;
	height = MIN(validRoi[0].y+validRoi[0].height, validRoi[1].y+validRoi[1].height) - y;
	
	cout << "end calibration. "<< endl;
	
	//projector initial
	Mat black = imread("black.png");
	Mat output1, output2;
	resize(black, black, Size(disp_width, disp_height), INTER_LINEAR);
	
	cout << "end projector initial. "<<endl;
	
	//draw boundary
	int line_offset = 2;
	line(black, Point(line_offset,line_offset),Point(line_offset, disp_height - line_offset), Scalar(255,255,255),5,8,0);
	line(black, Point(line_offset,line_offset),Point(disp_width - line_offset,line_offset), Scalar(255,255,255),5,8,0);
	line(black, Point(disp_width - line_offset,line_offset),Point(disp_width - line_offset,disp_height - line_offset), Scalar(255,255,255),5,8,0);
	line(black, Point(line_offset,disp_height - line_offset),Point(disp_width - line_offset,disp_height - line_offset), Scalar(255,255,255),5,8,0);
	
	splitDisp(black, output1, output2);
	imshow("display1", output1);
	imshow("display2", output2);
	waitKey(1);
	
	cout << "end displaying initial black"<<endl;

	//VideoWriter video("video_input.avi", CV_FOURCC('M','J','P','G'), 60, Size(380,288));
	int number = 0;
	time_t start,end;
	time(&start);
	cout << "Enter loop!" << endl;
	while(1)
	{
		number++;
		if (number > 20) return 0;
		
		cap >> frame;
		
		imwrite("./image/frame.png", frame);
		Mat left(frame, Range::all(), Range(0, 1280));
		imwrite("./image/left.png", left);
		remap(left, left, rmap[0][0], rmap[0][1], INTER_LINEAR);
		imwrite("./image/leftremap.png", left);
		//Mat calibrate(left, Range(y, y+height),Range(x,x+width));
		//imwrite("./image/calibrate.png", calibrate);
		int cut_start_x = 310;
		int cut_start_y = 120;
		
		Mat current(left, Rect(cut_start_x, cut_start_y, 670, 500));
		imwrite("./image/current.png", current);
		imshow("display", current);
		waitKey(15);
		//video.write(current);
		//imshow("display", current);//open to get range
		//waitKey(1);
		if(number % 200 == 0)
			{
				time(&end);
				double seconds = difftime(end,start);
				double fps = 200/seconds;
				cout<<"fps: "<<fps<<endl;
				time(&start);
			}
	}
	return 0;
		
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

void splitDisp(Mat& input, Mat& output1, Mat& output2)
{
	output1 = Mat(input, Rect(0, 0, disp_width/2, disp_height-1));
	rotate(output1,90,output1);
	output2 = Mat(input, Rect(disp_width/2, 0, disp_width/2, disp_height));
	rotate(output2,90,output2);
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

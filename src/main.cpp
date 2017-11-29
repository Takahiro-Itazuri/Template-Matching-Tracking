#include<iostream>
#include<opencv2\opencv.hpp>

cv::Mat img, img_roi, temp;
cv::Point2i sp(-1, -1), ep(-1, -1);
cv::Rect post_pos;
cv::Size search_window(50, 50);
bool bSelecting = false, bSelected = false;
bool bShow = false;

void onMouse(int event, int x, int y, int flags, void* userdata) {
	img_roi = img.clone();

	switch (event)
	{
	case cv::EVENT_LBUTTONDOWN:
		sp = cv::Point2i(x, y);
		bSelecting = true;
		bSelected = false;
		break;

	case cv::EVENT_LBUTTONUP:
		ep = cv::Point2i(x, y);
		cv::rectangle(img_roi, sp, ep, cv::Scalar(0, 255, 0), 2, 8, 0);
		bSelecting = false;
		bSelected = true;
		break;

	case cv::EVENT_MOUSEMOVE:
		if (bSelecting) {
			ep = cv::Point2i(x, y);
			cv::rectangle(img_roi, sp, ep, cv::Scalar(255, 0, 0), 2, 8, 0);
		}

		if (bSelected) {
			cv::rectangle(img_roi, sp, ep, cv::Scalar(0, 0, 255), 2, 8, 0);
		}
		break;
	}

	cv::imshow("Object Selection", img_roi);
}

int main(int argc, char* argv[]) {
	// Load video
	std::string str = "occulusion_30fps";
	cv::VideoCapture video( str + ".mp4");
	cv::VideoWriter writer(str + "_TempleteMatching.avi", CV_FOURCC('X', 'V', 'I', 'D'), video.get(CV_CAP_PROP_FPS), cv::Size(video.get(CV_CAP_PROP_FRAME_WIDTH), video.get(CV_CAP_PROP_FRAME_HEIGHT)));

	video >> img;
	cv::namedWindow("Object Selection");
	cv::imshow("Object Selection", img);
	cv::setMouseCallback("Object Selection", onMouse);
	cv::waitKey(0);

	cv::destroyWindow("Object Selection");
	temp = (img(cv::Rect(sp, ep))).clone();
	cv::imshow("ROI", temp);
	
	post_pos = cv::Rect(sp, ep);
	while (1) {
		video >> img;
		if (img.empty()) {
			break;
		}

		cv::Point2i pos(-1, -1);
		double min = INFINITY;

		for (int y = post_pos.y - search_window.height / 2, _y = post_pos.y + search_window.height / 2, H = img.rows, _H = post_pos.height; y < _y; ++y) {
			for (int x = post_pos.x - search_window.width / 2, _x = post_pos.x + search_window.width / 2, W = img.cols, _W = post_pos.width; x < _x; ++x) {

				if (y < 0 || y + _H > H || x < 0 || x + _W > W) continue;

				double score = 0;
				for (int h = 0; h < _H; ++h) {
					cv::Vec3b* p_temp = temp.ptr<cv::Vec3b>(h);
					cv::Vec3b* p_img = img.ptr<cv::Vec3b>(y + h);
					for (int w = 0; w < _W; ++w) {
						for (int c = 0; c < 3; ++c) {
							score += std::pow(((int)p_temp[w][c] - (int)p_img[x + w][c]), 2) / (_W * _H * 3.0);
						}
					}
				}

				if (score < min) {
					pos = cv::Point2i(x, y);
					min = score;
				}

				if (bShow) {
					img_roi = img.clone();
					cv::rectangle(img_roi, cv::Rect(cv::Point2i(x, y), temp.size()), cv::Scalar(255, 0, 0), 1, 8, 0);
					cv::rectangle(img_roi, cv::Rect(cv::Point2i(pos.x, pos.y), temp.size()), cv::Scalar(0, 0, 255), 2, 8, 0);
					cv::imshow("Searching...", img_roi);
					cv::waitKey(1);
				}
			}
		}

		cv::rectangle(img, cv::Rect(pos, temp.size()), cv::Scalar(0, 0, 255), 2, 8, 0);
		post_pos = cv::Rect(pos, temp.size());
		cv::imshow("tracking", img);
		writer << img;
		cv::waitKey(1);
	}

	return 0;
}
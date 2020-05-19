#include <iostream>
#include <opencv2/opencv.hpp>

int main(){

    std::cout<<"OpenCV C++";
    
    cv::Mat img;
    img = cv::imread("../dependencies/dog.jpg");
    cv::imshow("Image of a Dog",img);

    cv::waitKey();
    return 0;
    
}

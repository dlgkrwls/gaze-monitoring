#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    const std::string imagePath = "C:\\Users\\user\\Desktop\\lee\\Gaze-monitoring\\assets\\images\\24.jpg";
    cv::Mat image = cv::imread(imagePath);
    if (image.empty())
    {
        std::cerr << "Could not read the image: " << imagePath << std::endl;
        return -1;
    }
    std::cout << "img width:" << image.cols << std::endl;
    std::cout << "img height:" << image.rows << std::endl;
    std::cout << "img channels:" << image.channels() << std::endl;
    return 0;
}

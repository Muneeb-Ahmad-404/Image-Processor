#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <vector>
#include <filesystem>
#include <iostream>

using namespace cv;
using namespace std;
using namespace filesystem;

void loadingImages(const string& folderPath, vector<Mat>& imgs);
void convertToGrayscale(const vector<Mat>& imgs, vector<Mat>& grayImgs);
void resizeImages(const vector<Mat>& grayImgs, vector<Mat>& resizedImgs);

int main()
{
    string folderPath = "images/";
    vector<Mat> imgs;
    vector<Mat> grayImgs;
    vector<Mat> resizedImgs;
    
    loadingImages(folderPath, imgs);
    convertToGrayscale(imgs, grayImgs);
    resizeImages(grayImgs, resizedImgs);
    
    return 0;
}

void loadingImages(const string& folderPath, vector<Mat>& imgs) {
    int totalImages = 0;

    if(!exists(folderPath) || !is_directory(folderPath)){
        cout<< "Directory does not exist: " << folderPath << endl;
        return;
    }

    for(const auto& entry : directory_iterator(folderPath)){
        string accepted_types[] = {".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"};
        string type = entry.path().extension().string();

        bool isAccepted = false;
        for(const auto& accepted_type : accepted_types){
            if(type == accepted_type){
                isAccepted = true;
                break;
            }
        }

        if(!isAccepted){
            cout << "Skipped:  unsupported file type: " << entry.path().filename() << endl;
            continue;
        }

        Mat img = imread(entry.path().string(), IMREAD_COLOR);
        
        if(img.empty()){
            cout << "Skipped: Could not read the image: " << entry.path().filename() << endl;
            continue;
        }
        
        imgs.push_back(img);
        totalImages++;
        cout << "Loaded: " << entry.path().filename() << ": " << img.cols << " x " << img.rows << endl;
    }
    
    cout << "Total images loaded: " << totalImages << endl;
}

void convertToGrayscale(const vector<Mat>& imgs, vector<Mat>& grayImgs) {
    for(const auto& img : imgs){
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        if(gray.empty()){
            cout << "Skipped: Could not convert to grayscale" << endl;
            continue;
        }
        grayImgs.push_back(gray);
        //check
        // imshow("Grayscale Image", gray);
        // waitKey(0);
        cout<< "Converted to grayscale" << endl;
    }
}

void resizeImages(const vector<Mat>& grayImgs, vector<Mat>& resizedImgs) {
    for(const auto& gray : grayImgs){
        Mat resized;
        resize(gray, resized, Size(100, 100));
        if(resized.empty()){
            cout << "Skipped: Could not resize the image" << endl;
            continue;
        }
        resizedImgs.push_back(resized);
        //check
        // imshow("Resized Image", resized);
        // waitKey(0);
        cout<< "Resized to 100x100" << endl;
    }
}
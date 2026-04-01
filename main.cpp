#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/photo.hpp>
#include <vector>
#include <filesystem>
#include <iostream>
#include <omp.h>
#include <chrono>

using namespace cv;
using namespace std;
using namespace filesystem;

bool loadingImages(const string& folderPath, vector<Mat>& imgs);
void convertToGrayscale(const vector<Mat>& imgs, vector<Mat>& grayImgs);
void resizeImages(const vector<Mat>& grayImgs, vector<Mat>& resizedImgs);
void hairRemoval(const vector<Mat>& imgs);

int main()
{
    string folderPath = "images/";
    vector<Mat> imgs;
    vector<Mat> grayImgs;
    vector<Mat> resizedImgs;
    vector<Mat> hairRemovedImgs;
    
    bool loaded = loadingImages(folderPath, imgs);
    if(!loaded){
        cout << "Failed to load images from folder: " << folderPath << endl;
        return -1;
    }
    hairRemoval(imgs);
    return 0;
}

bool loadingImages(const string& folderPath, vector<Mat>& imgs) {
    int totalLoaded = 0;
    if(!exists(folderPath) || !is_directory(folderPath)){
        cout<< "Directory does not exist: " << folderPath << endl;
        return false;
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
            // cout << "Skipped:  unsupported file type: " << entry.path().filename() << endl;
            continue;
        }

        Mat img = imread(entry.path().string(), IMREAD_COLOR);
        
        if(img.empty()){
            // cout << "Skipped: Could not read the image: " << entry.path().filename() << endl;
            continue;
        }
        
        imgs.push_back(img);
        totalLoaded++;
        // cout << "Loaded: " << entry.path().filename() << ": " << img.cols << " x " << img.rows << endl;
    }
    
    cout << "Total images loaded: " << totalLoaded << endl;
    return totalLoaded > 0;//true if at least one image was loaded, false otherwise
}

void convertToGrayscale(const vector<Mat>& imgs, vector<Mat>& grayImgs) {
    int totalConverted = 0;
    for(const auto& img : imgs){
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        if(gray.empty()){
            // cout << "Skipped: Could not convert to grayscale" << endl;
            continue;
        }
        grayImgs.push_back(gray);
        totalConverted++;
        // check
        // imshow("Grayscale Image", gray);
        // waitKey(0);
        // cout<< "Converted to grayscale" << endl;
    }
    // cout << "Total images converted to grayscale: " << totalConverted << endl;
}

void resizeImages(const vector<Mat>& grayImgs, vector<Mat>& resizedImgs) {
    int totalResized = 0;
    for(const auto& gray : grayImgs){
        Mat resized;
        resize(gray, resized, Size(1200, 800));
        if(resized.empty()){
            // cout << "Skipped: Could not resize the image" << endl;
            continue;
        }
        resizedImgs.push_back(resized);
        totalResized++;
        //check
        // imshow("Resized Image", resized);
        // waitKey(0);
        // cout<< "Resized to 100x100" << endl;
    }
    // cout << "Total images resized: " << totalResized << endl;
}

void hairRemoval(const vector<Mat>& imgs) {
    int totalProcessed = 0;
    auto totalStart = chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < imgs.size(); i++){
        auto start = chrono::high_resolution_clock::now();

        const Mat& orig = imgs[i];

        // Step 0: Downscale original for inpainting
        double scaleFactor = 800.0 / max(orig.cols, orig.rows); // max dimension ~800px
        Mat smallOrig, smallGray;
        resize(orig, smallOrig, Size(), scaleFactor, scaleFactor, INTER_AREA);
        cvtColor(smallOrig, smallGray, COLOR_BGR2GRAY);

        // Step 1: Multiscale Blackhat
        Mat blackhat, mask, finalSmall;
        vector<int> kernelSizes = {15, 25, 35};
        Mat accumMask = Mat::zeros(smallGray.size(), CV_8UC1);

        for(auto kSize : kernelSizes){
            Mat khat, msk;
            Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(kSize, kSize));
            morphologyEx(smallGray, khat, MORPH_BLACKHAT, kernel);
            normalize(khat, khat, 0, 255, NORM_MINMAX);

            double minVal, maxVal;
            minMaxLoc(khat, &minVal, &maxVal);
            double threshVal = minVal + 0.15 * (maxVal - minVal);
            threshold(khat, msk, threshVal, 255, THRESH_BINARY);

            dilate(msk, msk, getStructuringElement(MORPH_ELLIPSE, Size(3,3)));
            bitwise_or(accumMask, msk, accumMask);
        }

        int nonZero = countNonZero(accumMask);
        Mat final;
        if(nonZero == 0){
            final = orig.clone();
        } else {
            // Step 2: Inpaint on downscaled image
            inpaint(smallOrig, accumMask, finalSmall, 3.0, INPAINT_TELEA);

            // Step 3: Upscale to original size
            resize(finalSmall, final, orig.size(), 0, 0, INTER_CUBIC);
        }

        // Save output immediately
        string outPath = "processed_images/hair_removed_" + to_string(i) + ".png";
        imwrite(outPath, final);

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;

        #pragma omp critical
        {
            totalProcessed++;
            cout << "Processed image " << i << " in " << elapsed.count() << " sec, saved as: " << outPath << endl;
        }
    }

    auto totalEnd = chrono::high_resolution_clock::now();
    chrono::duration<double> totalElapsed = totalEnd - totalStart;
    cout << "Total images processed: " << totalProcessed << " in " << totalElapsed.count() << " sec" << endl;
}
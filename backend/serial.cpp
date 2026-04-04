#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/photo.hpp>
#include <vector>
#include <filesystem>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace filesystem;

void hairRemovalSerial(const vector<string>& imagePaths, const string& outFolder);

int main(){
    string folderPath = "../images";
    string outFolder = "../processed_images/serial";
    create_directories(outFolder);

    vector<string> imagePaths;
    for(const auto& entry : directory_iterator(folderPath)){
        if(entry.path().extension() == ".jpg" || entry.path().extension() == ".png")
            imagePaths.push_back(entry.path().string());
    }

    hairRemovalSerial(imagePaths, outFolder);
    
    return 0;
}

void hairRemovalSerial(const vector<string>& imagePaths, const string& outFolder) {
    int totalProcessed = 0;
    auto totalStart = chrono::high_resolution_clock::now();

    for(size_t i = 0; i < imagePaths.size(); i++){
        auto start = chrono::high_resolution_clock::now();
        string imgPath = imagePaths[i];

        Mat orig = imread(imgPath, IMREAD_COLOR);
        if(orig.empty()){
            cout << "Skipped: could not read " << imgPath << endl;
            continue;
        }

        // Step 0: Downscale original for inpainting
        double scaleFactor = 800.0 / max(orig.cols, orig.rows); // max dimension ~800px
        Mat smallOrig, smallGray;
        resize(orig, smallOrig, Size(), scaleFactor, scaleFactor, INTER_AREA);
        cvtColor(smallOrig, smallGray, COLOR_BGR2GRAY);

        // Step 1: Multiscale Blackhat
        Mat accumMask = Mat::zeros(smallGray.size(), CV_8UC1);
        vector<int> kernelSizes = {15, 25, 35};

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

        Mat final;
        int nonZero = countNonZero(accumMask);
        if(nonZero == 0){
            final = orig.clone();
        } else {
            // Step 2: Inpaint on downscaled image
            Mat finalSmall;
            inpaint(smallOrig, accumMask, finalSmall, 3.0, INPAINT_TELEA);

            // Step 3: Upscale to original size
            resize(finalSmall, final, orig.size(), 0, 0, INTER_CUBIC);
        }

        // Save output in folder
        string outPath = outFolder + "/hair_removed_" + path(imgPath).filename().string();
        imwrite(outPath, final);

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        totalProcessed++;
        cout << "Processed image " << i << " in " << elapsed.count() 
             << " sec, saved as: " << outPath << endl;
    }

    auto totalEnd = chrono::high_resolution_clock::now();
    chrono::duration<double> totalElapsed = totalEnd - totalStart;
    cout << "Total images processed: " << totalProcessed 
         << " in " << totalElapsed.count() << " sec" << endl;
}
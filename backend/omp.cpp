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

void hairRemovalOMP(const vector<string>& imagePaths, const string& outFolder);

int main(int argc, char** argv){
    string folderPath = "../images";
    string outFolder = "../processed_images/omp";
    create_directories(outFolder);

    vector<string> imagePaths;
    for(const auto& entry : directory_iterator(folderPath)){
        if(entry.path().extension() == ".jpg" || entry.path().extension() == ".png")
            imagePaths.push_back(entry.path().string());
    }
    hairRemovalOMP(imagePaths, outFolder);

    return 0;
}

void hairRemovalOMP(const vector<string>& imagePaths, const string& outFolder) {
    int totalProcessed = 0;
    auto totalStart = chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < imagePaths.size(); i++){
        auto start = chrono::high_resolution_clock::now();

        string imgPath = imagePaths[i];
        Mat orig = imread(imgPath, IMREAD_COLOR);
        if(orig.empty()) continue;

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

        // Save output
        string outPath = outFolder + "/hair_removed_" + to_string(i) + "_" + path(imgPath).filename().string();
        imwrite(outPath, final);

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;

        #pragma omp critical
        {
            totalProcessed++;
            cout << "Processed image " << i << " in " << elapsed.count() 
                 << " sec, saved as: " << outPath << endl;
        }
    }

    auto totalEnd = chrono::high_resolution_clock::now();
    chrono::duration<double> totalElapsed = totalEnd - totalStart;
    cout << "Total images processed: " << totalProcessed 
         << " in " << totalElapsed.count() << " sec" << endl;
}
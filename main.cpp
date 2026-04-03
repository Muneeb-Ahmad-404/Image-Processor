#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/photo.hpp>
#include <opencv2/core/ocl.hpp>
#include <vector>
#include <filesystem>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <mpi.h>

using namespace cv;
using namespace std;
using namespace filesystem;
using namespace ocl;

bool loadingImages(const string& folderPath, vector<Mat>& imgs);
void convertToGrayscale(const vector<Mat>& imgs, vector<Mat>& grayImgs);
void resizeImages(const vector<Mat>& grayImgs, vector<Mat>& resizedImgs);
void hairRemovalSerial(const vector<string>& imagePaths, const string& outFolder);
void hairRemovalOMP(const vector<string>& imagePaths, const string& outFolder);
void hairRemovalMPI(const vector<string>& imagePaths, const string& outFolder, int rank, int size);
void hairRemovalOCL(const vector<string>& imagePaths, const string& outFolder, int rank, int size);

int main(int argc, char** argv){
    cv::ocl::setUseOpenCL(true);
    cout << "OpenCL enabled: " << cv::ocl::useOpenCL() << endl;

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string folderPath = "images";
    string outFolder = "processed_images";
    create_directories(outFolder);

    vector<string> imagePaths;
    for(const auto& entry : directory_iterator(folderPath)){
        if(entry.path().extension() == ".jpg" || entry.path().extension() == ".png")
            imagePaths.push_back(entry.path().string());
    }

    hairRemovalSerial(imagePaths, outFolder);
    hairRemovalOMP(imagePaths, outFolder);
    hairRemovalMPI(imagePaths, outFolder, rank, size);
    hairRemovalOCL(imagePaths, outFolder, rank, size);

    MPI_Finalize();
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

void hairRemovalMPI(const vector<string>& imagePaths, const string& outFolder, int rank, int size) {
    int totalProcessed = 0;
    auto totalStart = chrono::high_resolution_clock::now();

    // Split work across MPI ranks
    vector<string> localImages;
    for(size_t i = 0; i < imagePaths.size(); i++){
        if(i % size == rank) // simple round-robin distribution
            localImages.push_back(imagePaths[i]);
    }

    #pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < localImages.size(); i++){
        auto start = chrono::high_resolution_clock::now();
        string imgPath = localImages[i];
        Mat orig = imread(imgPath, IMREAD_COLOR);
        if(orig.empty()) continue;

        // Downscale for faster inpainting
        double scaleFactor = 800.0 / max(orig.cols, orig.rows);
        Mat smallOrig, smallGray;
        resize(orig, smallOrig, Size(), scaleFactor, scaleFactor, INTER_AREA);
        cvtColor(smallOrig, smallGray, COLOR_BGR2GRAY);

        // Multiscale Blackhat
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
            Mat finalSmall;
            inpaint(smallOrig, accumMask, finalSmall, 3.0, INPAINT_TELEA);
            resize(finalSmall, final, orig.size(), 0, 0, INTER_CUBIC);
        }

        string outPath = outFolder + "/hair_removed_" + path(imgPath).filename().string();
        imwrite(outPath, final);

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;

        #pragma omp critical
        {
            totalProcessed++;
            cout << "[Rank " << rank << "] Processed: " << imgPath
                 << " in " << elapsed.count() << " sec, saved as: " << outPath << endl;
        }
    }

    auto totalEnd = chrono::high_resolution_clock::now();
    chrono::duration<double> totalElapsed = totalEnd - totalStart;
    #pragma omp critical
    cout << "[Rank " << rank << "] Total images processed: " << totalProcessed
         << " in " << totalElapsed.count() << " sec" << endl;
}

void hairRemovalOCL(const vector<string>& imagePaths, const string& outFolder, int rank, int size) {
    int totalProcessed = 0;
    auto totalStart = chrono::high_resolution_clock::now();

    vector<string> localImages;
    for(size_t i = 0; i < imagePaths.size(); i++){
        if(i % size == rank)
            localImages.push_back(imagePaths[i]);
    }

    #pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < localImages.size(); i++){
        auto start = chrono::high_resolution_clock::now();

        string imgPath = localImages[i];
        Mat origCPU = imread(imgPath, IMREAD_COLOR);
        if(origCPU.empty()) continue;

        // Move to GPU (UMat)
        UMat orig;
        origCPU.copyTo(orig);

        // Downscale
        double scaleFactor = 800.0 / max(orig.cols, orig.rows);
        UMat smallOrig, smallGray;
        resize(orig, smallOrig, Size(), scaleFactor, scaleFactor, INTER_AREA);
        cvtColor(smallOrig, smallGray, COLOR_BGR2GRAY);

        // Multiscale mask (GPU)
        UMat accumMask = UMat::zeros(smallGray.size(), CV_8UC1);
        vector<int> kernelSizes = {15, 25, 35};

        for(auto kSize : kernelSizes){
            UMat khat, msk;
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

        UMat finalUMat;

        if(countNonZero(accumMask) == 0){
            orig.copyTo(finalUMat);
        } else {
            // inpaint still CPU-heavy → unavoidable
            Mat smallOrigCPU, maskCPU, finalSmallCPU;
            smallOrig.copyTo(smallOrigCPU);
            accumMask.copyTo(maskCPU);

            inpaint(smallOrigCPU, maskCPU, finalSmallCPU, 3.0, INPAINT_TELEA);

            UMat finalSmall;
            finalSmallCPU.copyTo(finalSmall);

            resize(finalSmall, finalUMat, orig.size(), 0, 0, INTER_CUBIC);
        }

        // Back to CPU for saving
        Mat final;
        finalUMat.copyTo(final);

        string outPath = outFolder + "/hair_removed_" + path(imgPath).filename().string();
        imwrite(outPath, final);

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;

        #pragma omp critical
        {
            totalProcessed++;
            cout << "[Rank " << rank << "] Processed: " << imgPath
                 << " in " << elapsed.count() << " sec" << endl;
        }
    }

    auto totalEnd = chrono::high_resolution_clock::now();
    chrono::duration<double> totalElapsed = totalEnd - totalStart;

    #pragma omp critical
    cout << "[Rank " << rank << "] Total images processed: "
         << totalProcessed << " in " << totalElapsed.count() << " sec" << endl;
}
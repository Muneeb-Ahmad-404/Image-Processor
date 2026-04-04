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

void hairRemovalOCL(const vector<string>& imagePaths, const string& outFolder, int rank, int size);

int main(int argc, char** argv){
    cv::ocl::setUseOpenCL(true);
    cout << "OpenCL enabled: " << cv::ocl::useOpenCL() << endl;

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string folderPath = "../images";
    string outFolder = "../processed_images/ocl";
    create_directories(outFolder);

    vector<string> imagePaths;
    for(const auto& entry : directory_iterator(folderPath)){
        if(entry.path().extension() == ".jpg" || entry.path().extension() == ".png")
            imagePaths.push_back(entry.path().string());
    }

    hairRemovalOCL(imagePaths, outFolder, rank, size);

    MPI_Finalize();
    return 0;
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
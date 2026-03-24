#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <filesystem>
#include <iostream>

using namespace cv;
using namespace std;
using namespace filesystem;

int main()
{
    string folderPath = "images/";
    vector<Mat> imgs;
    int totalImages = 0;

    if(!exists(folderPath) || !is_directory(folderPath)){
        cout<< "Directory does not exist: " << folderPath << endl;
        return 1;
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
    return 0;
}
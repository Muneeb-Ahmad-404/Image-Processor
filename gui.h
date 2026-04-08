#ifndef GUI_H
#define GUI_H

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QSlider>
#include <QStringList>
#include <QThread>
#include <QProcess>
#include <QTimer>
#include <chrono>

class Worker : public QThread {
    Q_OBJECT
public:
    Worker(const QStringList &images, const QString &method, QObject *parent = nullptr)
        : QThread(parent), imageList(images), methodName(method) {}

signals:
    void processingDone();

protected:
    void run() override {
        QString folder;
        if (methodName == "Serial") folder = "serial";
        else if (methodName == "OMP") folder = "omp";
        else if (methodName == "MPI") folder = "mpi";
        else folder = "ocl";

        for (const QString &img : imageList) {
            QString inputPath  = "images/" + img;
            QString outputPath = "processed_images/" + folder + "/hair_removed_" + img;

            QString program;
            QStringList args;

            if (methodName == "MPI" || methodName == "OCL") {
                program = "mpirun";
                args << "-np" << "4" << "./backend/" + methodName.toLower();
            } else {
                program = "./backend/" + methodName.toLower();
            }

            args << inputPath << outputPath;

            QProcess proc;
            proc.start(program, args);
            proc.waitForStarted();
            proc.waitForFinished(-1);
        }

        emit processingDone();
    }

private:
    QStringList imageList;
    QString methodName;
};


class GUI : public QWidget {
    Q_OBJECT
public:
    explicit GUI(QWidget *parent = nullptr);

private slots:
    void runAll();
    void showCurrentImage();
    void pollOutputFolder();

private:
    void loadImages();

    QLabel     *originalLabel;
    QLabel     *processedLabel;
    QLabel     *timeLabel;
    QComboBox  *methodBox;
    QSlider    *slider;
    QTimer     *pollTimer;

    QStringList imageList;
    QString     currentImage;
    QString     currentFolder;

    std::chrono::high_resolution_clock::time_point startTime;
};

#endif
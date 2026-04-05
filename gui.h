#ifndef GUI_H
#define GUI_H

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QSlider>
#include <QStringList>
#include <QThread>
#include <QElapsedTimer>
#include <QProcess>

class Worker : public QThread {
    Q_OBJECT
public:
    Worker(const QStringList &images, const QString &method, QObject *parent = nullptr)
        : QThread(parent), imageList(images), methodName(method) {}

signals:
    void updateImage(const QString &orig, const QString &proc);
    void finished(double);

protected:
    void run() override {
        QElapsedTimer timer;
        timer.start();

        QString folder;
        if(methodName=="Serial") folder="serial";
        else if(methodName=="OMP") folder="omp";
        else if(methodName=="MPI") folder="mpi";
        else folder="ocl";

        for(const QString &img : imageList) {
            QString inputPath = "images/" + img;
            QString outputPath = "processed_images/" + folder + "/hair_removed_" + img;

            QStringList args;
            if(methodName=="MPI" || methodName=="OCL") {
                args << "-np" << "4" << "./backend/" + methodName.toLower();
            } else {
                args << "./backend/" + methodName.toLower();
            }
            args << inputPath << outputPath;

            QProcess proc;
            proc.start(methodName=="MPI" || methodName=="OCL" ? "mpirun" : "./backend/" + methodName.toLower(), args);
            proc.waitForFinished(-1);

            emit updateImage(inputPath, outputPath);
        }

        emit finished(timer.elapsed()/1000.0);
    }

private:
    QStringList imageList;
    QString methodName;
};


class GUI : public QWidget {
    Q_OBJECT
public:
    GUI(QWidget *parent = nullptr);

private slots:
    void runAll();
    void showCurrentImage();

private:
    void loadImages();

    QLabel *originalLabel;
    QLabel *processedLabel;
    QLabel *timeLabel;
    QComboBox *methodBox;
    QSlider *slider;
    QStringList imageList;
    QString currentImage;
};

#endif // GUI_H
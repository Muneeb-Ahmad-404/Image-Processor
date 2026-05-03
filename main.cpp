#include "gui.h"
#include <QApplication>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDir>
#include <QPixmap>
#include <QFile>
#include <chrono>

using namespace std::chrono;

// ================= GUI =================

GUI::GUI(QWidget *parent) : QWidget(parent) {
    setWindowTitle("Hair Removal Benchmark");

    QVBoxLayout *mainLayout    = new QVBoxLayout(this);
    QHBoxLayout *imageLayout   = new QHBoxLayout();
    QVBoxLayout *sliderLayout  = new QVBoxLayout();
    QHBoxLayout *controlLayout = new QHBoxLayout();

    originalLabel  = new QLabel("Original Image");
    processedLabel = new QLabel("Processed Image");
    originalLabel->setFixedSize(400, 400);
    processedLabel->setFixedSize(400, 400);

    imageLayout->addWidget(originalLabel);
    imageLayout->addWidget(processedLabel);

    slider = new QSlider(Qt::Vertical);
    sliderLayout->addWidget(slider);
    imageLayout->addLayout(sliderLayout);

    methodBox = new QComboBox();
    methodBox->addItems({"Serial", "OMP", "MPI", "OCL"});

    QPushButton *runButton = new QPushButton("Run");
    timeLabel = new QLabel("Time: 0.000 sec");

    controlLayout->addWidget(methodBox);
    controlLayout->addWidget(runButton);
    controlLayout->addWidget(timeLabel);

    mainLayout->addLayout(imageLayout);
    mainLayout->addLayout(controlLayout);

    pollTimer = new QTimer(this);
    pollTimer->setInterval(500);
    connect(pollTimer, &QTimer::timeout, this, &GUI::pollOutputFolder);

    loadImages();
    slider->setMaximum(qMax(0, imageList.size() - 1));

    connect(runButton, &QPushButton::clicked, this, &GUI::runAll);
    connect(slider,    &QSlider::valueChanged, this, &GUI::showCurrentImage);
    connect(methodBox, &QComboBox::currentTextChanged, this, &GUI::showCurrentImage);

    if (!imageList.isEmpty())
        showCurrentImage();
}

// ================= LOAD =================

void GUI::loadImages() {
    QDir dir("images");
    QStringList filters;
    filters << "*.jpg" << "*.png" << "*.jpeg";
    imageList = dir.entryList(filters, QDir::Files);
}

// ================= DISPLAY =================

void GUI::showCurrentImage() {
    int idx = slider->value();
    if (idx < 0 || idx >= imageList.size()) return;

    currentImage = imageList[idx];

    QString origPath = "images/" + currentImage;
    if (QFile::exists(origPath)) {
        QPixmap pix(origPath);
        originalLabel->setPixmap(pix.scaled(originalLabel->size(), Qt::KeepAspectRatio));
    }

    QString method = methodBox->currentText();
    if      (method == "Serial") currentFolder = "serial";
    else if (method == "OMP")    currentFolder = "omp";
    else if (method == "MPI")    currentFolder = "mpi";
    else                         currentFolder = "ocl";

    QString procPath = "processed_images/" + currentFolder + "/hair_removed_" + currentImage;

    if (QFile::exists(procPath)) {
        QPixmap pix(procPath);
        processedLabel->setPixmap(pix.scaled(processedLabel->size(), Qt::KeepAspectRatio));
    } else {
        processedLabel->setText("Not yet processed");
    }
}

// ================= POLL =================

void GUI::pollOutputFolder() {
    QString outputDir = "processed_images/" + currentFolder;
    QDir dir(outputDir);
    QStringList filters;
    filters << "*.jpg" << "*.png" << "*.jpeg";
    int count = dir.entryList(filters, QDir::Files).size();

    if (count >= imageList.size()) {
        pollTimer->stop();

        auto end = high_resolution_clock::now();
        double seconds = duration<double>(end - startTime).count();
        timeLabel->setText("Time: " + QString::number(seconds, 'f', 3) + " sec");

        showCurrentImage();
    }
}

// ================= RUN =================

void GUI::runAll() {
    if (imageList.isEmpty()) return;

    QString method = methodBox->currentText();
    if      (method == "Serial") currentFolder = "serial";
    else if (method == "OMP")    currentFolder = "omp";
    else if (method == "MPI")    currentFolder = "mpi";
    else                         currentFolder = "ocl";

    // Clear the output folder so the count starts from 0
    QString outputDir = "processed_images/" + currentFolder;
    QDir dir(outputDir);
    QStringList filters;
    filters << "*.jpg" << "*.png" << "*.jpeg";
    for (const QString &f : dir.entryList(filters, QDir::Files))
        dir.remove(f);

    timeLabel->setText("Time: running...");

    startTime = high_resolution_clock::now();

    Worker *worker = new Worker(imageList, method, this);
    connect(worker, &Worker::processingDone, worker, &QObject::deleteLater);
    worker->start();

    pollTimer->start();
}

// ================= MAIN =================

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    GUI window;
    window.show();
    return app.exec();
}

#include "gui.moc"gi
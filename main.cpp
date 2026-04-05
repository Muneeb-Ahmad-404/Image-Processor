#include "gui.h"
#include <QApplication>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDir>
#include <QPixmap>
#include <QFile>
#include <QMetaObject>

GUI::GUI(QWidget *parent) : QWidget(parent) {
    setWindowTitle("Hair Removal Benchmark");

    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    QHBoxLayout *imageLayout = new QHBoxLayout();
    QVBoxLayout *sliderLayout = new QVBoxLayout();
    QHBoxLayout *controlLayout = new QHBoxLayout();

    originalLabel = new QLabel("Original Image");
    processedLabel = new QLabel("Processed Image");
    originalLabel->setFixedSize(400,400);
    processedLabel->setFixedSize(400,400);
    imageLayout->addWidget(originalLabel);
    imageLayout->addWidget(processedLabel);

    slider = new QSlider(Qt::Vertical);
    sliderLayout->addWidget(slider);
    imageLayout->addLayout(sliderLayout);

    methodBox = new QComboBox();
    methodBox->addItems({"Serial","OMP","MPI","OCL"});
    QPushButton *runButton = new QPushButton("Run");
    timeLabel = new QLabel("Time: 0.0 sec");
    controlLayout->addWidget(methodBox);
    controlLayout->addWidget(runButton);
    controlLayout->addWidget(timeLabel);

    mainLayout->addLayout(imageLayout);
    mainLayout->addLayout(controlLayout);

    loadImages();
    slider->setMaximum(imageList.size()-1);

    connect(runButton, &QPushButton::clicked, this, &GUI::runAll);
    connect(slider, &QSlider::valueChanged, this, &GUI::showCurrentImage);

    if(!imageList.isEmpty())
        showCurrentImage();
}

void GUI::loadImages() {
    QDir dir("images");
    QStringList filters; 
    filters << "*.jpg" << "*.png" << "*.jpeg";
    imageList = dir.entryList(filters, QDir::Files);
}

void GUI::showCurrentImage() {
    int idx = slider->value();
    if(idx<0 || idx>=imageList.size()) return;
    currentImage = imageList[idx];

    QString origPath = "images/" + currentImage;
    if(QFile::exists(origPath)) {
        QPixmap pix(origPath);
        originalLabel->setPixmap(pix.scaled(originalLabel->size(), Qt::KeepAspectRatio));
    }

    QString folder;
    QString method = methodBox->currentText();
    if(method=="Serial") folder="serial";
    else if(method=="OMP") folder="omp";
    else if(method=="MPI") folder="mpi";
    else folder="ocl";

    QString procPath = "processed_images/"+folder+"/hair_removed_"+currentImage;
    if(QFile::exists(procPath)) {
        QPixmap pix(procPath);
        processedLabel->setPixmap(pix.scaled(processedLabel->size(), Qt::KeepAspectRatio));
    }
}

void GUI::runAll() {
    if(imageList.isEmpty()) return;

    methodBox->setEnabled(false);

    Worker *worker = new Worker(imageList, methodBox->currentText(), this);
    connect(worker, &Worker::updateImage, this, [=](const QString &orig, const QString &proc){
        if(currentImage == QFileInfo(orig).fileName()) {
            QPixmap pix(proc);
            processedLabel->setPixmap(pix.scaled(processedLabel->size(), Qt::KeepAspectRatio));
        }
    });
    connect(worker, &Worker::finished, this, [=](double t){
        timeLabel->setText("Time: " + QString::number(t) + " sec");
        methodBox->setEnabled(true);
        worker->deleteLater();
    });
    worker->start();
}

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    GUI window;
    window.show();
    return app.exec();
}

#include "gui.moc"
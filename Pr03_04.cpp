#include <iostream>
#include <mpi.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int mandelbrot(double cr, double ci, int max_iter) {
    double zr = 0.0, zi = 0.0;
    int n = 0;

    while (n < max_iter && zr * zr + zi * zi < 4.0) {
        double temp = zr * zr - zi * zi + cr;
        zi = 2.0 * zr * zi + ci;
        zr = temp;
        n++;
    }

    return n;
}

Vec3b getPastelColor(int n) {
    int r = 127 + 63 * sin(0.1 * n);
    int g = 127 + 63 * sin(0.2 * n);
    int b = 127 + 63 * sin(0.3 * n);
    return Vec3b(r, g, b);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = 800;
    int height = 800;
    double xmin = -2.0;
    double xmax = 1.0;
    double ymin = -1.5;
    double ymax = 1.5;
    int max_iter = 1000;

    Mat image(height, width, CV_8UC3);

    int rows_per_process = height / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? height : start_row + rows_per_process;

    for (int y = start_row; y < end_row; y++) {
        for (int x = 0; x < width; x++) {
            double cr = xmin + (xmax - xmin) * x / width;
            double ci = ymin + (ymax - ymin) * y / height;
            int iterations = mandelbrot(cr, ci, max_iter);

            if (iterations == max_iter) {
                image.at<Vec3b>(y, x) = Vec3b(0, 0, 0); 
            }
            else {
                image.at<Vec3b>(y, x) = getPastelColor(iterations); 
            }
        }
    }

    if (rank == 0) {
        Mat result_image(height, width, CV_8UC3);
        MPI_Gather(image.data, width * rows_per_process * 3, MPI_UNSIGNED_CHAR,
            result_image.data, width * rows_per_process * 3, MPI_UNSIGNED_CHAR,
            0, MPI_COMM_WORLD);

        namedWindow("Mandelbrot Fractal", WINDOW_NORMAL);
        imshow("Mandelbrot Fractal", result_image);
        waitKey(0);
    }

    MPI_Finalize();
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define MAX_COLOR 255

typedef struct {
    int width, height;
    unsigned char *data;
} Image;

// Function Prototypes
Image readPPM(const char *filename);
void writePPM(const char *filename, Image img);
void applyBlur(Image *img, int use_openmp);
void applyEdgeDetection(Image *img, int use_openmp);

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <input.ppm> <output.ppm> <filter_type>\n", argv[0]);
        printf("Filter types:\n  1 - Blur\n  2 - Edge Detection\n");
        return 1;
    }

    Image img = readPPM(argv[1]);
    int filter_type = atoi(argv[3]);

    // Allow OpenMP to adjust the number of threads dynamically
    omp_set_dynamic(1);

    // Serial Execution
    double start = omp_get_wtime();
    if (filter_type == 1) {
        applyBlur(&img, 0);
    } else if (filter_type == 2) {
        applyEdgeDetection(&img, 0);
    }
    double end = omp_get_wtime();
    double time_serial = end - start;
    
    printf("Execution time without OpenMP: %f seconds\n", time_serial);

    // Reload the image for parallel execution
    img = readPPM(argv[1]);

    // Parallel Execution
    start = omp_get_wtime();
    if (filter_type == 1) {
        applyBlur(&img, 1);
    } else if (filter_type == 2) {
        applyEdgeDetection(&img, 1);
    }
    end = omp_get_wtime();
    double time_parallel = end - start;

    // Print the number of threads used
    int num_threads;
    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }

    printf("Execution time with OpenMP: %f seconds\n", time_parallel);
    printf("Speedup: %.2fx\n", time_serial / time_parallel);
    printf("Number of threads used: %d\n", num_threads);

    writePPM(argv[2], img);
    free(img.data);
    return 0;
}

// Function to read a PPM image
Image readPPM(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        exit(1);
    }

    char format[3];
    int w, h, maxColor;
    fscanf(fp, "%s\n%d %d\n%d\n", format, &w, &h, &maxColor);

    if (strcmp(format, "P6") != 0 || maxColor != MAX_COLOR) {
        printf("Error: Unsupported image format.\n");
        fclose(fp);
        exit(1);
    }

    Image img;
    img.width = w;
    img.height = h;
    img.data = (unsigned char *)malloc(3 * w * h);
    fread(img.data, 3, w * h, fp);
    fclose(fp);
    
    return img;
}

// Function to write a PPM image
void writePPM(const char *filename, Image img) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error: Cannot write to file %s\n", filename);
        exit(1);
    }

    fprintf(fp, "P6\n%d %d\n%d\n", img.width, img.height, MAX_COLOR);
    fwrite(img.data, 3, img.width * img.height, fp);
    fclose(fp);
}

// Apply Blur Filter
void applyBlur(Image *img, int use_openmp) {
    int w = img->width, h = img->height;
    unsigned char *temp = (unsigned char *)malloc(3 * w * h);

    #pragma omp parallel for collapse(2) schedule(dynamic) if(use_openmp)
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int sumR = 0, sumG = 0, sumB = 0;
            int count = 0;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int idx = 3 * ((y + dy) * w + (x + dx));
                    sumR += img->data[idx];
                    sumG += img->data[idx + 1];
                    sumB += img->data[idx + 2];
                    count++;
                }
            }

            int idx = 3 * (y * w + x);
            temp[idx] = sumR / count;
            temp[idx + 1] = sumG / count;
            temp[idx + 2] = sumB / count;
        }
    }

    memcpy(img->data, temp, 3 * w * h);
    free(temp);
}

// Apply Edge Detection Filter
void applyEdgeDetection(Image *img, int use_openmp) {
    int w = img->width, h = img->height;
    unsigned char *temp = (unsigned char *)malloc(3 * w * h);

    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    #pragma omp parallel for collapse(2) schedule(dynamic) if(use_openmp)
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int sumRx = 0, sumGx = 0, sumBx = 0;
            int sumRy = 0, sumGy = 0, sumBy = 0;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int idx = 3 * ((y + dy) * w + (x + dx));
                    int weightX = Gx[dy + 1][dx + 1];
                    int weightY = Gy[dy + 1][dx + 1];

                    sumRx += img->data[idx] * weightX;
                    sumGx += img->data[idx + 1] * weightX;
                    sumBx += img->data[idx + 2] * weightX;

                    sumRy += img->data[idx] * weightY;
                    sumGy += img->data[idx + 1] * weightY;
                    sumBy += img->data[idx + 2] * weightY;
                }
            }

            int idx = 3 * (y * w + x);
            temp[idx] = fmin(sqrt(sumRx * sumRx + sumRy * sumRy), 255);
            temp[idx + 1] = fmin(sqrt(sumGx * sumGx + sumGy * sumGy), 255);
            temp[idx + 2] = fmin(sqrt(sumBx * sumBx + sumBy * sumBy), 255);
        }
    }

    memcpy(img->data, temp, 3 * w * h);
    free(temp);
}

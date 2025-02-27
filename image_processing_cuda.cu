#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16  // CUDA block size

// Sobel filter kernels (Edge Detection)
__constant__ int Gx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

__constant__ int Gy[3][3] = {
    {-1, -2, -1},
    {0, 0, 0},
    {1, 2, 1}
};

// Struct to hold image data
typedef struct {
    int width, height;
    unsigned char *data;
} Image;

// Function to read PPM file
void readPPM(const char *filename, Image *img) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error opening file!\n");
        exit(1);
    }

    char format[3];
    int maxColor;
    fscanf(fp, "%s\n%d %d\n%d\n", format, &img->width, &img->height, &maxColor);
    img->data = (unsigned char *)malloc(3 * img->width * img->height);
    fread(img->data, 3, img->width * img->height, fp);
    fclose(fp);
}

// Function to write PPM file
void writePPM(const char *filename, Image *img) {
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P6\n%d %d\n255\n", img->width, img->height);
    fwrite(img->data, 3, img->width * img->height, fp);
    fclose(fp);
}

// CUDA Kernel for Blurring
__global__ void blurCUDA(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int sumR = 0, sumG = 0, sumB = 0;
        int count = 0;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int idx = 3 * ((y + dy) * width + (x + dx));
                sumR += input[idx];
                sumG += input[idx + 1];
                sumB += input[idx + 2];
                count++;
            }
        }

        int idx = 3 * (y * width + x);
        output[idx] = sumR / count;
        output[idx + 1] = sumG / count;
        output[idx + 2] = sumB / count;
    }
}

// CUDA Kernel for Edge Detection (Sobel Filter)
__global__ void edgeDetectionCUDA(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int sumRx = 0, sumGx = 0, sumBx = 0;
        int sumRy = 0, sumGy = 0, sumBy = 0;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int idx = 3 * ((y + dy) * width + (x + dx));
                int weightX = Gx[dy + 1][dx + 1];
                int weightY = Gy[dy + 1][dx + 1];

                sumRx += input[idx] * weightX;
                sumGx += input[idx + 1] * weightX;
                sumBx += input[idx + 2] * weightX;

                sumRy += input[idx] * weightY;
                sumGy += input[idx + 1] * weightY;
                sumBy += input[idx + 2] * weightY;
            }
        }

        int idx = 3 * (y * width + x);
        output[idx] = min(sqrtf(sumRx * sumRx + sumRy * sumRy), 255.0f);
        output[idx + 1] = min(sqrtf(sumGx * sumGx + sumGy * sumGy), 255.0f);
        output[idx + 2] = min(sqrtf(sumBx * sumBx + sumBy * sumBy), 255.0f);
    }
}

// Apply CUDA Image Processing with Timing
void applyCUDAFilter(Image *img, void (*kernel)(unsigned char*, unsigned char*, int, int)) {
    unsigned char *d_input, *d_output;
    size_t size = 3 * img->width * img->height * sizeof(unsigned char);

    // Allocate memory on GPU
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy data to GPU
    cudaMemcpy(d_input, img->data, size, cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((img->width + BLOCK_SIZE - 1) / BLOCK_SIZE, (img->height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Start timing
    cudaEventRecord(start);

    // Launch CUDA kernel
    kernel<<<gridSize, blockSize>>>(d_input, d_output, img->width, img->height);
    cudaDeviceSynchronize();

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Processing time: %.6f seconds\n", milliseconds / 1000.0);

    // Copy result back to CPU
    cudaMemcpy(img->data, d_output, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
}

// Main Function
int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <input.ppm> <output.ppm> <filter>\n", argv[0]);
        printf("Filters: 1 (Blur), 2 (Edge Detection)\n");
        return 1;
    }

    Image img;
    readPPM(argv[1], &img);

    int filter_type = atoi(argv[3]);

    if (filter_type == 1) {
        applyCUDAFilter(&img, blurCUDA);
    } else if (filter_type == 2) {
        applyCUDAFilter(&img, edgeDetectionCUDA);
    } else {
        printf("Unknown filter type: %d\n", filter_type);
        free(img.data);
        return 1;
    }

    writePPM(argv[2], &img);
    free(img.data);
    return 0;
}
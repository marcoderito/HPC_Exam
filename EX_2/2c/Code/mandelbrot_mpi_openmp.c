#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MAX_ITER 1000

int main(int argc, char *argv[]) {
    if (argc != 8) {
        printf("Usage: %s n_x n_y x_L y_L x_R y_R I_max\n", argv[0]);
        return 1;
    }

    int n_x = atoi(argv[1]);
    int n_y = atoi(argv[2]);
    double x_L = atof(argv[3]);
    double y_L = atof(argv[4]);
    double x_R = atof(argv[5]);
    double y_R = atof(argv[6]);
    int I_max = atoi(argv[7]);

    FILE *output_file = fopen("output_file.pgm", "wb");
    if (output_file == NULL) {
        printf("Error: Unable to open output file\n");
        return 1;
    }

    // Calculate delta x and delta y
    double delta_x = (x_R - x_L) / n_x;
    double delta_y = (y_R - y_L) / n_y;

    // Allocate memory for the pixel matrix
    char *M = (char *)malloc(n_x * n_y * sizeof(char));
    if (M == NULL) {
        printf("Error: Unable to allocate memory\n");
        return 1;
    }

    // Iterate over each pixel
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n_y; i++) {
        for (int j = 0; j < n_x; j++) {
            double c_re = x_L + j * delta_x;
            double c_im = y_L + i * delta_y;

            double z_re = 0, z_im = 0;
            int iter = 0;
            while (z_re * z_re + z_im * z_im <= 4 && iter < I_max) {
                double temp = z_re * z_re - z_im * z_im + c_re;
                z_im = 2 * z_re * z_im + c_im;
                z_re = temp;
                iter++;
            }

            // Assign pixel value based on iteration count
            if (iter == I_max) {
                M[i * n_x + j] = 0;  // Inside Mandelbrot set
            } else {
                M[i * n_x + j] = iter;  // Outside Mandelbrot set
            }
        }
    }

    // Write the pixel matrix to the output file
    fprintf(output_file, "P5\n%d %d\n%d\n", n_x, n_y, I_max);
    fwrite(M, sizeof(char), n_x * n_y, output_file);

    // Cleanup
    free(M);
    fclose(output_file);

    return 0;
}

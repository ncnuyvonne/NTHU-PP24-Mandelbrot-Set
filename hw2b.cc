#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <immintrin.h>
void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);
    int size, rank;
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    int chunksize = height / size;
    int rest = height%size;
    int start = rank * chunksize + ((rank < rest) ? rank : rest);
    int range = ((rank < rest) ? chunksize+1 : chunksize);
    int end = start + range;
    int* part_image = (int*)malloc(width * range * sizeof(int));
    assert(part_image);

    /* mandelbrot set */
    double delta_y = (upper - lower) / height;
    double delta_x = (right - left) / width;
    double y0;
    #pragma omp parallel for num_threads(range) reduction(+:y0)
    for (int j = start; j < end; ++j) {
        double y0 = j * delta_y + lower;
        int rows = (j-start) * width;
        for (int i = 0; i < width; i += 8) {
            // Initialize AVX-512 vectors for 8 parallel calculations
            __m512d x0 = _mm512_set_pd(
                (i + 7) * delta_x + left,
                (i + 6) * delta_x + left,
                (i + 5) * delta_x + left,
                (i + 4) * delta_x + left,
                (i + 3) * delta_x + left,
                (i + 2) * delta_x + left,
                (i + 1) * delta_x + left,
                i * delta_x + left
            );
            __m512d y0_vec = _mm512_set1_pd(y0);
            __m512d x = _mm512_setzero_pd();
            __m512d y = _mm512_setzero_pd();
            __m512d length_squared = _mm512_setzero_pd();
            __m512d two = _mm512_set1_pd(2.0);
            __m512d four = _mm512_set1_pd(4.0);
            int repeats[8] = {0, 0, 0, 0, 0, 0, 0, 0};

            // Iterate until escape condition is met for each point
            for (int k = 0; k < iters; ++k) {
                __m512d x_squared = _mm512_mul_pd(x, x);
                __m512d y_squared = _mm512_mul_pd(y, y);
                __m512d xy = _mm512_mul_pd(x, y);

                length_squared = _mm512_add_pd(x_squared, y_squared);

                // Check if all points have escaped
                __mmask8 mask = _mm512_cmp_pd_mask(length_squared, four, _CMP_LT_OQ);
                if (mask == 0) {
                    break;  // All points have escaped
                }

                // Update x and y
                __m512d temp_x = _mm512_add_pd(_mm512_sub_pd(x_squared, y_squared), x0);
                y = _mm512_fmadd_pd(two, xy, y0_vec);  // y = 2 * x * y + y0
                x = temp_x;

                // Update repeat counts for points still in the set
                for (int idx = 0; idx < 8; ++idx) {
                    if (mask & (1 << idx)) {
                        repeats[idx]++;
                    }
                }
            }

            // Store results back to the image array
            for (int idx = 0; idx < 8 && (i + idx) < width; ++idx) {
                part_image[rows + i + idx] = repeats[idx];
            }
        }
    }

    int test = width * range;
    int* recvcounts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));
    #pragma omp for
    for (int i = 0; i < size; i++) {
        int local_chunk = (i < rest) ? (chunksize + 1) : chunksize;
        recvcounts[i] = local_chunk * width;
        displs[i] = (i == 0) ? 0 : (displs[i - 1] + recvcounts[i - 1]);
    }
    MPI_Gatherv(part_image, test, MPI_INT, image, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);


    /* draw and cleanup */
    if(rank==0)write_png(filename, iters, width, height, image);
    free(image);
    free(part_image);

    MPI_Finalize();
}

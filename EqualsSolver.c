#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <string.h>

void print_x_i(const double** a, const int i, const int n, const int m,
const int rank) {
    if (i < 1 || i > n) return;
    if (rank == (i / m)) {
        printf("x[%d] = %e\n", i, a[(i-1) % m][n-1]);

    }
}

int main(int argc, char** argv) {

    if (argc < 2) {
        return 1;
    }

    int n = atoi(argv[1]);
    if (n % 4 != 0) {
        return 1;
    }
    int m = n / 4;

    double **a = (double**)calloc(m, sizeof(double*));
    int i, j;
    for (i = 0; i < m; i++) {
        a[i] = (double*)calloc(n+1, sizeof(double));
    }
    double* kline = calloc((n+1), sizeof(double));


    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    srand((rank+1)+time(NULL));

    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            a[i][j] = (double)rand()/RAND_MAX;
            a[i][n] += a[i][j]*(j+1);
        }
    }
    // print_m(a, n+1, m, rank);
    int k;
    for (k = 0; k < n; ++k) {
        int host = k / m;
        int row = k % m;

        if (rank == host) {
            double diag = a[row][row+rank*m];

            for (i = 0; i < (n+1); ++i) {
                a[row][i] /= diag;
            }
            memcpy(kline, a[row], sizeof(double)*(n+1));
        }

        MPI_Bcast(kline, n+1, MPI_DOUBLE, host, MPI_COMM_WORLD);

        for (i = 0; i < m; ++i) {
            if (rank == host && i == row) continue;
            double mad = a[i][k];

            for (j = 0; j < n+1; ++j) {
                a[i][j] -= mad*kline[j];
            }
        }
    }

    print_x_i(a, 1977, n+1, m, rank);
    print_x_i(a, 336, n+1, m, rank);
    print_x_i(a, 2784, n+1, m, rank);

    for (i = 0; i < m; i++) {
        free(a[i]);
    }

    free(a);

    MPI_Finalize();


    return 0;
}

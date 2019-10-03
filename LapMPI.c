#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <memory.h>
#include "mpi.h"

int main(int argc, char* argv[])
{
    int procRank, procCount;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &procCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    int N = 5000;
    int i,j,t;
    int itCount = 10000;

    double norm;
    double eps = 1e-6;

    double* arr = new double[N*(N/procCount)+2*N]();
    double* arr1 = new double[N*(N/procCount)];

    std::fill_n(arr1,N*(N/procCount),1);

    if(procRank == 0)
    {
        double* sol = new double[N*N]();
        for(i=0;i<N;i++)
        {
            sol[i] = 1;
            sol[N*(N-1)+i]=1;
            sol[i*N]=1;
            sol[i*N+(N-1)]=1;
        }
        MPI_Scatter(sol,N*N/(procCount),MPI_DOUBLE,&arr[N],N*N/(procCount),MPI_DOUBLE,0,MPI_COMM_WORLD);

        for(t=0;t<itCount;t++)
        {
            norm = 0;
             MPI_Sendrecv(&arr[N*(N/procCount)],N,MPI_DOUBLE,1,0,
                &arr[N*(N/procCount)+N],N,MPI_DOUBLE,1,0,
                MPI_COMM_WORLD,MPI_STATUS_IGNORE);

            for(i=1;i<N/procCount;i++)
            {
                for(j=1;j<N-1;j++)
                {
                    arr1[i*N+j] = 0.25*(arr[(i+2)*N+j]+arr[i*N+j]+arr[(i+1)*N+j+1]+arr[(i+1)*N+j-1]);
                    norm+=(arr1[i*N+j]-arr[(i+1)*N+j])*(arr1[i*N+j]-arr[(i+1)*N+j]);
                }
            }

            memcpy(&arr[N],arr1,N*(N/procCount)*sizeof(double));

            MPI_Allreduce(&norm,&norm,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            if(sqrt(norm)<eps) break;
        }

        MPI_Gather(arr1,N*(N/procCount),MPI_DOUBLE,sol,N*(N/procCount),MPI_DOUBLE,0,MPI_COMM_WORLD);
        delete(sol);
        printf("N: %d кол-во итераций: %d невязка: %.10e \n",N,t,sqrt(norm));

    } else {
        int shift = 0;

        for(t=0;t<itCount;t++){
            norm = 0;
            if(procRank == procCount-1){
                shift=1;
                MPI_Sendrecv(&arr[N],N,MPI_DOUBLE,procRank-1,0,
                        arr,N,MPI_DOUBLE,procRank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            } else {
                MPI_Sendrecv(&arr[N],N,MPI_DOUBLE,procRank-1,0,
                    &arr[N*(N/procCount)+N],N,MPI_DOUBLE,procRank+1,0,
                        MPI_COMM_WORLD,MPI_STATUS_IGNORE);

                MPI_Sendrecv(&arr[N*(N/procCount)],N,MPI_DOUBLE,procRank+1,0,
                    arr,N,MPI_DOUBLE,procRank-1,0,
                        MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
            for(i=0;i<N/procCount-shift;i++){
               for(j=1;j<N-1;j++){
                    arr1[i*N+j] = 0.25*(arr[(i+2)*N+j]+arr[i*N+j]+arr[(i+1)*N+j+1]+arr[(i+1)*N+j-1]);
                    norm+=(arr1[i*N+j]-arr[(i+1)*N+j])*(arr1[i*N+j]-arr[(i+1)*N+j]);
                }
            }

            memcpy(&arr[N],arr1,N*(N/procCount)*sizeof(double));
            MPI_Allreduce(&norm,&norm,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            if(sqrt(norm)<eps) break;
        }

        MPI_Gather(arr1,N*(N/procCount),MPI_DOUBLE,NULL,0,MPI_DOUBLE,0,MPI_COMM_WORLD);

    }
    delete(arr);
    delete(arr1);

    MPI_Finalize();


    return 0;
}

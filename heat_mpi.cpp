#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>


#define IDX(i, k, stride) ((i) * (stride) + (k))


inline double StencilFTCS(const double* phi, int i, int k, int stride, double dx2i, double dy2i) {
    return (phi[IDX(i + 1, k, stride)] + phi[IDX(i - 1, k, stride)] - 2.0 * phi[IDX(i, k, stride)]) * dy2i +
           (phi[IDX(i, k + 1, stride)] + phi[IDX(i, k - 1, stride)] - 2.0 * phi[IDX(i, k, stride)]) * dx2i;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    
    int imax = 80;
    int kmax = 80;
    int itmax = 20000;
    if (argc >= 2) {
        imax = std::atoi(argv[1]);
        kmax = imax;
    }
    if (argc >= 3) {
        itmax = std::atoi(argv[2]);
    }

    const double eps = 1.0e-08;
    const double dx = 1.0 / kmax;
    const double dy = 1.0 / imax;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dx2i = 1.0 / dx2;
    const double dy2i = 1.0 / dy2;
    const double dt = std::min(dx2, dy2) / 4.0;

    
    int p_rows = 1, p_cols = p;
    for (int r = (int)std::sqrt(p); r >= 1; --r) {
        if (p % r == 0) {
            p_rows = r;
            p_cols = p / r;
            break;
        }
    }


    int row = rank / p_cols;
    int col = rank % p_cols;

    
    int north = (row > 0) ? rank - p_cols : MPI_PROC_NULL;
    int south = (row < p_rows - 1) ? rank + p_cols : MPI_PROC_NULL;
    int west  = (col > 0) ? rank - 1 : MPI_PROC_NULL;
    int east  = (col < p_cols - 1) ? rank + 1 : MPI_PROC_NULL;

    
    int total_i = imax - 1;
    int total_k = kmax - 1;

    
    int base_i = total_i / p_rows;
    int rem_i  = total_i % p_rows;
    int n_loc  = base_i + (row < rem_i ? 1 : 0);
    int i_start = 1 + row * base_i + std::min(row, rem_i);

    
    int base_k = total_k / p_cols;
    int rem_k  = total_k % p_cols;
    int k_loc  = base_k + (col < rem_k ? 1 : 0);
    int k_start = 1 + col * base_k + std::min(col, rem_k);

    int stride = k_loc + 2;
    std::vector<double> phi((n_loc + 2) * stride, 0.0);
    std::vector<double> phin((n_loc + 2) * stride, 0.0);

    
    for (int li = 0; li <= n_loc + 1; ++li) {
        int gi = i_start + li - 1;
        for (int lk = 0; lk <= k_loc + 1; ++lk) {
            int gk = k_start + lk - 1;
            double val = 0.0;
            if (gk == kmax) {
                val = 1.0;
            } else if (gk == 0) {
                val = 0.0;
            } else if (gi == 0 || gi == imax) {
                val = gk * dx;
            }
            phi[IDX(li, lk, stride)]  = val;
            phin[IDX(li, lk, stride)] = val;
        }
    }


    
    MPI_Datatype col_type;
    MPI_Type_vector(n_loc, 1, stride, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();
    double t_comp = 0.0;
    double t_comm = 0.0;

    int it = 0;
    double dphimax_global = 1.0;



    
    for (it = 1; it <= itmax; ++it) {
        double dphimax_loc = 0.0;
        MPI_Request reqs[8];

        // comm no bloqueada
        double tc0 = MPI_Wtime();

        MPI_Irecv(&phi[IDX(0, 1, stride)], k_loc, MPI_DOUBLE, north, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&phi[IDX(n_loc + 1, 1, stride)], k_loc, MPI_DOUBLE, south, 1, MPI_COMM_WORLD, &reqs[1]);
        MPI_Irecv(&phi[IDX(1, 0, stride)], 1, col_type,west, 2, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(&phi[IDX(1, k_loc + 1, stride)], 1, col_type, east,3, MPI_COMM_WORLD, &reqs[3]);

        
        MPI_Isend(&phi[IDX(1, 1, stride)],k_loc, MPI_DOUBLE, north, 1, MPI_COMM_WORLD, &reqs[4]);
        MPI_Isend(&phi[IDX(n_loc, 1, stride)],k_loc, MPI_DOUBLE, south, 0, MPI_COMM_WORLD, &reqs[5]);
        MPI_Isend(&phi[IDX(1, 1, stride)], 1, col_type, west,  3, MPI_COMM_WORLD, &reqs[6]);
        MPI_Isend(&phi[IDX(1, k_loc, stride)],1, col_type, east,  2, MPI_COMM_WORLD, &reqs[7]);
        t_comm += (MPI_Wtime() - tc0);

        // computo interno
        double tp0 = MPI_Wtime();
        for (int li = 2; li <= n_loc - 1; ++li) {
            for (int lk = 2; lk <= k_loc - 1; ++lk) {
                double dphi = StencilFTCS(phi.data(), li, lk, stride, dx2i, dy2i) * dt;
                dphimax_loc = std::max(dphimax_loc, dphi);
                phin[IDX(li, lk, stride)] = phi[IDX(li, lk, stride)] + dphi;
            }
        }
        t_comp += (MPI_Wtime() - tp0);

        tc0 = MPI_Wtime();
        MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
        t_comm += (MPI_Wtime() - tc0);

        // computo de bordes
        tp0 = MPI_Wtime();
        if (n_loc >= 1 && k_loc >= 1) {
            
            for (int lk = 1; lk <= k_loc; ++lk) {
                double dphi = StencilFTCS(phi.data(), 1, lk, stride, dx2i, dy2i) * dt;
                dphimax_loc = std::max(dphimax_loc, dphi);
                phin[IDX(1, lk, stride)] = phi[IDX(1, lk, stride)] + dphi;
            }
            
            if (n_loc > 1) {
                for (int lk = 1; lk <= k_loc; ++lk) {
                    double dphi = StencilFTCS(phi.data(), n_loc, lk, stride, dx2i, dy2i) * dt;
                    dphimax_loc = std::max(dphimax_loc, dphi);
                    phin[IDX(n_loc, lk, stride)] = phi[IDX(n_loc, lk, stride)] + dphi;
                }
            }
            
            for (int li = 2; li <= n_loc - 1; ++li) {
                double dphi_w = StencilFTCS(phi.data(), li, 1, stride, dx2i, dy2i) * dt;
                dphimax_loc = std::max(dphimax_loc, dphi_w);
                phin[IDX(li, 1, stride)] = phi[IDX(li, 1, stride)] + dphi_w;

                if (k_loc > 1) {
                    double dphi_e = StencilFTCS(phi.data(), li, k_loc, stride, dx2i, dy2i) * dt;
                    dphimax_loc = std::max(dphimax_loc, dphi_e);
                    phin[IDX(li, k_loc, stride)] = phi[IDX(li, k_loc, stride)] + dphi_e;
                }
            }
        }
        t_comp += (MPI_Wtime() - tp0);


        tc0 = MPI_Wtime();
        MPI_Allreduce(&dphimax_loc, &dphimax_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        t_comm += (MPI_Wtime() - tc0);

        
        tp0 = MPI_Wtime();
        std::swap(phi, phin);
        t_comp += (MPI_Wtime() - tp0);

        if (dphimax_global < eps) break;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_total = MPI_Wtime() - t_start;

    
    double max_t_total, max_t_comp, max_t_comm;
    MPI_Reduce(&t_total, &max_t_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comp,  &max_t_comp,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm,  &max_t_comm,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int it_ejecutadas = (it > itmax) ? itmax : it;
        double total_flops = 11.0 * (double)(imax - 1) * (double)(kmax - 1) * (double)it_ejecutadas;
        double gflops_s = (total_flops / max_t_total) / 1.0e9;

        printf("p=%d, n=%d, it=%d, T_total=%.6f, T_comp=%.6f, T_comm=%.6f, GFLOPs/s=%.6f, dphimax=%.6e\n",
               p, imax, it_ejecutadas, max_t_total, max_t_comp, max_t_comm, gflops_s, dphimax_global);
    }

    MPI_Type_free(&col_type);
    MPI_Finalize();
    return 0;
}
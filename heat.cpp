#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define IDX(i, k, stride) ((i) * (stride) + (k))

int main(int argc, char** argv) {
    int imax = 80;
    int kmax = 80;
    int itmax = 20000;
    if (argc >= 2) { imax = std::atoi(argv[1]); kmax = imax; }
    if (argc >= 3) { itmax = std::atoi(argv[2]); }

    const double eps = 1.0e-08;
    const double dx = 1.0 / kmax;
    const double dy = 1.0 / imax;
    const double dx2i = 1.0 / (dx * dx);
    const double dy2i = 1.0 / (dy * dy);
    const double dt = std::min(dx * dx, dy * dy) / 4.0;

    int stride = kmax + 1;
    std::vector<double> phi((imax + 1) * stride, 0.0);
    std::vector<double> phin((imax + 1) * stride, 0.0);

    for (int i = 0; i <= imax; ++i) phi[IDX(i, kmax, stride)] = 1.0;
    for (int k = 1; k < kmax; ++k) {
        phi[IDX(0, k, stride)] = phi[IDX(0, k - 1, stride)] + dx;
        phi[IDX(imax, k, stride)] = phi[IDX(imax, k - 1, stride)] + dx;
    }

    clock_t t0 = clock();
    int it;
    double dphimax;

    for (it = 1; it <= itmax; ++it) {
        dphimax = 0.0;
        for (int k = 1; k < kmax; ++k) {
            for (int i = 1; i < imax; ++i) {
                double dphi = ((phi[IDX(i+1, k, stride)] + phi[IDX(i-1, k, stride)] - 2.0 * phi[IDX(i, k, stride)]) * dy2i +
                               (phi[IDX(i, k+1, stride)] + phi[IDX(i, k-1, stride)] - 2.0 * phi[IDX(i, k, stride)]) * dx2i) * dt;
                dphimax = std::max(dphimax, dphi);
                phin[IDX(i, k, stride)] = phi[IDX(i, k, stride)] + dphi;
            }
        }
        for (int k = 1; k < kmax; ++k) {
            for (int i = 1; i < imax; ++i) {
                phi[IDX(i, k, stride)] = phin[IDX(i, k, stride)];
            }
        }
        if (dphimax < eps) break;
    }

    double t_total = (double)(clock() - t0) / CLOCKS_PER_SEC;
    int it_ejecutadas = (it > itmax) ? itmax : it;
    double total_flops = 11.0 * (double)(imax - 1) * (double)(kmax - 1) * (double)it_ejecutadas;
    double gflops_s = (total_flops / t_total) / 1.0e9;

    printf("SECUENCIAL, p=1, n=%d, it=%d, T_total=%.6f, GFLOPs/s=%.6f\n",
           imax, it_ejecutadas, t_total, gflops_s);
    return 0;
}
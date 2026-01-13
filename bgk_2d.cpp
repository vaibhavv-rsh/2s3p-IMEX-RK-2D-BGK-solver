#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include <array>
#include <omp.h>
#include <algorithm>
inline int idx2(int i, int j, int Ny)
{
    return i * Ny + j;
}
inline int idx4(int i, int j, int k, int l,
                int Ny, int Nv)
{
    return ((i * Ny + j) * Nv + k) * Nv + l;
}
inline void writeVTK2D(
    const std::vector<double> &rho,
    const std::vector<double> &ux,
    const std::vector<double> &uy,
    const std::vector<double> &T,
    int Nx,
    int Ny, double dx, double dy, int step)
{
    std::vector<double> x(Nx), y(Ny);
    for (int i = 0; i < Nx; i++)
    {
        x[i] = i * dx;
    }
    for (int j = 0; j < Ny; j++)
    {
        y[j] = j * dy;
    }
    std::string output_dir = "./solutions_2d/";
    std::string mkdir_command = "mkdir -p " + output_dir;
    std::system(mkdir_command.c_str());
    std::string filename = "./solutions_2d/solution_" + std::to_string(step) + ".vtk";
    std::ofstream out(filename);
    out << "# vtk DataFile Version 3.0\n";
    out << "2D BGK simulation\n";
    out << "ASCII\n";
    out << "DATASET STRUCTURED_GRID\n";
    out << "DIMENSIONS " << Nx << " " << Ny << " 1\n";
    out << "POINTS " << Nx * Ny << " float\n";
    for (int j = 0; j < Ny; j++)
    {
        for (int i = 0; i < Nx; i++)
        {
            out << x[i] << " " << y[j] << " 0\n";
        }
    }
    // Density
    out << "CELL_DATA " << (Nx - 1) * (Ny - 1) << "\n";
    out << "SCALARS rho float 1\nLOOKUP_TABLE default\n";
    for (int j = 0; j < Ny - 1; j++)
    {
        for (int i = 0; i < Nx - 1; i++)
        {
            out << rho[idx2(i, j, Ny)] << "\n";
        }
    }

    // Velocity
    out << "VECTORS velocity float\n";
    for (int j = 0; j < Ny - 1; j++)
    {
        for (int i = 0; i < Nx - 1; i++)
        {
            out << ux[idx2(i, j, Ny)] << " " << uy[idx2(i, j, Ny)] << " 0\n";
        }
    }

    out << "SCALARS temperature float 1\nLOOKUP_TABLE default\n";
    for (int j = 0; j < Ny - 1; j++)
    {
        for (int i = 0; i < Nx - 1; i++)
            out << T[idx2(i, j, Ny)] << "\n";
    }

    out.close();
}
struct Parameters
{
    int Nx = 100;
    int Ny = 50;
    int Nv = 50;
    double Lx = 1.0;
    double Ly = 1.0;
    double T_end = 0.2;
    double gamma_gas = 1.4;
    double tau = 0.001;
    double cfl = 0;
};
double maxwellian(double rho, double ux, double uy, double T,
                  double vx, double vy)
{
    double c2 = (vx - ux) * (vx - ux) + (vy - uy) * (vy - uy);
    return rho / (2.0 * M_PI * T) * exp(-c2 / (2.0 * T));
}
double computeResidual(
    const std::vector<double> &f,
    const std::vector<double> &f_new,
    Parameters params)
{
    double res = 0.0, norm = 0.0;
#pragma omp parallel for reduction(+ : res, norm) collapse(4)
    for (int i = 0; i < params.Nx; ++i)
        for (int j = 0; j < params.Ny; ++j)
            for (int k = 0; k < params.Nv; ++k)
                for (int l = 0; l < params.Nv; ++l)
                {
                    int id = idx4(i, j, k, l, params.Ny, params.Nv);
                    double diff = f_new[id] - f[id];
                    res += diff * diff;
                    norm += f[id] * f[id];
                }
    return sqrt(res / norm);
}
void initialize(std::vector<double> &f,
                std::vector<double> &rho,
                std::vector<double> &ux,
                std::vector<double> &uy,
                std::vector<double> &vx,
                std::vector<double> &vy,
                const Parameters params)
{
    // -------- Initial condition (2D Sod) --------
    for (int i = 0; i < params.Nx; ++i)
    {
        // double x = (i + 0.5) * params.Lx / params.Nx;
        double x = i * params.Lx / params.Nx;
        for (int j = 0; j < params.Ny; ++j)
        {
            double rho0 = 0.0, u0 = 0.0, v0 = 0.0, p0 = 0.0;

            if (x <= 0.5)
            {
                rho0 = 1.0;
                p0 = 1.0;
            }
            else
            {
                rho0 = 0.125;
                p0 = 0.1;
            }

            double T0 = p0 / rho0;

            for (int k = 0; k < params.Nv; ++k)
            {
                for (int l = 0; l < params.Nv; ++l)
                {
                    int id = idx4(i, j, k, l, params.Ny, params.Nv);
                    f[id] =
                        maxwellian(rho0, u0, v0, T0, vx[k], vy[l]);
                }
            }
        }
    }
}
void computeMacroscopic(
    const std::vector<double> &f,
    std::vector<double> &rho,
    std::vector<double> &ux,
    std::vector<double> &uy,
    const std::vector<double> &vx,
    const std::vector<double> &vy,
    std::vector<double> &T,
    Parameters params,
    double dv)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < params.Nx; ++i)
        for (int j = 0; j < params.Ny; ++j)
        {
            double r = 0, mx = 0, my = 0, e = 0;

            for (int k = 0; k < params.Nv; ++k)
                for (int l = 0; l < params.Nv; ++l)
                {
                    double val = f[idx4(i, j, k, l, params.Ny, params.Nv)];
                    r += val;
                    mx += vx[k] * val;
                    my += vy[l] * val;
                    e += 0.5 * (vx[k] * vx[k] + vy[l] * vy[l]) * val;
                }

            r *= dv * dv;
            mx *= dv * dv;
            my *= dv * dv;
            e *= dv * dv;
            int id = idx2(i, j, params.Ny);
            rho[id] = r;
            ux[id] = mx / r;
            uy[id] = my / r;
            T[id] = ((e / r) - 0.5 * (ux[id] * ux[id] + uy[id] * uy[id]));
            // T[i][j] = max(T[i][j], 1e-12);
            if (T[id] < 1e-12)
                std::cout << "Warning: Negative Temperature at (" << i << "," << j << ") : " << T[id] << std::endl;
        }
}
void computeMoments(
    const std::vector<double> &f,
    std::vector<double> &r_,
    std::vector<double> &mx_,
    std::vector<double> &my_,
    std::vector<double> &e_,
    const std::vector<double> &vx,
    const std::vector<double> &vy,
    Parameters params,
    double dv)
{
    for (int i = 0; i < params.Nx; ++i)
    {
        for (int j = 0; j < params.Ny; ++j)
        {
            double r = 0, mx = 0, my = 0, e = 0;

            for (int k = 0; k < params.Nv; ++k)
            {
                for (int l = 0; l < params.Nv; ++l)
                {

                    double val = f[idx4(i, j, k, l, params.Ny, params.Nv)];
                    r += val;
                    mx += vx[k] * val;
                    my += vy[l] * val;
                    e += 0.5 * (vx[k] * vx[k] + vy[l] * vy[l]) * val;
                }
            }

            r *= dv * dv;
            mx *= dv * dv;
            my *= dv * dv;
            e *= dv * dv;

            // r = std::max(r, 1e-12);
            // e = std::max(e, 0.5 * (mx * mx + my * my) / r);
            int id = idx2(i, j, params.Ny);
            r_[id] = r;
            mx_[id] = mx;
            my_[id] = my;
            e_[id] = e;
        }
    }
}
inline double WENO5_Reconstruct(
    double f0, double f1, double f2, double f3, double f4)
{
    const double eps = 1e-12;

    double q1 = (2.0 * f0 - 7.0 * f1 + 11.0 * f2) / 6.0;
    double q2 = (-f1 + 5.0 * f2 + 2.0 * f3) / 6.0;
    double q3 = (2.0 * f2 + 5.0 * f3 - f4) / 6.0;

    double b1 = (13.0 / 12.0) * (f0 - 2.0 * f1 + f2) * (f0 - 2.0 * f1 + f2) + (1.0 / 4.0) * (f0 - 4.0 * f1 + 3.0 * f2) * (f0 - 4.0 * f1 + 3.0 * f2);
    double b2 = (13.0 / 12.0) * (f1 - 2.0 * f2 + f3) * (f1 - 2.0 * f2 + f3) + (1.0 / 4.0) * (f1 - f3) * (f1 - f3);
    double b3 = (13.0 / 12.0) * (f2 - 2.0 * f3 + f4) * (f2 - 2.0 * f3 + f4) + (1.0 / 4.0) * (3.0 * f2 - 4.0 * f3 + f4) * (3.0 * f2 - 4.0 * f3 + f4);

    double a1 = 0.1 / pow(eps + b1, 2);
    double a2 = 0.6 / pow(eps + b2, 2);
    double a3 = 0.3 / pow(eps + b3, 2);

    double wsum = a1 + a2 + a3;
    return (a1 * q1 + a2 * q2 + a3 * q3) / wsum;
}

void setWENOboundaryConditions(
    std::vector<double> &f,
    const std::vector<double> &vx, const std::vector<double> &vy,
    const Parameters &params)
{
    for (int j = 0; j < params.Ny; ++j)
        for (int k = 0; k < params.Nv; ++k)
            for (int l = 0; l < params.Nv; ++l)
            {
                f[idx4(0, j, k, l, params.Ny, params.Nv)] =
                    f[idx4(3, j, k, l, params.Ny, params.Nv)];
                f[idx4(1, j, k, l, params.Ny, params.Nv)] =
                    f[idx4(3, j, k, l, params.Ny, params.Nv)];
                f[idx4(2, j, k, l, params.Ny, params.Nv)] =
                    f[idx4(3, j, k, l, params.Ny, params.Nv)];

                f[idx4(params.Nx - 1, j, k, l, params.Ny, params.Nv)] =
                    f[idx4(params.Nx - 4, j, k, l, params.Ny, params.Nv)];
                f[idx4(params.Nx - 2, j, k, l, params.Ny, params.Nv)] =
                    f[idx4(params.Nx - 4, j, k, l, params.Ny, params.Nv)];
                f[idx4(params.Nx - 3, j, k, l, params.Ny, params.Nv)] =
                    f[idx4(params.Nx - 4, j, k, l, params.Ny, params.Nv)];
            }

    for (int i = 0; i < params.Nx; ++i)
        for (int k = 0; k < params.Nv; ++k)
            for (int l = 0; l < params.Nv; ++l)
            {
                f[idx4(i, 0, k, l, params.Ny, params.Nv)] =
                    f[idx4(i, 3, k, l, params.Ny, params.Nv)];
                f[idx4(i, 1, k, l, params.Ny, params.Nv)] =
                    f[idx4(i, 3, k, l, params.Ny, params.Nv)];
                f[idx4(i, 2, k, l, params.Ny, params.Nv)] =
                    f[idx4(i, 3, k, l, params.Ny, params.Nv)];

                f[idx4(i, params.Ny - 1, k, l, params.Ny, params.Nv)] =
                    f[idx4(i, params.Ny - 4, k, l, params.Ny, params.Nv)];
                f[idx4(i, params.Ny - 2, k, l, params.Ny, params.Nv)] =
                    f[idx4(i, params.Ny - 4, k, l, params.Ny, params.Nv)];
                f[idx4(i, params.Ny - 3, k, l, params.Ny, params.Nv)] =
                    f[idx4(i, params.Ny - 4, k, l, params.Ny, params.Nv)];
            }
}
// void setWENOboundaryConditions(
//     std::vector<double> &f,
//     const std::vector<double> &vx, const std::vector<double> &vy,
//     const Parameters &params)
// {
//     for (int j = 0; j < params.Ny; ++j)
//         for (int k = 0; k < params.Nv; ++k)
//             for (int l = 0; l < params.Nv; ++l)
//             {
//                 // LEFT boundary (i = 0)
//                 if (vx[k] < 0) // pointing into wall
//                     f[idx4(0, j, k, l, params.Ny, params.Nv)] = f[idx4(1, j, params.Nv - 1 - k, l, params.Ny, params.Nv)];
//                 else
//                     f[idx4(0, j, k, l, params.Ny, params.Nv)] = f[idx4(1, j, k, l, params.Ny, params.Nv)];

//                 // RIGHT boundary (i = Nx-1)
//                 if (vx[k] > 0) // pointing into wall
//                     f[idx4(params.Nx - 1, j, k, l, params.Ny, params.Nv)] = f[idx4(params.Nx - 2, j, params.Nv - 1 - k, l, params.Ny, params.Nv)];
//                 else
//                     f[idx4(params.Nx - 1, j, k, l, params.Ny, params.Nv)] = f[idx4(params.Nx - 2, j, k, l, params.Ny, params.Nv)];

//                 f[idx4(1, j, k, l, params.Ny, params.Nv)] =
//                     f[idx4(3, j, k, l, params.Ny, params.Nv)];
//                 f[idx4(2, j, k, l, params.Ny, params.Nv)] =
//                     f[idx4(3, j, k, l, params.Ny, params.Nv)];

//                 f[idx4(params.Nx - 2, j, k, l, params.Ny, params.Nv)] =
//                     f[idx4(params.Nx - 4, j, k, l, params.Ny, params.Nv)];
//                 f[idx4(params.Nx - 3, j, k, l, params.Ny, params.Nv)] =
//                     f[idx4(params.Nx - 4, j, k, l, params.Ny, params.Nv)];
//             }
//     for (int i = 0; i < params.Nx; ++i)
//         for (int k = 0; k < params.Nv; ++k)
//             for (int l = 0; l < params.Nv; ++l)
//             {
//                 // BOTTOM boundary (j = 0)
//                 if (vy[l] < 0) // pointing into wall
//                     f[idx4(i, 0, k, l, params.Ny, params.Nv)] = f[idx4(i, 1, k, params.Nv - 1 - l, params.Ny, params.Nv)];
//                 else
//                     f[idx4(i, 0, k, l, params.Ny, params.Nv)] = f[idx4(i, 1, k, l, params.Ny, params.Nv)];

//                 // TOP boundary (j = Ny-1)
//                 if (vy[l] > 0) // pointing into wall
//                     f[idx4(i, params.Ny - 1, k, l, params.Ny, params.Nv)] = f[idx4(i, params.Ny - 2, k, params.Nv - 1 - l, params.Ny, params.Nv)];
//                 else
//                     f[idx4(i, params.Ny - 1, k, l, params.Ny, params.Nv)] = f[idx4(i, params.Ny - 2, k, l, params.Ny, params.Nv)];

//                 f[idx4(i, 1, k, l, params.Ny, params.Nv)] =
//                     f[idx4(i, 3, k, l, params.Ny, params.Nv)];
//                 f[idx4(i, 2, k, l, params.Ny, params.Nv)] =
//                     f[idx4(i, 3, k, l, params.Ny, params.Nv)];

//                 f[idx4(i, params.Ny - 2, k, l, params.Ny, params.Nv)] =
//                     f[idx4(i, params.Ny - 4, k, l, params.Ny, params.Nv)];
//                 f[idx4(i, params.Ny - 3, k, l, params.Ny, params.Nv)] =
//                     f[idx4(i, params.Ny - 4, k, l, params.Ny, params.Nv)];
//             }
// }
double maxVelocity(
    const std::vector<double> &vx,
    const std::vector<double> &vy,
    Parameters params)
{
    double vmax = 0.0;
    for (int k = 0; k < params.Nv; ++k)
    {
        vmax = std::max(vmax, std::abs(vx[k]));
        vmax = std::max(vmax, std::abs(vy[k]));
    }
    return vmax;
}
void computeFexplicit(
    const std::vector<double> &f,
    std::vector<double> &Lf,
    const std::vector<double> &vx,
    const std::vector<double> &vy,
    double dt, double dx, double dy,
    Parameters params)
{
    // Lf = f;
    const double eps = 1e-12;
#pragma omp parallel for collapse(2)
    for (int i = 3; i < params.Nx - 3; ++i)
        for (int j = 3; j < params.Ny - 3; ++j)
            for (int k = 0; k < params.Nv; ++k)
                for (int l = 0; l < params.Nv; ++l)
                {
                    double fbar = f[idx4(i, j, k, l, params.Ny, params.Nv)];
                    double fRx = 0.0;
                    double fLx = 0.0;
                    /* ---------- X direction ---------- */
                    if (vx[k] > 0)
                    {
                        // f_{i+1/2}^-
                        fRx = WENO5_Reconstruct(
                            f[idx4(i - 2, j, k, l, params.Ny, params.Nv)],
                            f[idx4(i - 1, j, k, l, params.Ny, params.Nv)],
                            f[idx4(i, j, k, l, params.Ny, params.Nv)],
                            f[idx4(i + 1, j, k, l, params.Ny, params.Nv)],
                            f[idx4(i + 2, j, k, l, params.Ny, params.Nv)]);
                        // f_{i-1/2}^-
                        fLx = WENO5_Reconstruct(
                            f[idx4(i - 3, j, k, l, params.Ny, params.Nv)],
                            f[idx4(i - 2, j, k, l, params.Ny, params.Nv)],
                            f[idx4(i - 1, j, k, l, params.Ny, params.Nv)],
                            f[idx4(i, j, k, l, params.Ny, params.Nv)],
                            f[idx4(i + 1, j, k, l, params.Ny, params.Nv)]);
                    }
                    else
                    {
                        // f_{i+1/2}^+
                        fRx = WENO5_Reconstruct(
                            f[idx4(i + 3, j, k, l, params.Ny, params.Nv)],
                            f[idx4(i + 2, j, k, l, params.Ny, params.Nv)],
                            f[idx4(i + 1, j, k, l, params.Ny, params.Nv)],
                            f[idx4(i, j, k, l, params.Ny, params.Nv)],
                            f[idx4(i - 1, j, k, l, params.Ny, params.Nv)]);
                        // f_{i-1/2}^+
                        fLx = WENO5_Reconstruct(
                            f[idx4(i + 2, j, k, l, params.Ny, params.Nv)],
                            f[idx4(i + 1, j, k, l, params.Ny, params.Nv)],
                            f[idx4(i, j, k, l, params.Ny, params.Nv)],
                            f[idx4(i - 1, j, k, l, params.Ny, params.Nv)],
                            f[idx4(i - 2, j, k, l, params.Ny, params.Nv)]);
                    }
                    // positivityLimiter(fbar, fLx, fRx);
                    double Fx_plus = vx[k] * fRx;
                    double Fx_minus = vx[k] * fLx;
                    /*========================
                      Y-direction fluxes
                    ========================*/
                    double fLy, fRy;
                    if (vy[l] > 0)
                    {
                        // f_{j+1/2}^-
                        fRy = WENO5_Reconstruct(
                            f[idx4(i, j - 2, k, l, params.Ny, params.Nv)],
                            f[idx4(i, j - 1, k, l, params.Ny, params.Nv)],
                            f[idx4(i, j, k, l, params.Ny, params.Nv)],
                            f[idx4(i, j + 1, k, l, params.Ny, params.Nv)],
                            f[idx4(i, j + 2, k, l, params.Ny, params.Nv)]);
                        // f_{j-1/2}^-
                        fLy = WENO5_Reconstruct(
                            f[idx4(i, j - 3, k, l, params.Ny, params.Nv)],
                            f[idx4(i, j - 2, k, l, params.Ny, params.Nv)],
                            f[idx4(i, j - 1, k, l, params.Ny, params.Nv)],
                            f[idx4(i, j, k, l, params.Ny, params.Nv)],
                            f[idx4(i, j + 1, k, l, params.Ny, params.Nv)]);
                    }
                    else
                    {
                        // f_{i+1/2}^+
                        fRy = WENO5_Reconstruct(
                            f[idx4(i, j + 3, k, l, params.Ny, params.Nv)],
                            f[idx4(i, j + 2, k, l, params.Ny, params.Nv)],
                            f[idx4(i, j + 1, k, l, params.Ny, params.Nv)],
                            f[idx4(i, j, k, l, params.Ny, params.Nv)],
                            f[idx4(i, j - 1, k, l, params.Ny, params.Nv)]);
                        // f_{i-1/2}^+
                        fLy = WENO5_Reconstruct(
                            f[idx4(i, j + 2, k, l, params.Ny, params.Nv)],
                            f[idx4(i, j + 1, k, l, params.Ny, params.Nv)],
                            f[idx4(i, j, k, l, params.Ny, params.Nv)],
                            f[idx4(i, j - 1, k, l, params.Ny, params.Nv)],
                            f[idx4(i, j - 2, k, l, params.Ny, params.Nv)]);
                    }
                    // positivityLimiter(fbar, fLy, fRy);

                    double Fy_plus = vy[l] * fRy;
                    double Fy_minus = vy[l] * fLy;

                    Lf[idx4(i, j, k, l, params.Ny, params.Nv)] = -((Fx_plus - Fx_minus) / dx +
                                                                   (Fy_plus - Fy_minus) / dy);
                }
}
double compute_max_entropy(const std::vector<double> &f, Parameters params)
{
    double entropy = 0.0;
    for (int i = 3; i < params.Nx - 3; ++i)
        for (int j = 3; j < params.Ny - 3; ++j)
            for (int k = 0; k < params.Nv; ++k)
                for (int l = 0; l < params.Nv; ++l)
                {
                    {
                        {
                            double fbar = f[idx4(i, j, k, l, params.Ny, params.Nv)];
                            if (fbar > 0)
                                entropy += fbar * log(fbar);
                            // entropy += fbar * log(fbar) - fbar;
                        }
                    }
                }
    return entropy;
}
double totalVariation2D(
    const std::vector<double> &q,
    const Parameters &params)
{
    double tv = 0.0;

    for (int i = 0; i < params.Nx - 1; ++i)
        for (int j = 0; j < params.Ny - 1; ++j)
        {
            tv += std::abs(q[idx2(i + 1, j, params.Ny)] - q[idx2(i, j, params.Ny)]);
            tv += std::abs(q[idx2(i, j + 1, params.Ny)] - q[idx2(i, j, params.Ny)]);
        }

    return tv;
}
void print_vector(std::string filename, const std::vector<double> &data)
{
    std::ofstream file(filename);
    for (double v : data)
    {
        file << v << "\n";
    }
    file.close();
}
int main()
{
#pragma omp parallel
    {
#pragma omp master
        std::cout << "Running with " << omp_get_num_threads() << " threads.\n";
    }
    Parameters params;
    params.Nx = 101;
    params.Ny = 51;
    params.Nv = 15;
    params.Lx = 1.0;
    params.Ly = 1.0;
    params.T_end = 0.2;
    params.gamma_gas = 2.0;
    params.tau = 1e-5; // nearly collisionless
    params.cfl = 0.3;

    double dx = params.Lx / (params.Nx - 1);
    double dy = params.Ly / (params.Ny - 1);
    double dv = 12.0 / (params.Nv - 1);
    // velocity grid
    std::vector<double> vx(params.Nv), vy(params.Nv);
    for (int k = 0; k < params.Nv; ++k)
    {
        vx[k] = -6.0 + k * dv;
        vy[k] = -6.0 + k * dv;
    }

    std::vector<double> f(params.Nx * params.Ny * params.Nv * params.Nv);
    std::vector<double> rho(params.Nx * params.Ny);
    std::vector<double> ux(params.Nx * params.Ny);
    std::vector<double> uy(params.Nx * params.Ny);
    std::vector<double> T(params.Nx * params.Ny);

    double t = 0.0;
    std::vector<double> entropy;
    std::vector<double> TV_rho;

    initialize(f, rho, ux, uy, vx, vy, params);
    auto Fex = f;
    auto Fim = f;
    auto f0 = f;
    auto f1 = f;
    auto f2 = f;
    auto f3 = f;
    auto F1 = f;
    auto F2 = f;

    while (t < params.T_end)
    {
        double vmax = maxVelocity(vx, vy, params);
        double dt = params.cfl * std::min(dx, dy) / (vmax);
        //    double dt = params.cfl * min(dx, dy) / (20.0 * vmax);
        if (t + dt > params.T_end)
            dt = params.T_end - t;
#pragma omp parallel
        {

            // ---------- Stage 1 ----------
            computeMacroscopic(f, rho, ux, uy, vx, vy, T, params, dv);
#pragma omp for collapse(2)
            for (int i = 0; i < params.Nx; i++)
            {
                for (int j = 0; j < params.Ny; j++)
                {
                    for (int k = 0; k < params.Nv; k++)
                    {
#pragma omp simd
                        for (int l = 0; l < params.Nv; l++)
                        {
                            int id2 = idx2(i, j, params.Ny);
                            int id4 = idx4(i, j, k, l, params.Ny, params.Nv);
                            double alpha = 0.5 * dt / params.tau;
                            double M = maxwellian(
                                rho[id2], ux[id2], uy[id2], T[id2],
                                vx[k], vy[l]);
                            f1[id4] =
                                (f[id4] + alpha * M) / (1.0 + alpha);
                        }
                    }
                }
            }
#pragma omp single
            setWENOboundaryConditions(f1, vx, vy, params);
#pragma omp barrier
            // ---------- Stage 2 ----------
            computeFexplicit(f1, Fex, vx, vy, dt, dx, dy, params);
            std::vector<double> M1(f1.size());
            std::transform(
                f1.begin(), f1.end(),
                Fex.begin(),
                M1.begin(),
                [dt](double a, double b)
                {
                    return a + dt * b;
                });
            computeMacroscopic(M1, rho, ux, uy, vx, vy, T, params, dv);
#pragma omp for collapse(2)
            for (int i = 0; i < params.Nx; i++)
            {
                for (int j = 0; j < params.Ny; j++)
                {
                    for (int k = 0; k < params.Nv; k++)
                    {
#pragma omp simd
                        for (int l = 0; l < params.Nv; l++)
                        {
                            int id2 = idx2(i, j, params.Ny);
                            int id4 = idx4(i, j, k, l, params.Ny, params.Nv);
                            double alpha = 0.5 * dt * dt / (params.tau);
                            double M = maxwellian(
                                rho[id2], ux[id2], uy[id2], T[id2],
                                vx[k], vy[l]);
                            f2[id4] =
                                (f1[id4] + dt * Fex[id4] + alpha * M) / (1.0 + alpha);
                        }
                    }
                }
            }
#pragma omp single
            setWENOboundaryConditions(f2, vx, vy, params);
#pragma omp barrier

            //  ---------- Stage 3 ----------
            computeFexplicit(f2, Fex, vx, vy, dt, dx, dy, params);
            std::vector<double> M2(f2.size());
            std::vector<double> fT(f2.size());
            std::transform(
                f2.begin(), f2.end(),
                f1.begin(),
                fT.begin(),
                [](double a, double b)
                {
                    return a + b;
                });
            std::transform(
                fT.begin(), fT.end(),
                Fex.begin(),
                M2.begin(),
                [dt](double a, double b)
                {
                    return 0.5 * (a + dt * b);
                });
            computeMacroscopic(M2, rho, ux, uy, vx, vy, T, params, dv);
#pragma omp for collapse(2)
            for (int i = 0; i < params.Nx; i++)
            {
                for (int j = 0; j < params.Ny; j++)
                {
                    for (int k = 0; k < params.Nv; k++)
                    {
#pragma omp simd
                        for (int l = 0; l < params.Nv; l++)
                        {
                            int id2 = idx2(i, j, params.Ny);
                            int id4 = idx4(i, j, k, l, params.Ny, params.Nv);
                            double alpha = 0.5 * dt / params.tau;
                            double M = maxwellian(
                                rho[id2], ux[id2], uy[id2], T[id2],
                                vx[k], vy[l]);
                            f3[id4] =
                                (0.5 * (f1[id4] + f2[id4] + dt * Fex[id4]) + alpha * M) / (1.0 + alpha);
                        }
                    }
                }
            }
#pragma omp single
            setWENOboundaryConditions(f3, vx, vy, params);
#pragma omp barrier
        }
        std::cout << "Residual at time " << t << ": " << std::scientific << std::setw(10) << std::setprecision(3) << computeResidual(f, f3, params) << "\n";
        // f.swap(f3);
        f = f3;
        computeMacroscopic(f, rho, ux, uy, vx, vy, T, params, dv);
        TV_rho.push_back(totalVariation2D(rho, params));
        t += dt;
        entropy.push_back(compute_max_entropy(f, params) * dx * dy * dv * dv);
        //  cout << "t = " << t << endl;
        writeVTK2D(rho, ux, uy, T, params.Nx, params.Ny, dx, dy, int(t / dt));
    }
    print_vector("./entropy_sol.txt", entropy);
    print_vector("./TV_rho_sol.txt", TV_rho);
}

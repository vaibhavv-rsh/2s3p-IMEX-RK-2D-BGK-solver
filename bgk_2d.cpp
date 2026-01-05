#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include <array>
#include <omp.h>

using namespace std;

void writeVTK2D(
    const vector<vector<double>> &rho,
    const vector<vector<double>> &ux,
    const vector<vector<double>> &uy,
    const vector<vector<double>> &T,
    int Nx,
    int Ny, double dx, double dy, int step)
{
    vector<double> x(Nx), y(Ny);
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
    ofstream out(filename);
    out << "# vtk DataFile Version 3.0\n";
    out << "2D BGK simulation\n";
    out << "ASCII\n";
    out << "DATASET STRUCTURED_GRID\n";
    out << "DIMENSIONS " << Nx << " " << Ny << " 1\n";
    // out << "ORIGIN 0 0 0\n";
    // out << "SPACING 1 1 1\n";
    // out << "POINT_DATA " << Nx * Ny << "\n";
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
            out << rho[i][j] << "\n";
        }
    }

    // Velocity
    out << "VECTORS velocity float\n";
    for (int j = 0; j < Ny - 1; j++)
    {
        for (int i = 0; i < Nx - 1; i++)
        {
            out << ux[i][j] << " " << uy[i][j] << " 0\n";
        }
    }

    out << "SCALARS temperature float 1\nLOOKUP_TABLE default\n";
    for (int j = 0; j < Ny - 1; j++)
    {
        for (int i = 0; i < Nx - 1; i++)
            out << T[i][j] << "\n";
    }

    out.close();
}

// ---------------- Parameters ----------------
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
// --------------------------------------------
// --------------------------------------------
double maxwellian(double rho, double ux, double uy, double T,
                  double vx, double vy)
{
    double c2 = (vx - ux) * (vx - ux) + (vy - uy) * (vy - uy);
    return rho / (2.0 * M_PI * T) * exp(-c2 / (2.0 * T));
}
double computeResidual(
    const vector<vector<vector<vector<double>>>> &f,
    const vector<vector<vector<vector<double>>>> &f_new,
    Parameters params)
{
    double res = 0.0, norm = 0.0;
    for (int i = 0; i < params.Nx; ++i)
        for (int j = 0; j < params.Ny; ++j)
            for (int k = 0; k < params.Nv; ++k)
                for (int l = 0; l < params.Nv; ++l)
                {
                    double diff = f_new[i][j][k][l] - f[i][j][k][l];
                    res += diff * diff;
                    norm += f[i][j][k][l] * f[i][j][k][l];
                }
    return sqrt(res / norm);
}
void initialize(vector<vector<vector<vector<double>>>> &f,
                vector<vector<double>> &rho,
                vector<vector<double>> &ux,
                vector<vector<double>> &uy,
                vector<double> &vx,
                vector<double> &vy,
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

            if (x < 0.5)
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
                for (int l = 0; l < params.Nv; ++l)
                    f[i][j][k][l] =
                        maxwellian(rho0, u0, v0, T0, vx[k], vy[l]);
        }
    }
}
void computeMacroscopic(
    const vector<vector<vector<vector<double>>>> &f,
    vector<vector<double>> &rho,
    vector<vector<double>> &ux,
    vector<vector<double>> &uy,
    const vector<double> &vx,
    const vector<double> &vy,
    vector<vector<double>> &T,
    Parameters params,
    double dv)
{

    for (int i = 0; i < params.Nx; ++i)
        for (int j = 0; j < params.Ny; ++j)
        {
            double r = 0, mx = 0, my = 0, e = 0;

            for (int k = 0; k < params.Nv; ++k)
                for (int l = 0; l < params.Nv; ++l)
                {
                    double val = f[i][j][k][l];
                    r += val;
                    mx += vx[k] * val;
                    my += vy[l] * val;
                    e += 0.5 * (vx[k] * vx[k] + vy[l] * vy[l]) * val;
                }

            r *= dv * dv;
            mx *= dv * dv;
            my *= dv * dv;
            e *= dv * dv;

            rho[i][j] = r;
            ux[i][j] = mx / r;
            uy[i][j] = my / r;
            // double kinetic = 0.5 * r * (ux[i][j] * ux[i][j] + uy[i][j] * uy[i][j]);
            // double eint = e - kinetic;
            // double p = (params.gamma_gas - 1.0) * eint;
            // T[i][j] = p / r;
            T[i][j] = ((e / r) - 0.5 * (ux[i][j] * ux[i][j] + uy[i][j] * uy[i][j]));
            // T[i][j] = max(T[i][j], 1e-12);
            if (T[i][j] < 1e-12)
                std::cout << "Warning: Negative Temperature at (" << i << "," << j << ") : " << T[i][j] << std::endl;
        }
}

void positivityLimiter(
    double fbar,
    double &fL,
    double &fR)
{
    const double w1 = 1.0 / 12.0;
    const double w4 = 1.0 / 12.0;
    const double w23 = 5.0 / 6.0;

    double fM = (fbar - w1 * fL - w4 * fR) / w23;

    double theta = 1.0;

    if (fL < 0.0)
        theta = std::min(theta, fbar / (fbar - fL));
    if (fR < 0.0)
        theta = std::min(theta, fbar / (fbar - fR));
    if (fM < 0.0)
        theta = std::min(theta, fbar / (fbar - fM));

    fL = theta * (fL - fbar) + fbar;
    fR = theta * (fR - fbar) + fbar;
}

void computeMoments(
    const vector<vector<vector<vector<double>>>> &f,
    vector<vector<double>> &r_,
    vector<vector<double>> &mx_,
    vector<vector<double>> &my_,
    vector<vector<double>> &e_,
    const vector<double> &vx,
    const vector<double> &vy,
    Parameters params,
    double dv)
{
    for (int i = 0; i < params.Nx; ++i)
        for (int j = 0; j < params.Ny; ++j)
        {
            double r = 0, mx = 0, my = 0, e = 0;

            for (int k = 0; k < params.Nv; ++k)
                for (int l = 0; l < params.Nv; ++l)
                {
                    double val = f[i][j][k][l];
                    r += val;
                    mx += vx[k] * val;
                    my += vy[l] * val;
                    e += 0.5 * (vx[k] * vx[k] + vy[l] * vy[l]) * val;
                }

            r *= dv * dv;
            mx *= dv * dv;
            my *= dv * dv;
            e *= dv * dv;

            r = max(r, 1e-12);
            e = max(e, 0.5 * (mx * mx + my * my) / r);

            r_[i][j] = r;
            mx_[i][j] = mx;
            my_[i][j] = my;
            e_[i][j] = e;
        }
}
double WENO5_Reconstruct(
    double f0, double f1, double f2, double f3, double f4)
{
    const double eps = 1e-12;

    double q1 = (2.0 * f0 - 7.0 * f1 + 11.0 * f2) / 6.0;
    double q2 = (-f1 + 5.0 * f2 + 2.0 * f3) / 6.0;
    double q3 = (2.0 * f2 + 5.0 * f3 - f4) / 6.0;

    double b1 = (13.0 / 12.0) * pow(f0 - 2.0 * f1 + f2, 2) + (1.0 / 4.0) * pow(f0 - 4.0 * f1 + 3.0 * f2, 2);
    double b2 = (13.0 / 12.0) * pow(f1 - 2.0 * f2 + f3, 2) + (1.0 / 4.0) * pow(f1 - f3, 2);
    double b3 = (13.0 / 12.0) * pow(f2 - 2.0 * f3 + f4, 2) + (1.0 / 4.0) * pow(3.0 * f2 - 4.0 * f3 + f4, 2);

    double a1 = 0.1 / pow(eps + b1, 2);
    double a2 = 0.6 / pow(eps + b2, 2);
    double a3 = 0.3 / pow(eps + b3, 2);

    double wsum = a1 + a2 + a3;
    return (a1 * q1 + a2 * q2 + a3 * q3) / wsum;
}

void WENO5Step(
    const vector<vector<vector<vector<double>>>> &f,
    vector<vector<vector<vector<double>>>> &f_new,
    const vector<double> &vx,
    const vector<double> &vy,
    double dt,
    double dx,
    double dy,
    Parameters params)
{
    const double eps = 1e-12;
    for (int i = 3; i < params.Nx - 3; ++i)
        for (int j = 3; j < params.Ny - 3; ++j)
            for (int k = 0; k < params.Nv; ++k)
                for (int l = 0; l < params.Nv; ++l)
                {
                    double fbar = f[i][j][k][l];
                    // double dfdx = 0.0;
                    // double dfdy = 0.0;
                    double fRx = 0.0;
                    double fLx = 0.0;
                    /* ---------- X direction ---------- */
                    if (vx[k] > 0)
                    {
                        // f_{i+1/2}^-
                        fRx = WENO5_Reconstruct(
                            f[i - 2][j][k][l],
                            f[i - 1][j][k][l],
                            f[i][j][k][l],
                            f[i + 1][j][k][l],
                            f[i + 2][j][k][l]);
                        // f_{i-1/2}^-
                        fLx = WENO5_Reconstruct(
                            f[i - 3][j][k][l],
                            f[i - 2][j][k][l],
                            f[i - 1][j][k][l],
                            f[i][j][k][l],
                            f[i + 1][j][k][l]);
                    }
                    else
                    {
                        // f_{i+1/2}^+
                        fRx = WENO5_Reconstruct(
                            f[i + 3][j][k][l],
                            f[i + 2][j][k][l],
                            f[i + 1][j][k][l],
                            f[i][j][k][l],
                            f[i - 1][j][k][l]);
                        // f_{i-1/2}^+
                        fLx = WENO5_Reconstruct(
                            f[i + 2][j][k][l],
                            f[i + 1][j][k][l],
                            f[i][j][k][l],
                            f[i - 1][j][k][l],
                            f[i - 2][j][k][l]);
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
                            f[i][j - 2][k][l],
                            f[i][j - 1][k][l],
                            f[i][j][k][l],
                            f[i][j + 1][k][l],
                            f[i][j + 2][k][l]);
                        // f_{j-1/2}^-
                        fLy = WENO5_Reconstruct(
                            f[i][j - 3][k][l],
                            f[i][j - 2][k][l],
                            f[i][j - 1][k][l],
                            f[i][j][k][l],
                            f[i][j + 1][k][l]);
                    }
                    else
                    {
                        // f_{i+1/2}^+
                        fRy = WENO5_Reconstruct(
                            f[i][j + 3][k][l],
                            f[i][j + 2][k][l],
                            f[i][j + 1][k][l],
                            f[i][j][k][l],
                            f[i][j - 1][k][l]);
                        // f_{i-1/2}^+
                        fLy = WENO5_Reconstruct(
                            f[i][j + 2][k][l],
                            f[i][j + 1][k][l],
                            f[i][j][k][l],
                            f[i][j - 1][k][l],
                            f[i][j - 2][k][l]);
                    }
                    // positivityLimiter(fbar, fLy, fRy);

                    double Fy_plus = vy[l] * fRy;
                    double Fy_minus = vy[l] * fLy;

                    // f_new[i][j][k][l] -= dt * (vx[k] * dfdx / dx +
                    //                            vy[l] * dfdy / dy);
                    f_new[i][j][k][l] -= dt * ((Fx_plus - Fx_minus) / dx +
                                               (Fy_plus - Fy_minus) / dy);

                    // cfl condition should not violated:
                    // dt <= min(dx, dy) / (12.0 * max(|vx|, |vy|))
                }
}
void setWENOboundaryConditions(
    vector<vector<vector<vector<double>>>> &f,
    vector<vector<double>> &rho,
    vector<vector<double>> &ux,
    vector<vector<double>> &uy,
    vector<vector<double>> &T,
    const Parameters &params)
{
    for (int j = 0; j < params.Ny; ++j)
    {
        f[0][j] = f[3][j];
        f[1][j] = f[3][j];
        f[2][j] = f[3][j];

        f[params.Nx - 1][j] = f[params.Nx - 4][j];
        f[params.Nx - 2][j] = f[params.Nx - 4][j];
        f[params.Nx - 3][j] = f[params.Nx - 4][j];
        rho[0][j] = rho[3][j];
        rho[1][j] = rho[3][j];
        rho[2][j] = rho[3][j];

        rho[params.Nx - 1][j] = rho[params.Nx - 4][j];
        rho[params.Nx - 2][j] = rho[params.Nx - 4][j];
        rho[params.Nx - 3][j] = rho[params.Nx - 4][j];

        ux[0][j] = ux[3][j];
        ux[1][j] = ux[3][j];
        ux[2][j] = ux[3][j];

        ux[params.Nx - 1][j] = ux[params.Nx - 4][j];
        ux[params.Nx - 2][j] = ux[params.Nx - 4][j];
        ux[params.Nx - 3][j] = ux[params.Nx - 4][j];

        uy[0][j] = uy[3][j];
        uy[1][j] = uy[3][j];
        uy[2][j] = uy[3][j];

        uy[params.Nx - 1][j] = uy[params.Nx - 4][j];
        uy[params.Nx - 2][j] = uy[params.Nx - 4][j];
        uy[params.Nx - 3][j] = uy[params.Nx - 4][j];

        T[0][j] = T[3][j];
        T[1][j] = T[3][j];
        T[2][j] = T[3][j];

        T[params.Nx - 1][j] = T[params.Nx - 4][j];
        T[params.Nx - 2][j] = T[params.Nx - 4][j];
        T[params.Nx - 3][j] = T[params.Nx - 4][j];
    }

    for (int i = 0; i < params.Nx; ++i)
    {
        f[i][0] = f[i][3];
        f[i][1] = f[i][3];
        f[i][2] = f[i][3];

        f[i][params.Ny - 1] = f[i][params.Ny - 4];
        f[i][params.Ny - 2] = f[i][params.Ny - 4];
        f[i][params.Ny - 3] = f[i][params.Ny - 4];

        rho[i][0] = rho[i][3];
        rho[i][1] = rho[i][3];
        rho[i][2] = rho[i][3];

        rho[i][params.Ny - 1] = rho[i][params.Ny - 4];
        rho[i][params.Ny - 2] = rho[i][params.Ny - 4];
        rho[i][params.Ny - 3] = rho[i][params.Ny - 4];

        ux[i][0] = ux[i][3];
        ux[i][1] = ux[i][3];
        ux[i][2] = ux[i][3];

        ux[i][params.Ny - 1] = ux[i][params.Ny - 4];
        ux[i][params.Ny - 2] = ux[i][params.Ny - 4];
        ux[i][params.Ny - 3] = ux[i][params.Ny - 4];

        uy[i][0] = uy[i][3];
        uy[i][1] = uy[i][3];
        uy[i][2] = uy[i][3];

        uy[i][params.Ny - 1] = uy[i][params.Ny - 4];
        uy[i][params.Ny - 2] = uy[i][params.Ny - 4];
        uy[i][params.Ny - 3] = uy[i][params.Ny - 4];

        T[i][0] = T[i][3];
        T[i][1] = T[i][3];
        T[i][2] = T[i][3];

        T[i][params.Ny - 1] = T[i][params.Ny - 4];
        T[i][params.Ny - 2] = T[i][params.Ny - 4];
        T[i][params.Ny - 3] = T[i][params.Ny - 4];
    }
}
// void setWENOboundaryConditions(
//     vector<vector<vector<vector<double>>>> &f,
//     const Parameters &params)
// {
//     for (int j = 0; j < params.Ny; ++j)
//     {
//         f[0][j] = f[3][j];
//         f[1][j] = f[3][j];
//         f[2][j] = f[3][j];

//         f[params.Nx - 1][j] = f[params.Nx - 4][j];
//         f[params.Nx - 2][j] = f[params.Nx - 4][j];
//         f[params.Nx - 3][j] = f[params.Nx - 4][j];
//     }

//     for (int i = 0; i < params.Nx; ++i)
//     {
//         f[i][0] = f[i][3];
//         f[i][1] = f[i][3];
//         f[i][2] = f[i][3];

//         f[i][params.Ny - 1] = f[i][params.Ny - 4];
//         f[i][params.Ny - 2] = f[i][params.Ny - 4];
//         f[i][params.Ny - 3] = f[i][params.Ny - 4];
//     }
// }

double maxVelocity(
    const vector<double> &vx,
    const vector<double> &vy,
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
    const vector<vector<vector<vector<double>>>> &f,
    vector<vector<vector<vector<double>>>> &Lf,
    const vector<double> &vx,
    const vector<double> &vy,
    double dt, double dx, double dy,
    Parameters params)
{
    Lf = f; // start with f
    const double eps = 1e-12;
    for (int i = 3; i < params.Nx - 3; ++i)
        for (int j = 3; j < params.Ny - 3; ++j)
            for (int k = 0; k < params.Nv; ++k)
                for (int l = 0; l < params.Nv; ++l)
                {
                    double fbar = f[i][j][k][l];
                    // double dfdx = 0.0;
                    // double dfdy = 0.0;
                    double fRx = 0.0;
                    double fLx = 0.0;
                    /* ---------- X direction ---------- */
                    if (vx[k] > 0)
                    {
                        // f_{i+1/2}^-
                        fRx = WENO5_Reconstruct(
                            f[i - 2][j][k][l],
                            f[i - 1][j][k][l],
                            f[i][j][k][l],
                            f[i + 1][j][k][l],
                            f[i + 2][j][k][l]);
                        // f_{i-1/2}^-
                        fLx = WENO5_Reconstruct(
                            f[i - 3][j][k][l],
                            f[i - 2][j][k][l],
                            f[i - 1][j][k][l],
                            f[i][j][k][l],
                            f[i + 1][j][k][l]);
                    }
                    else
                    {
                        // f_{i+1/2}^+
                        fRx = WENO5_Reconstruct(
                            f[i + 3][j][k][l],
                            f[i + 2][j][k][l],
                            f[i + 1][j][k][l],
                            f[i][j][k][l],
                            f[i - 1][j][k][l]);
                        // f_{i-1/2}^+
                        fLx = WENO5_Reconstruct(
                            f[i + 2][j][k][l],
                            f[i + 1][j][k][l],
                            f[i][j][k][l],
                            f[i - 1][j][k][l],
                            f[i - 2][j][k][l]);
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
                            f[i][j - 2][k][l],
                            f[i][j - 1][k][l],
                            f[i][j][k][l],
                            f[i][j + 1][k][l],
                            f[i][j + 2][k][l]);
                        // f_{j-1/2}^-
                        fLy = WENO5_Reconstruct(
                            f[i][j - 3][k][l],
                            f[i][j - 2][k][l],
                            f[i][j - 1][k][l],
                            f[i][j][k][l],
                            f[i][j + 1][k][l]);
                    }
                    else
                    {
                        // f_{i+1/2}^+
                        fRy = WENO5_Reconstruct(
                            f[i][j + 3][k][l],
                            f[i][j + 2][k][l],
                            f[i][j + 1][k][l],
                            f[i][j][k][l],
                            f[i][j - 1][k][l]);
                        // f_{i-1/2}^+
                        fLy = WENO5_Reconstruct(
                            f[i][j + 2][k][l],
                            f[i][j + 1][k][l],
                            f[i][j][k][l],
                            f[i][j - 1][k][l],
                            f[i][j - 2][k][l]);
                    }
                    // positivityLimiter(fbar, fLy, fRy);

                    double Fy_plus = vy[l] * fRy;
                    double Fy_minus = vy[l] * fLy;

                    Lf[i][j][k][l] = -((Fx_plus - Fx_minus) / dx +
                                       (Fy_plus - Fy_minus) / dy);
                }
}
double compute_max_entropy(const vector<vector<vector<vector<double>>>> &f, Parameters params)
{
    double entropy = 0.0;
    for (int i = 3; i < params.Nx - 3; ++i)
        for (int j = 3; j < params.Ny - 3; ++j)
            for (int k = 0; k < params.Nv; ++k)
                for (int l = 0; l < params.Nv; ++l)
                {
                    {
                        {
                            double fbar = f[i][j][k][l];
                            if (fbar > 0)
                                entropy += fbar * log(fbar);
                            // entropy += fbar * log(fbar) - fbar;
                        }
                    }
                }
    return entropy;
}
double totalVariation2D(
    const vector<vector<double>> &q,
    const Parameters &params)
{
    double tv = 0.0;

    for (int i = 0; i < params.Nx - 1; ++i)
        for (int j = 0; j < params.Ny - 1; ++j)
        {
            tv += std::abs(q[i + 1][j] - q[i][j]);
            tv += std::abs(q[i][j + 1] - q[i][j]);
        }

    return tv;
}
void print_vector(std::string filename, const vector<double> &data)
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
    // #pragma omp parallel
    //     {
    // #pragma omp master
    //         std::cout << "Running with " << omp_get_num_threads() << " threads.\n";
    //     }
    Parameters params;
    params.Nx = 101;
    params.Ny = 51;
    params.Nv = 15;
    params.Lx = 1.0;
    params.Ly = 1.0;
    params.T_end = 0.2;
    params.gamma_gas = 2.0;
    params.tau = 1e-4; // nearly collisionless
    params.cfl = 1.5;

    double dx = params.Lx / (params.Nx - 1);
    double dy = params.Ly / (params.Ny - 1);
    double dv = 6.0 / (params.Nv - 1);
    // velocity grid
    vector<double> vx(params.Nv), vy(params.Nv);
    for (int k = 0; k < params.Nv; ++k)
    {
        vx[k] = -3.0 + k * dv;
        vy[k] = -3.0 + k * dv;
    }

    // f[x][y][vx][vy]
    vector<vector<vector<vector<double>>>> f(params.Nx, vector<vector<vector<double>>>(params.Ny,
                                                                                       vector<vector<double>>(params.Nv, vector<double>(params.Nv))));
    // macroscopic fields
    vector<vector<double>> rho(params.Nx, vector<double>(params.Ny));
    vector<vector<double>> ux(params.Nx, vector<double>(params.Ny));
    vector<vector<double>> uy(params.Nx, vector<double>(params.Ny));
    vector<vector<double>> T(params.Nx, vector<double>(params.Ny));

    double t = 0.0;
    vector<double> entropy;
    vector<double> TV_rho;
    // -------- Initialization --------
    initialize(f, rho, ux, uy, vx, vy, params);
    // writeVTK2D(rho, ux, uy, params.Nx, params.Ny, 0);
    //  ---------------- Time loop ----------------

    auto Fex = f;
    auto Fim = f;
    while (t < params.T_end)
    {
        double vmax = maxVelocity(vx, vy, params);
        double dt = params.cfl * min(dx, dy) / (vmax);
        //    double dt = params.cfl * min(dx, dy) / (20.0 * vmax);
        if (t + dt > params.T_end)
            dt = params.T_end - t;

        auto f0 = f;
        auto f1 = f;
        auto f2 = f;
        auto f3 = f;
        auto F1 = f;
        auto F2 = f;

        // ---------- Stage 1 ----------
        computeMacroscopic(f, rho, ux, uy, vx, vy, T, params, dv);
        TV_rho.push_back(totalVariation2D(rho, params));
        for (int i = 0; i < params.Nx; i++)
        {
            for (int j = 0; j < params.Ny; j++)
            {
                for (int k = 0; k < params.Nv; k++)
                {
                    for (int l = 0; l < params.Nv; l++)
                    {
                        double alpha = 0.5 * dt / params.tau;
                        double M = maxwellian(
                            rho[i][j], ux[i][j], uy[i][j], T[i][j],
                            vx[k], vy[l]);
                        f1[i][j][k][l] =
                            (f[i][j][k][l] + alpha * M) / (1.0 + alpha);
                    }
                }
            }
        }
        setWENOboundaryConditions(f1, rho, ux, uy, T, params);
        // setWENOboundaryConditions(f1, params);

        // ---------- Stage 2 ----------
        computeMacroscopic(f1, rho, ux, uy, vx, vy, T, params, dv);
        computeFexplicit(f1, Fex, vx, vy, dt, dx, dy, params);
        for (int i = 0; i < params.Nx; i++)
        {
            for (int j = 0; j < params.Ny; j++)
            {
                for (int k = 0; k < params.Nv; k++)
                {
                    for (int l = 0; l < params.Nv; l++)
                    {
                        double alpha = 0.5 * dt * dt / params.tau;
                        double M = maxwellian(
                            rho[i][j], ux[i][j], uy[i][j], T[i][j],
                            vx[k], vy[l]);
                        f2[i][j][k][l] =
                            (f1[i][j][k][l] + dt * Fex[i][j][k][l] + alpha * M) / (1.0 + alpha);
                    }
                }
            }
        }
        // setWENOboundaryConditions(f2, vx, vy, params);
        setWENOboundaryConditions(f2, rho, ux, uy, T, params);
        // setWENOboundaryConditions(f2, params);
        //  ---------- Stage 3 ----------
        computeMacroscopic(f2, rho, ux, uy, vx, vy, T, params, dv);
        computeFexplicit(f2, Fex, vx, vy, dt, dx, dy, params);
        for (int i = 0; i < params.Nx; i++)
        {
            for (int j = 0; j < params.Ny; j++)
            {
                for (int k = 0; k < params.Nv; k++)
                {
                    for (int l = 0; l < params.Nv; l++)
                    {
                        double alpha = 0.5 * dt / params.tau;
                        double M = maxwellian(
                            rho[i][j], ux[i][j], uy[i][j], T[i][j],
                            vx[k], vy[l]);
                        f3[i][j][k][l] =
                            (0.5 * (f1[i][j][k][l] + f2[i][j][k][l] + dt * Fex[i][j][k][l]) + alpha * M) / (1.0 + alpha);
                    }
                }
            }
        }
        // setWENOboundaryConditions(f3, vx, vy, params);
        setWENOboundaryConditions(f3, rho, ux, uy, T, params);
        // setWENOboundaryConditions(f3, params);
        std::cout << "Residual at time " << t << ": " << std::scientific << std::setw(10) << std::setprecision(3) << computeResidual(f, f3, params) << "\n";
        f.swap(f3);
        t += dt;
        entropy.push_back(compute_max_entropy(f, params) * dx * dy * dv * dv);
        // cout << "t = " << t << endl;
        writeVTK2D(rho, ux, uy, T, params.Nx, params.Ny, dx, dy, int(t / dt));
    }
    print_vector("./entropy_1eminus4_cfl_1.5.txt", entropy);
    print_vector("./TV_rho_1eminus4_cfl_1.5.txt", TV_rho);
}

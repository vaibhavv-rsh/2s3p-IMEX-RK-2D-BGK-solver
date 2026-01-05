#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

// global enums for easy variable access
enum SCHEME
{
    CONSTANT = 0,
    MUSCL
};
enum LIMITER
{
    NONE = 0,
    MINMOD,
    VANLEER
};
enum FACE
{
    WEST = 0,
    EAST
};

// definition for case parameter structure to hold case-specific settings
struct caseParameters
{
    int Nx, Ny;
    double gamma;
    double Lx, Ly;
    double endTime;
    double CFL;
    double dx, dy;
    double time;
    int timeStep;
};

double totalVariation2D(
    const std::vector<std::vector<std::array<double, 4>>> &q,
    const caseParameters &params)
{
    double tv = 0.0;

    for (int i = 0; i < params.Nx - 1; ++i)
        for (int j = 0; j < params.Ny - 1; ++j)
        {
            tv += std::abs(q[i + 1][j][0] - q[i][j][0]);
            tv += std::abs(q[i][j + 1][0] - q[i][j][0]);
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

    /* ---------- Pre-processing ---------- */

    // Read parameters
    auto numericalScheme = SCHEME::MUSCL;
    auto limiter = LIMITER::VANLEER;
    caseParameters parameters;

    parameters.Nx = 101, parameters.Ny = 51;
    // parameters.gamma = 1.4;
    parameters.gamma = 2.0;
    parameters.Lx = 1.0, parameters.Ly = 1.0;
    parameters.endTime = 0.2;
    parameters.CFL = 0.3;

    parameters.dx = parameters.Lx / (parameters.Nx - 1);
    parameters.dy = parameters.Ly / (parameters.Ny - 1);
    parameters.time = 0.0;
    parameters.timeStep = 0;

    // Allocate memory
    std::vector<double> x(parameters.Nx);
    std::vector<double> y(parameters.Ny);
    std::vector<std::vector<std::array<double, 4>>> U(parameters.Nx,
                                                      std::vector<std::array<double, 4>>(parameters.Ny));
    std::vector<std::vector<std::array<std::array<double, 4>, 2>>> Uxfaces(parameters.Nx,
                                                                           std::vector<std::array<std::array<double, 4>, 2>>(parameters.Ny));
    std::vector<std::vector<std::array<std::array<double, 4>, 2>>> Uyfaces(parameters.Nx,
                                                                           std::vector<std::array<std::array<double, 4>, 2>>(parameters.Ny));
    std::vector<std::vector<std::array<std::array<double, 4>, 2>>> Fxfaces(parameters.Nx,
                                                                           std::vector<std::array<std::array<double, 4>, 2>>(parameters.Ny));
    std::vector<std::vector<std::array<std::array<double, 4>, 2>>> Fyfaces(parameters.Nx,
                                                                           std::vector<std::array<std::array<double, 4>, 2>>(parameters.Ny));

    // Create/read mesh
    for (int i = 0; i < parameters.Nx; i++)
    {
        x[i] = i * parameters.dx;
    }
    for (int j = 0; j < parameters.Ny; j++)
    {
        y[j] = j * parameters.dy;
    }

    // Initialise solution
    double rho = 0.0;
    double u = 0.0;
    double v = 0.0;
    double p = 0.0;

    for (int i = 0; i < parameters.Nx; i++)
    {
        for (int j = 0; j < parameters.Ny; j++)
        {
            if (x[i] <= 0.5)
            {
                rho = 1.0;
                u = 0.0;
                v = 0.0;
                p = 1.0;
            }
            else
            {
                rho = 0.125;
                u = 0.0;
                v = 0.0;
                p = 0.1;
            }

            U[i][j][0] = rho;
            U[i][j][1] = rho * u;
            U[i][j][2] = rho * v;
            U[i][j][3] = p / (parameters.gamma - 1.0) + 0.5 * rho * (std::pow(u, 2) + std::pow(v, 2));
        }
    }
    std::vector<double> TV_rho;
    /* ---------- Solving ---------- */
    while (parameters.time < parameters.endTime)
    {

        // Preparing solution update (store old solution and calculate stable timestep)
        auto UOld = U;

        // calculate stable time step
        double speedMax = 0.0;
        for (int i = 0; i < parameters.Nx; i++)
        {
            for (int j = 0; j < parameters.Ny; j++)
            {
                // we need the primitive variables first to compute the wave speed (based on speed of sound and local velocity)
                double rho = U[i][j][0];
                double u = U[i][j][1] / rho;
                double v = U[i][j][2] / rho;
                double p = (parameters.gamma - 1.0) * (U[i][j][3] - 0.5 * rho * (std::pow(u, 2) + std::pow(v, 2)));

                // calculate wave speed for each cell
                double a = std::sqrt(parameters.gamma * p / rho);
                speedMax = std::max(speedMax,
                                    std::fabs(u) + a + std::fabs(v) + a);
            }
        }
        double dt = parameters.CFL / (speedMax * (1.0 / parameters.dx + 1.0 / parameters.dy));

        // Solve equations

        if (numericalScheme == SCHEME::MUSCL)
        {
            // use lower-order scheme near boundaries
            for (int j = 0; j < parameters.Ny; ++j)
            {
                for (int variable = 0; variable < 4; ++variable)
                {
                    // Left boundary (i = 0)
                    Uxfaces[0][j][FACE::WEST][variable] = U[0][j][variable];
                    Uxfaces[0][j][FACE::EAST][variable] = U[0][j][variable];

                    // Right boundary (i = Nx-1)
                    Uxfaces[parameters.Nx - 1][j][FACE::WEST][variable] =
                        U[parameters.Nx - 1][j][variable];
                    Uxfaces[parameters.Nx - 1][j][FACE::EAST][variable] =
                        U[parameters.Nx - 1][j][variable];
                }
            }
            for (int i = 0; i < parameters.Nx; ++i)
            {
                for (int variable = 0; variable < 4; ++variable)
                {
                    // Bottom boundary (j = 0)
                    Uyfaces[i][0][FACE::WEST][variable] = U[i][0][variable];
                    Uyfaces[i][0][FACE::EAST][variable] = U[i][0][variable];

                    // Top boundary (j = Ny-1)
                    Uyfaces[i][parameters.Ny - 1][FACE::WEST][variable] =
                        U[i][parameters.Ny - 1][variable];
                    Uyfaces[i][parameters.Ny - 1][FACE::EAST][variable] =
                        U[i][parameters.Ny - 1][variable];
                }
            }

            // use high-resolution MUSCL scheme on interior nodes / cells
            for (int i = 1; i < parameters.Nx - 1; i++)
            {
                for (int j = 0; j < parameters.Ny; j++)
                {
                    for (int variable = 0; variable < 4; ++variable)
                    {
                        auto duR = U[i + 1][j][variable] - U[i][j][variable];
                        auto duL = U[i][j][variable] - U[i - 1][j][variable];

                        double rL = duL / (duR + 1e-8);
                        double rR = duR / (duL + 1e-8);

                        double psiL = 1.0;
                        double psiR = 1.0;

                        // apply limiter to make scheme TVD (total variation diminishing)
                        if (limiter == LIMITER::MINMOD)
                        {
                            psiL = std::max(0.0, std::min(1.0, rL));
                            psiR = std::max(0.0, std::min(1.0, rR));
                        }
                        else if (limiter == LIMITER::VANLEER)
                        {
                            psiL = (rL + std::fabs(rL)) / (1.0 + std::fabs(rL));
                            psiR = (rR + std::fabs(rR)) / (1.0 + std::fabs(rR));
                        }

                        Uxfaces[i][j][FACE::WEST][variable] = U[i][j][variable] - 0.5 * psiL * duR;
                        Uxfaces[i][j][FACE::EAST][variable] = U[i][j][variable] + 0.5 * psiR * duL;
                    }
                }
            }

            for (int i = 0; i < parameters.Nx; i++)
            {
                for (int j = 1; j < parameters.Ny - 1; j++)
                {
                    for (int variable = 0; variable < 4; ++variable)
                    {
                        auto duT = U[i][j + 1][variable] - U[i][j][variable];
                        auto duB = U[i][j][variable] - U[i][j - 1][variable];

                        double rB = duB / (duT + 1e-8);
                        double rT = duT / (duB + 1e-8);

                        double psiB = 1.0;
                        double psiT = 1.0;

                        // apply limiter to make scheme TVD (total variation diminishing)
                        if (limiter == LIMITER::MINMOD)
                        {
                            psiB = std::max(0.0, std::min(1.0, rB));
                            psiT = std::max(0.0, std::min(1.0, rT));
                        }
                        else if (limiter == LIMITER::VANLEER)
                        {
                            psiB = (rB + std::fabs(rB)) / (1.0 + std::fabs(rB));
                            psiT = (rT + std::fabs(rT)) / (1.0 + std::fabs(rT));
                        }

                        Uyfaces[i][j][FACE::WEST][variable] = U[i][j][variable] - 0.5 * psiB * duT;
                        Uyfaces[i][j][FACE::EAST][variable] = U[i][j][variable] + 0.5 * psiT * duB;
                    }
                }
            }
        }

        // compute fluxes at faces
        std::array<double, 4> fluxL, fluxR;
        for (int i = 1; i < parameters.Nx - 1; i++)
        {
            for (int j = 0; j < parameters.Ny; j++)
            {
                for (int face = FACE::WEST; face <= FACE::EAST; ++face)
                {
                    int indexOffset = 0;
                    if (face == FACE::WEST)
                        indexOffset = 0;
                    else if (face == FACE::EAST)
                        indexOffset = 1;

                    auto rhoL = Uxfaces[i - 1 + indexOffset][j][FACE::EAST][0];
                    auto uL = Uxfaces[i - 1 + indexOffset][j][FACE::EAST][1] / rhoL;
                    auto vL = Uxfaces[i - 1 + indexOffset][j][FACE::EAST][2] / rhoL;
                    auto EL = Uxfaces[i - 1 + indexOffset][j][FACE::EAST][3];
                    auto pL = (parameters.gamma - 1.0) * (EL - 0.5 * rhoL * (std::pow(uL, 2) + std::pow(vL, 2)));
                    auto aL = std::sqrt(parameters.gamma * pL / rhoL);

                    auto rhoR = Uxfaces[i + indexOffset][j][FACE::WEST][0];
                    auto uR = Uxfaces[i + indexOffset][j][FACE::WEST][1] / rhoR;
                    auto vR = Uxfaces[i + indexOffset][j][FACE::WEST][2] / rhoR;
                    auto ER = Uxfaces[i + indexOffset][j][FACE::WEST][3];
                    auto pR = (parameters.gamma - 1.0) * (ER - 0.5 * rhoR * (uR * uR + vR * vR));
                    auto aR = std::sqrt(parameters.gamma * pR / rhoR);

                    fluxL[0] = rhoL * uL;
                    fluxL[1] = rhoL * uL * uL + pL;
                    fluxL[2] = rhoL * uL * vL;
                    fluxL[3] = uL * (EL + pL);

                    fluxR[0] = rhoR * uR;
                    fluxR[1] = rhoR * uR * uR + pR;
                    fluxR[2] = rhoR * uR * vR;
                    fluxR[3] = uR * (ER + pR);

                    // Rusanov Riemann solver
                    auto speedMax = std::max(std::fabs(uL) + aL, std::fabs(uR) + aR);
                    for (int variable = 0; variable < 4; ++variable)
                    {
                        const auto &qL = Uxfaces[i - 1 + indexOffset][j][FACE::EAST][variable];
                        const auto &qR = Uxfaces[i + indexOffset][j][FACE::WEST][variable];
                        const auto &fL = fluxL[variable];
                        const auto &fR = fluxR[variable];
                        Fxfaces[i][j][face][variable] = 0.5 * (fL + fR) - speedMax * (qR - qL);
                    }
                }
            }
        }
        std::array<double, 4> fluxB, fluxT;

        for (int i = 0; i < parameters.Nx; i++)
        {
            for (int j = 1; j < parameters.Ny - 1; j++)
            {
                for (int face = FACE::WEST; face <= FACE::EAST; ++face)
                {
                    int indexOffset = (face == FACE::WEST) ? 0 : 1;

                    // ----- Bottom state -----
                    auto rhoB = Uyfaces[i][j - 1 + indexOffset][FACE::EAST][0];
                    auto uB = Uyfaces[i][j - 1 + indexOffset][FACE::EAST][1] / rhoB;
                    auto vB = Uyfaces[i][j - 1 + indexOffset][FACE::EAST][2] / rhoB;
                    auto EB = Uyfaces[i][j - 1 + indexOffset][FACE::EAST][3];

                    auto pB = (parameters.gamma - 1.0) * (EB - 0.5 * rhoB * (uB * uB + vB * vB));
                    auto aB = std::sqrt(parameters.gamma * pB / rhoB);

                    // ----- Top state -----
                    auto rhoT = Uyfaces[i][j + indexOffset][FACE::WEST][0];
                    auto uT = Uyfaces[i][j + indexOffset][FACE::WEST][1] / rhoT;
                    auto vT = Uyfaces[i][j + indexOffset][FACE::WEST][2] / rhoT;
                    auto ET = Uyfaces[i][j + indexOffset][FACE::WEST][3];

                    auto pT = (parameters.gamma - 1.0) * (ET - 0.5 * rhoT * (uT * uT + vT * vT));
                    auto aT = std::sqrt(parameters.gamma * pT / rhoT);

                    // ----- Physical fluxes in y-direction -----
                    fluxB[0] = rhoB * vB;
                    fluxB[1] = rhoB * uB * vB;
                    fluxB[2] = rhoB * vB * vB + pB;
                    fluxB[3] = vB * (EB + pB);

                    fluxT[0] = rhoT * vT;
                    fluxT[1] = rhoT * uT * vT;
                    fluxT[2] = rhoT * vT * vT + pT;
                    fluxT[3] = vT * (ET + pT);

                    // ----- Rusanov solver -----
                    auto speedMax = std::max(std::fabs(vB) + aB,
                                             std::fabs(vT) + aT);

                    for (int variable = 0; variable < 4; variable++)
                    {
                        const auto &qB =
                            Uyfaces[i][j - 1 + indexOffset][FACE::EAST][variable];
                        const auto &qT =
                            Uyfaces[i][j + indexOffset][FACE::WEST][variable];

                        Fyfaces[i][j][face][variable] =
                            0.5 * (fluxB[variable] + fluxT[variable]) - 0.5 * speedMax * (qT - qB);
                    }
                }
            }
        }
        // calculate updated solution
        for (int i = 1; i < parameters.Nx - 1; i++)
            for (int j = 1; j < parameters.Ny - 1; j++)
                for (int k = 0; k < 4; k++)
                {
                    const auto &dFx = Fxfaces[i][j][FACE::EAST][k] - Fxfaces[i][j][FACE::WEST][k];
                    const auto &dFy = Fyfaces[i][j][FACE::EAST][k] - Fyfaces[i][j][FACE::WEST][k];
                    U[i][j][k] = UOld[i][j][k] - ((dt / parameters.dx) * dFx + (dt / parameters.dy) * dFy);
                }

        // Update boundary conditions
        auto rhoL = 1.0;
        auto uL = 0.0;

        auto rhoR = 0.125;
        auto uR = 0.0;

        // LEFT boundary (i = 0)
        for (int j = 0; j < parameters.Ny; j++)
        {
            U[0][j][0] = U[1][j][0]; // rho
            U[0][j][1] = 0.0;        // rho*u → u = 0
            U[0][j][2] = U[1][j][2]; // rho*v (copy tangential)
            U[0][j][3] = U[1][j][3]; // energy
        }

        // RIGHT boundary (i = Nx-1)
        for (int j = 0; j < parameters.Ny; j++)
        {
            U[parameters.Nx - 1][j][0] = U[parameters.Nx - 2][j][0];
            U[parameters.Nx - 1][j][1] = 0.0;
            U[parameters.Nx - 1][j][2] = U[parameters.Nx - 2][j][2];
            U[parameters.Nx - 1][j][3] = U[parameters.Nx - 2][j][3];
        }
        for (int i = 0; i < parameters.Nx; i++)
        {
            U[i][0][0] = U[i][1][0];
            U[i][0][1] = U[i][1][1]; // rho*u (copy tangential)
            U[i][0][2] = 0.0;        // rho*v → v = 0
            U[i][0][3] = U[i][1][3];
        }

        // TOP boundary (j = Ny-1)
        for (int i = 0; i < parameters.Nx; i++)
        {
            U[i][parameters.Ny - 1][0] = U[i][parameters.Ny - 2][0];
            U[i][parameters.Ny - 1][1] = U[i][parameters.Ny - 2][1];
            U[i][parameters.Ny - 1][2] = 0.0;
            U[i][parameters.Ny - 1][3] = U[i][parameters.Ny - 2][3];
        }

        std::string output_dir = "./euler_gamma_2/";
        std::string mkdir_command = "mkdir -p " + output_dir;
        std::system(mkdir_command.c_str());
        std::ostringstream time_step_temp, points_x_temp, points_y_temp;
        // time_step_temp << std::setfill('0') << std::setw(6);
        time_step_temp << parameters.timeStep;
        auto time_step = time_step_temp.str();
        points_x_temp << std::setfill('0') << std::setw(6);
        points_x_temp << parameters.Nx;
        auto points_x = points_x_temp.str();
        points_y_temp << std::setfill('0') << std::setw(6);
        points_y_temp << parameters.Ny;
        auto points_y = points_y_temp.str();
        std::string vtk_file = output_dir + "solution" + "_" + time_step + ".vtk";
        std::ofstream vtk_output;
        vtk_output.open(vtk_file);
        vtk_output << "# vtk DataFile Version 2.0\n";
        vtk_output << "2D Euler Solution\n";
        vtk_output << "ASCII\n";
        vtk_output << "DATASET STRUCTURED_GRID\n";
        vtk_output << "DIMENSIONS " << parameters.Nx << " " << parameters.Ny << " 1\n";
        vtk_output << "POINTS " << parameters.Nx * parameters.Ny << " float\n";

        for (int j = 0; j < parameters.Ny; j++)
        {
            for (int i = 0; i < parameters.Nx; i++)
            {
                vtk_output << x[i] << " " << y[j] << " 0\n";
            }
        }
        vtk_output << "CELL_DATA " << (parameters.Nx - 1) * (parameters.Ny - 1) << "\n";
        vtk_output << "SCALARS rho float 1\nLOOKUP_TABLE default\n";
        for (int j = 0; j < parameters.Ny - 1; j++)
        {
            for (int i = 0; i < parameters.Nx - 1; i++)
            {
                vtk_output << U[i][j][0] << "\n";
            }
        }

        vtk_output << "VECTORS velocity float\n";
        for (int j = 0; j < parameters.Ny - 1; j++)
        {
            for (int i = 0; i < parameters.Nx - 1; i++)
            {
                double rho = U[i][j][0];
                double u = U[i][j][1] / rho;
                double v = U[i][j][2] / rho;
                vtk_output << u << " " << v << " 0\n";
            }
        }

        vtk_output << "SCALARS pressure float 1\nLOOKUP_TABLE default\n";
        for (int j = 0; j < parameters.Ny - 1; j++)
        {
            for (int i = 0; i < parameters.Nx - 1; i++)
            {
                double rho = U[i][j][0];
                double u = U[i][j][1] / rho;
                double v = U[i][j][2] / rho;
                double p = (parameters.gamma - 1) * (U[i][j][3] - 0.5 * rho * (std::pow(u, 2) + std::pow(v, 2)));
                vtk_output << p << "\n";
            }
        }

        vtk_output.close();

        // output current time step information to screen
        std::cout << "Current time: " << std::scientific << std::setw(10) << std::setprecision(3) << parameters.time;
        std::cout << ", End time: " << std::scientific << std::setw(10) << std::setprecision(3) << parameters.endTime;
        std::cout << ", Current time step: " << std::fixed << std::setw(7) << parameters.timeStep;
        std::cout << "\r";

        // Increment solution time and time step
        parameters.time += dt;
        parameters.timeStep++;
        std::cout << "Solution written to: " << vtk_file << std::endl;

        TV_rho.push_back(totalVariation2D(U, parameters));
    }

    std::cout << "\nSimulation finished" << std::endl;
    print_vector("./TV_rho_euler.txt", TV_rho);

    return 0;
}

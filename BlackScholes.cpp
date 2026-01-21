//
//  main.cpp
//  Black_Scholes PDE for a european call option using Finite difference Methods
//
//  Created by Aryan Goel on 20/01/26.
//

#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <iostream>

// Standard normal CDF approximation
double normCDF(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

//Analytical Black-Scholes price for European Call
double blackScholesCall(double S, double K, double T, double r, double sigma){
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return S * normCDF(d1) - K * std::exp(-r * T) * normCDF(d2);
}

// PayOff class
class PayOff {
public:
    virtual ~PayOff() {}
    virtual double operator()(double S) const = 0;
};
class PayOffCall : public PayOff {

private:
    double K;
public:
    PayOffCall(double K_) : K(K_) {}
    double operator()(double S) const override {
        return std::max(S-K, 0.0);
    }
};

// VanillaOption Class
class VanillaOption {
private:
    double K;
    double T;
    std::shared_ptr<PayOff> payOffPtr;
public:
    VanillaOption(double K_, double T_, std::shared_ptr<PayOff> payOff_): K(K_), T(T_), payOffPtr(payOff_) {}
    double getStrike() const {return K; }
    double getMaturity() const {return T; }
    double payOff(double S) const {return (*payOffPtr)(S); }
};

// Convention DiffusionPDE
class ConvectionDiffusionPDE {
public:
    virtual ~ConvectionDiffusionPDE() {}
    virtual double diffusion(double S, double t) const = 0;
    virtual double convection(double S, double t) const = 0;
    virtual double reaction(double S, double t) const = 0;
    virtual double boundryLeft(double t, double V) const = 0;
    virtual double boundryRight(double t, double V) const = 0;
    virtual double initialCondition(double S) const = 0;
};

// BlackScholesPDE

class BlackScholesPDE : public ConvectionDiffusionPDE {
private:
    std::shared_ptr<VanillaOption> option;
    double r;
    double sigma;
public:
    BlackScholesPDE(std::shared_ptr<VanillaOption> option_, double r_, double sigma_) : option(option_), r(r_), sigma(sigma_) {}
    std::shared_ptr<VanillaOption> getOption() const { return option; }
    double diffusion(double S, double t) const override {
        return 0.5 * sigma * sigma * S * S;
    }
    double convection(double S, double t) const override {
        return r * S;
    }
    double reaction(double S, double t) const override {
        return -r;
    }
    double boundryLeft(double t, double V) const override {
        return 0.0;
    }
    double boundryRight(double t, double V) const override {
        double T = option->getMaturity();
        double K = option->getStrike();
        return S_max - K * std::exp(-r * (T-t));
    }
    double initialCondition(double S) const override{
        return option->payOff(S);
    }
    double S_max;
};

// FDMBase
class FDMBase {
protected:
    std::shared_ptr<ConvectionDiffusionPDE> pde;
    double S_max;
    double T;
    int N_S;
    int N_t;
    double dS;
    double dt;
    std::vector<std::vector<double>> V;
    
public:
    FDMBase(std::shared_ptr<ConvectionDiffusionPDE> pde_, double S_max_,int N_S_, int N_t_) : pde( pde_ ), S_max(S_max_), N_S(N_S_), N_t(N_t_) {
        T = std::dynamic_pointer_cast<BlackScholesPDE>(pde_)->getOption()->getMaturity();
        dS = S_max / N_S;
        dt = T / N_t;
        V.resize(N_t + 1, std::vector<double>(N_S +1, 0.0));
        std::dynamic_pointer_cast<BlackScholesPDE>(pde_)->S_max = S_max;
    }
    virtual void calculate() = 0;
    std::vector<std::vector<double>> getSolution() const { return V; }
    double getPrice(double S) const {
        int i = static_cast<int>( S / dS );
        if(i>= N_S) return V[0][N_S];
        double S_i = i * dS;
        double S_ip1 = (i + 1) * dS;
        // linear interpolation
        return V[0][i] + (V[0][i+1] - V[0][i]) * (S - S_i) / (S_ip1 - S_i);
    }
};

// FDMEulerExplicit
class FDMEulerExplicit : public FDMBase {
public:
    FDMEulerExplicit(std::shared_ptr<ConvectionDiffusionPDE> pde_, double S_max_, int N_S_, int N_t_) : FDMBase( pde_, S_max_, N_S_, N_t_) {}
    void calculate() override {
        for (int i = 0; i<= N_S; ++i){
            double S = i * dS;
            V[N_t][i] = pde->initialCondition(S);
        }
        for (int n = N_t - 1; n>=0; --n) {
            double t = n * dt;
            V[n][0] = pde->boundryLeft(t, V[n+1][0]);
            V[n][N_S] = pde->boundryRight(t, V[n+1][N_S]);
            for(int i = 1; i < N_S; ++i) {
                double S = i * dS;
                double diff = pde->diffusion(S,t);
                double conv = pde->convection(S, t);
                double reac = pde->reaction(S, t);
                double a = (diff / (dS * dS)) - (conv / (2 * dS));
                double b = -2 * diff / (dS * dS) + reac;
                double c = (diff / (dS * dS)) + (conv / (2 * dS));
                V[n][i] = V[n+1][i] + dt * (
                    a * V[n+1][i-1] +
                    b * V[n+1][i] +
                    c * V[n+1][i+1]
                );
            }
        }
    }
};

int main() {
    // Parameters
    double K = 100;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;
    double S_max = 300.0;
    int N_S = 200;  // Increased Resolution
    int N_t = 2000; // Increased Resolution
    
    // Create Object
    auto payOff = std::make_shared<PayOffCall>(K);
    auto option = std::make_shared<VanillaOption>(K, T, payOff);
    auto pde = std::make_shared<BlackScholesPDE>(option, r, sigma);
    auto fdm = std::make_shared<FDMEulerExplicit>(pde, S_max, N_S, N_t);
    
    // Run FDM
    fdm->calculate();
    
    // Output results
    double S = 100.0;
    double fdmPrice = fdm->getPrice(S);
    double analyticalPrice = blackScholesCall(S, K, T, r, sigma);
    std::cout << "FDM price at S = " << S << ": " << fdmPrice << std::endl;
    std::cout << "Analytical price at S = " << S << "; " << analyticalPrice << std::endl;
    std::cout << "Absolute error: " << std::abs(fdmPrice - analyticalPrice) << std::endl;
}

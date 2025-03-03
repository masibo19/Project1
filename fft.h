//
// Created by 86180 on 2025/3/3.
//
#ifndef FFT_LIBRARY_H
#define FFT_LIBRARY_H

#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

namespace FFTLibrary {

    Eigen::MatrixXcd fft(const Eigen::MatrixXd& X); // Y = fft(X)
    Eigen::MatrixXcd fft(const Eigen::MatrixXd& X, int n); // Y = fft(X, n)
    Eigen::MatrixXcd fft(const Eigen::MatrixXd& X, int n, int dim); // Y = fft(X, n, dim)
    Eigen::MatrixXcd ifft(const Eigen::MatrixXcd& X);
    Eigen::MatrixXcd ifft(const Eigen::MatrixXcd& X, int n);
    Eigen::MatrixXcd ifft(const Eigen::MatrixXcd& X, int n, int dim);
}

#endif // FFT_LIBRARY_H


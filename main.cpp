#include <iostream>
#include "fft.h"

int main() {
    Eigen::MatrixXd X(4, 3);
    X << 1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12;

    //Y = fft(X)
    std::cout<< "X:\n" << X <<"\n";
    Eigen::MatrixXcd Y1 = FFTLibrary::fft(X);
    std::cout << "fft(X):\n" << Y1 << "\n\n";

    //Y = fft(X, n)
    Eigen::MatrixXcd Y2 = FFTLibrary::fft(X, 4);
    std::cout << "fft(X, 4):\n" << Y2 << "\n\n";

    //Y = fft(X, n, dim)
    Eigen::MatrixXcd Y3 = FFTLibrary::fft(X, 4, 2);
    std::cout << "fft(X, 4, 2):\n" << Y3 << "\n\n";

    return 0;
}
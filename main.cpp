#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
using namespace Eigen;

namespace FFTLibrary {
    Eigen::VectorXcd fft(const Eigen::VectorXd& input)
    {
        Eigen::FFT<double> fftObj;
        Eigen::VectorXcd result(input.size());
        fftObj.fwd(result, input);
        return result;
    }
}

int main() {
    Eigen::VectorXd vec(8);
    vec << 1, 2, 3, 4, 5, 6, 7, 8;

    Eigen::VectorXcd result = FFTLibrary::fft(vec);

    std::cout << "FFT result:" << std::endl;
    for (int i = 0; i < result.size(); ++i) {
        std::cout << result(i).real() << " + " << result(i).imag() << "i" << std::endl;
    }

    return 0;
}

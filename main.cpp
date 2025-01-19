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
Eigen::VectorXcd fft(const Eigen::VectorXd& x, int N) {
    Eigen::FFT<double> fftObj;
    int size = x.size();

    // 创建一个复数向量来存储结果，大小为N
    Eigen::VectorXcd result(N);

    // 创建一个临时向量，大小为N
    Eigen::VectorXd temp(N);

    if (size < N) {
        // 如果x的长度小于N，进行零填充
        temp.head(size) = x;
        temp.tail(N - size).setZero();
    } else {
        // 如果x的长度大于或等于N，进行截断
        temp = x.head(N);
    }

    // 执行FFT
    fftObj.fwd(result, temp);

    return result;
}
int main() {
    Eigen::VectorXd vec(8);
    vec << 1, 2, 3, 4, 5, 6, 7, 8;

    Eigen::VectorXcd result = FFTLibrary::fft(vec);

    std::cout << "FFT result:" << std::endl;
    for (int i = 0; i < result.size(); ++i) {
        std::cout << result(i).real() << " + " << result(i).imag() << "i" << std::endl;
    }
    Eigen::VectorXd v(8);
    v << 1, 2, 3, 4, 5, 6, 7, 8;

    // 指定N
    int N = 10;

    Eigen::VectorXcd result1 = fft(v, N);

    // 输出结果
    std::cout << "FFT result:" << std::endl;
    for (int i = 0; i < result1.size(); ++i) {
        std::cout << result1(i).real() << " + " << result1(i).imag() << "i" << std::endl;
    }
    return 0;
}

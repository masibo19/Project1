#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/FFT>
#include <complex>
using namespace Eigen;
using namespace std;
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


template<typename T>
MatrixXcd fft(const MatrixX<T>& X, int n, int dim) {
    FFT<T> fft;
    MatrixXcd Y = MatrixXcd::Zero(X.rows(), X.cols());

    // 将输入矩阵转换为复数矩阵
    MatrixXcd X_cd = X.cast<std::complex<T>>();

    if (dim == 1) { // 对列进行FFT
        for (int i = 0; i < X.cols(); ++i) {
            VectorXcd col = X_cd.col(i);
            fft.fwd(col);
            Y.col(i) = col;
        }
    } else if (dim == 2) { // 对行进行FFT
        for (int i = 0; i < X.rows(); ++i) {
            VectorXcd row = X_cd.row(i);
            fft.fwd(row);
            Y.row(i) = row;
        }
    } else {
        cerr << "Unsupported dimension for FFT." << endl;
    }

    return Y;
}

int main() {
    Eigen::VectorXd vec(8);
    vec << 1, 2, 3, 4, 5, 6, 7, 8;

    Eigen::VectorXcd result = FFTLibrary::fft(vec);

    std::cout << "FFT(X) result:" << std::endl;
    for (int i = 0; i < result.size(); ++i) {
        std::cout << result(i).real() << " + " << result(i).imag() << "i" << std::endl;
    }
    Eigen::VectorXd v(8);
    v << 1, 2, 3, 4, 5, 6, 7, 8;

    // 指定N
    int N = 10;

    Eigen::VectorXcd result1 = fft(v, N);

    // 输出结果
    std::cout << "FFT(X,n) result:" << std::endl;
    for (int i = 0; i < result1.size(); ++i) {
        std::cout << result1(i).real() << " + " << result1(i).imag() << "i" << std::endl;
    }
    MatrixXd X(3, 3);
    X << 1, 2, 3,
            4, 5, 6,
            7, 8, 9;


    MatrixXcd Y = fft<double>(X, 0, 2); // 对行进行FFT
    cout << "FFT(X,n,dim) result(dim=2):\n" << Y << endl;

    return 0;
}

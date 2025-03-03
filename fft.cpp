#include "fft.h"
#include <stdexcept>
namespace FFTLibrary {

    Eigen::MatrixXcd fft(const Eigen::MatrixXd& X) {
        return fft(X, X.rows(), 1); // 默认按第一个维度计算 FFT
    }

    Eigen::MatrixXcd fft(const Eigen::MatrixXd& X, int n) {
        return fft(X, n, 1); // 默认按第一个维度计算 FFT
    }

    Eigen::MatrixXcd fft(const Eigen::MatrixXd& X, int n, int dim) {
        Eigen::FFT<double> fft;
        Eigen::MatrixXcd Y;

        if (dim == 1) { // 沿列方向 FFT
            Y = Eigen::MatrixXcd::Zero(n, X.cols());
            for (int col = 0; col < X.cols(); ++col) {
                int length = std::min<int>(X.rows(), n); // 取 min 防止越界
                Eigen::VectorXcd colData = Eigen::VectorXcd::Zero(n);
                colData.head(length) = X.col(col).head(length).cast<std::complex<double>>(); // 复制 & 截断
                fft.fwd(colData, colData);
                Y.col(col) = colData;
            }
        } else if(dim == 2){ // dim == 2，沿行方向 FFT
            Y = Eigen::MatrixXcd::Zero(X.rows(), n);
            for (int row = 0; row < X.rows(); ++row) {
                int length = std::min<int>(X.cols(), n); // 取 min 防止越界
                Eigen::VectorXcd rowData = Eigen::VectorXcd::Zero(n);
                rowData.head(length) = X.row(row).head(length).cast<std::complex<double>>(); // 复制 & 截断
                fft.fwd(rowData, rowData);
                Y.row(row) = rowData.transpose();
            }
        } else {
            throw std::invalid_argument("dim must be 1 or 2");
        }
        return Y;
    }

} // namespace FFTLibrary
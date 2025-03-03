#include "fft.h"

namespace FFTLibrary {

    Eigen::MatrixXcd fft(const Eigen::MatrixXd& X) {
        return fft(X, X.rows(), 0); // 默认按第一个维度计算 FFT
    }

    Eigen::MatrixXcd fft(const Eigen::MatrixXd& X, int n) {
        return fft(X, n, 0); // 默认按第一个维度计算 FFT
    }

    Eigen::MatrixXcd fft(const Eigen::MatrixXd& X, int n, int dim) {
        Eigen::FFT<double> fft;
        Eigen::MatrixXcd Y;

        if (dim == 0) { // 按行计算 FFT
            Y = Eigen::MatrixXcd::Zero(n, X.cols());
            for (int col = 0; col < X.cols(); ++col) {
                Eigen::VectorXcd colData(n);
                colData.head(X.rows()) = X.col(col).cast<std::complex<double>>();
                colData.tail(n - X.rows()).setZero(); // 零填充
                fft.fwd(colData, colData);
                Y.col(col) = colData;
            }
        } else if (dim == 1) { // 按列计算 FFT
            Y = Eigen::MatrixXcd::Zero(X.rows(), n);
            for (int row = 0; row < X.rows(); ++row) {
                Eigen::VectorXcd rowData(n);
                rowData.head(X.cols()) = X.row(row).cast<std::complex<double>>();
                rowData.tail(n - X.cols()).setZero(); // 零填充
                fft.fwd(rowData, rowData);
                Y.row(row) = rowData.transpose();
            }
        } else {
            throw std::invalid_argument("dim must be 0 or 1");
        }

        return Y;
    }

} // namespace FFTLibrary
#include "fft.h"
#include <stdexcept>
#include <cmath>
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
    Eigen::MatrixXcd ifft(const Eigen::MatrixXcd& X) {
        return ifft(X, X.rows(), 1); // 默认按第一个维度计算 FFT
    }

    Eigen::MatrixXcd ifft(const Eigen::MatrixXcd& X, int n) {
        return ifft(X, n, 1); // 默认按第一个维度计算 FFT
    }
    Eigen::MatrixXcd ifft(const Eigen::MatrixXcd& X, int n, int dim) {
        Eigen::FFT<double> fft;
        Eigen::MatrixXcd Y;

        if (dim == 1) { // 沿列方向 IFFT
            Y = Eigen::MatrixXcd::Zero(n, X.cols());
            for (int col = 0; col < X.cols(); ++col) {
                int length = std::min<int>(X.rows(), n); // 取 min 防止越界
                Eigen::VectorXcd colData = Eigen::VectorXcd::Zero(n);
                colData.head(length) = X.col(col).head(length).cast<std::complex<double>>(); // 复制 & 截断
                fft.inv(colData, colData);  // 使用逆变换（inv）
                Y.col(col) = colData;
            }
        } else if (dim == 2) { // dim == 2，沿行方向 IFFT
            Y = Eigen::MatrixXcd::Zero(X.rows(), n);
            for (int row = 0; row < X.rows(); ++row) {
                int length = std::min<int>(X.cols(), n); // 取 min 防止越界
                Eigen::VectorXcd rowData = Eigen::VectorXcd::Zero(n);
                rowData.head(length) = X.row(row).head(length).cast<std::complex<double>>(); // 复制 & 截断
                fft.inv(rowData, rowData);  // 使用逆变换（inv）
                Y.row(row) = rowData.transpose();
            }
        } else {
            throw std::invalid_argument("dim must be 1 or 2");
        }

        return Y;
    }
    std::vector<float> convolve(std::vector<float>& r1, std::vector<float>& r2) {

        Eigen::FFT<float> fft;
        std::vector<float> result;

        std::vector<std::complex<float> > fv1;
        std::vector<std::complex<float> > fv2;
        std::vector<std::complex<float> > mulvec;

        //computes cumulative length of r1 and r2;
        size_t N = r1.size() + r2.size() - 1 ;
        size_t M = pow(2, (log(N)/log(2)+1));

        //padds signals with zeros
        while( r1.size() != M ) r1.emplace_back(0.0);
        while( r2.size() != M ) r2.emplace_back(0.0);

        //Performs FWD FFT1
        fft.fwd(fv1, r1);

        //Performs FWD FFT2
        fft.fwd(fv2, r2);

        //multiply FV1 * FV2, element-wise
        std::transform( fv1.begin(), fv1.end(),
                        fv2.begin(), std::back_inserter(mulvec),
                        std::multiplies< std::complex<float> >() );

        //FFT INVERSE
        fft.inv(result, mulvec);
        return result;

    }

} // namespace FFTLibrary
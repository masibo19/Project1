#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
using namespace Eigen;
namespace FFTLibrary {
    Eigen::VectorXcd fft(const Eigen::VectorXd& input)
    {
        // 创建FFT对象
        Eigen::FFT<double> fftObj;

        // 创建一个复数向量来存储结果，大小与输入向量相同
        Eigen::VectorXcd result(input.size());

        // 执行FFT
        fftObj.fwd(result, input);

        return result;
    }
}

int main()
{
    // 创建一个示例向量
    Eigen::VectorXd vec(8);
    vec << 1, 2, 3, 4, 5, 6, 7, 8;

    // 调用fft函数
    Eigen::VectorXcd result = FFTLibrary::fft(vec);

    // 输出结果
    std::cout << "FFT result:" << std::endl;
    for (int i = 0; i < result.size(); ++i)
    {
        std::cout << result(i).real() << " + " << result(i).imag() << "i" << std::endl;
    }

    return 0;
}
#define __CL_ENABLE_EXCEPTIONS
#include <OpenCl/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

int main() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {
        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);
        int n, m;
        std::cin >> n >> m;

        std::vector<float> a(n * n);
        std::vector<float> b(m * m);
        for (int i = 0; i < n; i++) {
           for (int j = 0; j < n; j++) {
               std::cin >> a[i * n + j];
           }
        }

        for (int i = 0; i < m; i++) {
           for (int j = 0; j < m; j++) {
               std::cin >> b[i * m + j];
           }
        }

        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("convolution.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        try {
            program.build(devices);
        } catch (cl::Error const & e) {
            std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cout << log_str;
            return 0;
        }

        // create a message to send to kernel
        size_t const block_size = 16;

        // allocate device buffer to hold message
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(float) * n * n);
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(float) * m * m);
        cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * n * n);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * n * n, &a[0]);
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(float) * m * m, &b[0]);

        // load named kernel from opencl source
        cl::Kernel kernel_gmem(program, "convolution");
        kernel_gmem.setArg(0, dev_a);
        kernel_gmem.setArg(1, dev_b);
        kernel_gmem.setArg(2, dev_c);
        kernel_gmem.setArg(3, n);
        kernel_gmem.setArg(4, m);

        size_t pow = block_size;
        while (pow < n) {
            pow *= 2;
        }

        queue.enqueueNDRangeKernel(kernel_gmem, cl::NullRange,
                                   cl::NDRange(pow, pow), cl::NDRange(block_size, block_size));


        std::vector<float> c(n * n);
        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(float) * n * n, &c[0]);

        std::cout.precision(3);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << std::fixed << c[i * n + j] << " ";
            }

            std::cout << '\n';
        }
    } catch (cl::Error const & e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

   return 0;
}
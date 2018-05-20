#define __CL_ENABLE_EXCEPTIONS
#include <OpenCl/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

const size_t block_size = 256;

struct ProgramData {
    cl::Program& program;
    cl::Context& context;
    cl::CommandQueue& queue;

    ProgramData(cl::Program& p, cl::Context& c, cl::CommandQueue& q)
            : program(p)
            , context(c)
            , queue(q)

    {}
};

size_t get_blocks_number(size_t val) {
    double blocks = (double) val / block_size;

    size_t total_blocks = static_cast<size_t>(blocks);
    if (blocks > total_blocks) {
        total_blocks += 1;
    }

    return total_blocks;
}

void scan(ProgramData& data, cl::Buffer& in, cl::Buffer& out, size_t input_size) {
    size_t total_blocks = get_blocks_number(input_size);
    size_t array_size = total_blocks * block_size;

    cl::Buffer dev_partial_sums(data.context, CL_MEM_READ_WRITE,  sizeof(float) * total_blocks);

    cl::Kernel kernel_b(data.program, "scan_blelloch");
    cl::KernelFunctor scan_b(kernel_b, data.queue, cl::NullRange, cl::NDRange(array_size), cl::NDRange(block_size));

    cl::Event blelloch_event = scan_b(in, out, dev_partial_sums, cl::__local(sizeof(float) * block_size), input_size);
    blelloch_event.wait();

    if (total_blocks == 1) {
        return;
    }

    scan(data, dev_partial_sums, dev_partial_sums, total_blocks);

    cl::Kernel kernel_add(data.program, "add_rest_elements");
    cl::KernelFunctor add_elements(kernel_add, data.queue, cl::NullRange, cl::NDRange(array_size), cl::NDRange(block_size));

    cl::Event final_event = add_elements(dev_partial_sums, out, input_size);
    final_event.wait();
}


int main() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {
        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0]);

        // load opencl source
        std::ifstream cl_file("scan.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));

        // create program
        cl::Program program(context, source);
        // compile opencl source
        program.build(devices);

        freopen("input.txt", "r", stdin);
        size_t n;
        std::cin >> n;

        std::vector<float> a(n, 0), c(n);
        for (int i = 0; i < n; i++) {
            std::cin >> a[i];
        }

        // allocate device buffer to hold message
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY,  sizeof(float) * n);
        cl::Buffer dev_c(context, CL_MEM_READ_WRITE, sizeof(float) * n);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * n, &a[0]);
        ProgramData data(program, context, queue);
        scan(data, dev_a, dev_c, n);
        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(float) * n, &c[0]);

        for (size_t i = 0; i < n; i++) {
            std::cout << std::fixed << std::setprecision(3) << c[i] << " ";
        }
    } catch (cl::Error e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}
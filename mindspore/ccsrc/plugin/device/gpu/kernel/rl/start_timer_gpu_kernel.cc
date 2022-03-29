#include "plugin/device/gpu/kernel/rl/start_timer_gpu_kernel.h"
#include <ctime>

#include <memory>
#include <string>

#include "kernel/common_utils.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"


namespace mindspore {
namespace kernel {
    StartTimerKernel::StartTimerKernel(): input_size_(0){};
    StartTimerKernel::~StartTimerKernel() {};
    void StartTimerKernel::InitResource() {};
    void StartTimerKernel::InitSizeLists() {};

    size_t get_size(const std::vector<size_t> &input_shape)
    {
        size_t input_count_ = std::accumulate(input_shape.begin(), input_shape.end(),
                                       1, std::multiplies<size_t>());
        return input_count_ * sizeof(float);
    }

    void get_time(const void* input, void* output, const size_t input_size_, cudaStream_t stream)
    {
        clock_t curr_time;
        curr_time = clock();
        const auto dir = cudaMemcpyHostToDevice;
        //MS_LOG(ERROR) << "time"<<curr_time;
        //cudaMemcpyAsync(output, input, size, dir, stream);
        float st = (float)curr_time;
        auto res0 = cudaMemcpy(output, &st, sizeof(float), dir);
        //cudaMemcpy(output, input, size, dir);
        if(res0 == cudaSuccess)
        {
            //MS_LOG(ERROR) << "cudaMemcpy success ";
            return;
        }
        else
        {
            MS_LOG(ERROR) << "cudaMemcpy failed";
            return;
        }
    }

    bool StartTimerKernel::Init(const CNodePtr &kernel_node)
    {
        //size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
        //size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);

        //MS_LOG(ERROR) << "**************** init ************ "<<input_num<<" "<<output_num;

        auto inputA_shape =
            common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
        input_size_ = get_size(inputA_shape);
        input_size_list_.push_back(input_size_);
        input_size_list_.push_back(sizeof(float));
        output_size_list_.push_back(input_size_);
        //MS_LOG(ERROR) << "**************** done init ************ "<<input_num<<" "<<output_num<<" "<<input_size_;
        return true;
    }

    bool StartTimerKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr)
    {

        //MS_LOG(ERROR) << "**************** launch call ************ "<<inputs.size()<<" "<<outputs.size();
        float *input_addr = GetDeviceAddress<float>(inputs, 0);
        float *output_addr = GetDeviceAddress<float>(outputs, 0);

        cudaStream_t stream =  reinterpret_cast<cudaStream_t>(stream_ptr);

        get_time(input_addr, output_addr, input_size_, stream);
        return true;
    }

}
}


#include "plugin/device/gpu/kernel/rl/elap_timer_gpu_kernel.h"
#include <ctime>

#include <memory>
#include <string>

#include "kernel/common_utils.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"


namespace mindspore {
namespace kernel {

    ElapTimerKernel::ElapTimerKernel(): input_size_(0){};
    ElapTimerKernel::~ElapTimerKernel() {};
    void ElapTimerKernel::InitResource() {};
    void ElapTimerKernel::InitSizeLists() {};

    size_t get_input_size(const std::vector<size_t> &input_shape)
    {
        size_t input_count_ = std::accumulate(input_shape.begin(), input_shape.end(),
                                       1, std::multiplies<size_t>());
        return input_count_ * sizeof(float);
    }

    void get_elap_time(const void* input, void* output, const size_t input_size_, cudaStream_t stream)
    {
        clock_t curr_time;
        float st, et, elap_time = 0;
        const auto dir = cudaMemcpyHostToDevice;
        const auto dir2 = cudaMemcpyDeviceToHost;
        
        //cudaMemcpyAsync(output, input, size, dir, stream);
        auto res0 = cudaMemcpy(&st, input, sizeof(float), dir2);
        //MS_LOG(ERROR)<<"st time "<<st;
        curr_time = clock();
        et = (float)curr_time;
        //MS_LOG(ERROR)<<"et time "<<curr_time<<" "<<et;
        elap_time = (float)(et-st)/CLOCKS_PER_SEC;
        //MS_LOG(ERROR)<<"elap time "<<elap_time;
        //cudaMemcpy(output, input, size, dir);
        res0 = cudaMemcpy(output, &elap_time, sizeof(float), dir);
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

    bool ElapTimerKernel::Init(const CNodePtr &kernel_node)
    {
        //size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
        //size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);

        //MS_LOG(ERROR) << "**************** init ************ "<<input_num<<" "<<output_num;

        auto inputA_shape =
            common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
        input_size_ = get_input_size(inputA_shape);
        input_size_list_.push_back(input_size_);
        output_size_ = sizeof(float);
        output_size_list_.push_back(output_size_);
        //MS_LOG(ERROR) << "**************** done init ************ "<<input_num<<" "<<output_num<<" "<<input_size_<<" "<<output_size_;
        return true;
    }

    bool ElapTimerKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr)
    {
        //MS_LOG(ERROR) << "**************** launch call ************ "<<inputs.size()<<" "<<outputs.size();
        float *input_addr = GetDeviceAddress<float>(inputs, 0);
        float *output_addr = GetDeviceAddress<float>(outputs, 0);

        cudaStream_t stream =  reinterpret_cast<cudaStream_t>(stream_ptr);

        get_elap_time(input_addr, output_addr, input_size_, stream);
        return true;
    }

    MS_REG_GPU_KERNEL(ElapTimer, ElapTimerKernel)
    //MS_REG_GPU_KERNEL_TWO(ElapTimer, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeFloat32),
    //                  ElapTimerKernel, float, float)
}
}


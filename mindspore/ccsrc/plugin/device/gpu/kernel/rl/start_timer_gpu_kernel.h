#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_START_TIMER_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_START_TIMER_KERNEL_H_

#include <string>
#include <vector>

//#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore
{
namespace kernel
{
class StartTimerKernel : public NativeGpuKernelMod 
{
    public:
      StartTimerKernel();
      ~StartTimerKernel();
      //void rl_log_tensor(int idx, const void *send_buff, const void *recv_buff);
      bool Init(const CNodePtr &kernel_node) override;
      bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
    protected:
      void InitResource() override;
      void InitSizeLists() override;
    
    private:
      //size_t input_count_;
      //size_t output_count_;

      //std::vector<size_t> input_size_list_;
      //std::vector<size_t> output_size_list_;
      //std::vector<size_t> workspace_size_list_;
      size_t input_size_;
};
    MS_REG_GPU_KERNEL(StartTimer, StartTimerKernel)
    //MS_REG_GPU_KERNEL(StartTimer,
    //        KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), StartTimerKernel, float)

}  // namespace kernel
}  // namespace   mindspore
#endif 

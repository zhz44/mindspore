/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_MPI_COMMUNICATION_GROUP_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_MPI_COMMUNICATION_GROUP_H_

#include <mpi.h>
#include <string>
#include <vector>
#include "runtime/hardware/communication_group.h"

namespace mindspore {
namespace device {
namespace cpu {
class MPICommunicationGroup : public CommunicationGroup {
 public:
  explicit MPICommunicationGroup(uint32_t size, const std::string name, const std::vector<uint32_t> &group_ranks)
      : CommunicationGroup(size, name, group_ranks) {}

  ~MPICommunicationGroup() = default;

  void Initialize() override;
  void Finalize() override;

 private:
  MPI_Group mpi_group_;
};
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_MPI_COMMUNICATION_GROUP_H_
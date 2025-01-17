# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""vmap impl."""
from .vmap_base import get_vmap_rule, match_out_axis, bind_in_axes, vmap_general_rule, vmap_monad_rule

__all__ = ['get_vmap_rule', 'match_out_axis', 'bind_in_axes', 'vmap_general_rule', 'vmap_monad_rule']

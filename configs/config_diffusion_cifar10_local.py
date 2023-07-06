# coding=utf-8
# Copyright 2022 The Pix2Seq Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A config."""

# pylint: disable=invalid-name,line-too-long

from configs import config_diffusion_cifar10 as config_base


def get_config(config_str=None):
    """Returns config."""
    del config_str
    config = config_base.get_config()
    config.train.batch_size = 64
    config.train.checkpoint_steps = 1

    return config


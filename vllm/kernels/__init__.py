# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel implementations for vLLM."""

import vllm.ir.ops  # ensure all IrOp objects exist before backends register impls
from vllm.platforms import current_platform

# Delegate entirely to the platform. The base Platform.import_kernels() loads
# C extensions and the built-in vllm_c/aiter_ops registrations. OOT platforms
# override import_kernels() and import only their own kernel registration modules.
current_platform.import_kernels()

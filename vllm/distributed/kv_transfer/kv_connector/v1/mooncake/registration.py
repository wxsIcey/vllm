# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from operator import index
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.platforms.interface import Platform


@dataclass(frozen=True, order=True)
class MemoryRegion:
    address: int
    size_bytes: int

    @property
    def end(self) -> int:
        return self.address + self.size_bytes


def _storage_region(cache: torch.Tensor) -> MemoryRegion | None:
    if cache.layout != torch.strided:
        raise RuntimeError(
            "Mooncake KV cache registration requires strided tensors, "
            f"got layout={cache.layout}."
        )

    storage = cache.untyped_storage()
    address = storage.data_ptr()
    size_bytes = storage.nbytes()
    if size_bytes == 0:
        return None
    if address <= 0:
        raise RuntimeError(
            "Mooncake cannot register a KV cache storage with an invalid "
            f"base address: {address}."
        )

    if cache.numel() > 0:
        element_size = cache.element_size()
        storage_offset = cache.storage_offset()
        minimum_offset = storage_offset
        maximum_offset = storage_offset
        for dimension, stride in zip(cache.shape, cache.stride()):
            delta = (dimension - 1) * stride
            minimum_offset += min(delta, 0)
            maximum_offset += max(delta, 0)

        if minimum_offset < 0 or (maximum_offset + 1) * element_size > size_bytes:
            raise RuntimeError(
                "Mooncake KV cache view lies outside its backing storage: "
                f"shape={tuple(cache.shape)}, stride={tuple(cache.stride())}, "
                f"storage_offset={storage_offset}, storage_bytes={size_bytes}."
            )

        expected_address = address + storage_offset * element_size
        if cache.data_ptr() != expected_address:
            raise RuntimeError(
                "Mooncake KV cache view address does not match its backing "
                f"storage: view={cache.data_ptr()}, expected={expected_address}."
            )

    return MemoryRegion(address, size_bytes)


def _validate_regions(
    ranges: Iterable[tuple[int, int]], stage: str
) -> tuple[MemoryRegion, ...]:
    try:
        regions = tuple(
            MemoryRegion(index(address), index(size)) for address, size in ranges
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Mooncake platform returned invalid registration ranges during {stage}."
        ) from exc

    if any(region.address <= 0 or region.size_bytes <= 0 for region in regions):
        raise RuntimeError(
            "Mooncake platform returned a non-positive registration range "
            f"during {stage}."
        )
    return regions


def _merge_overlapping(
    regions: tuple[MemoryRegion, ...],
) -> tuple[MemoryRegion, ...]:
    if not regions:
        return ()

    ordered = sorted(regions)
    merged: list[MemoryRegion] = [ordered[0]]
    for region in ordered[1:]:
        previous = merged[-1]
        if region.address < previous.end:
            merged[-1] = MemoryRegion(
                previous.address, max(previous.end, region.end) - previous.address
            )
        else:
            merged.append(region)
    return tuple(merged)


def _validate_coverage(
    storage_regions: tuple[MemoryRegion, ...],
    registration_regions: tuple[MemoryRegion, ...],
    stage: str,
) -> None:
    for storage in storage_regions:
        if not any(
            region.address <= storage.address and storage.end <= region.end
            for region in registration_regions
        ):
            raise RuntimeError(
                "Mooncake platform registration ranges do not cover a complete "
                f"KV cache storage during {stage}: storage=({storage.address:#x}, "
                f"{storage.size_bytes} bytes)."
            )

    for region in registration_regions:
        if not any(
            region.address < storage.end and storage.address < region.end
            for storage in storage_regions
        ):
            raise RuntimeError(
                "Mooncake platform returned a registration range unrelated to "
                f"KV cache storage during {stage}: ({region.address:#x}, "
                f"{region.size_bytes} bytes)."
            )


def plan_kv_cache_memory_regions(
    kv_caches: Iterable[torch.Tensor], platform: Platform
) -> tuple[MemoryRegion, ...]:
    """Plan safe, platform-compatible ranges for Mooncake registration."""
    storage_regions = tuple(
        sorted(
            {
                region
                for cache in kv_caches
                if (region := _storage_region(cache)) is not None
            }
        )
    )
    if not storage_regions:
        return ()

    storage_ranges = tuple(
        (region.address, region.size_bytes) for region in storage_regions
    )
    recovered_ranges = _validate_regions(
        platform.get_mooncake_memory_registration_regions(storage_ranges),
        "address recovery",
    )
    _validate_coverage(storage_regions, recovered_ranges, "address recovery")
    coalesced_ranges = _validate_regions(
        platform.coalesce_mooncake_memory_registration_regions(
            tuple((region.address, region.size_bytes) for region in recovered_ranges)
        ),
        "coalescing",
    )
    registration_regions = _merge_overlapping(coalesced_ranges)
    _validate_coverage(storage_regions, registration_regions, "coalescing")

    region_limit = platform.get_mooncake_memory_registration_region_limit()
    if region_limit is not None:
        try:
            region_limit = index(region_limit)
        except TypeError as exc:
            raise RuntimeError(
                "Mooncake platform returned a non-integer registration "
                f"region limit: {region_limit!r}."
            ) from exc
        if region_limit <= 0:
            raise RuntimeError(
                "Mooncake platform returned an invalid registration region "
                f"limit: {region_limit}."
            )
        if len(registration_regions) > region_limit:
            raise RuntimeError(
                "Mooncake KV cache requires "
                f"{len(registration_regions)} registration regions, exceeding "
                f"the {platform.device_name} platform limit of {region_limit}. "
                "Use a cache layout with fewer physical allocations."
            )

    return registration_regions

/*!
 * Copyright 2017-2025 by Contributors
 * \file device_properties.h
 */
#ifndef PLUGIN_SYCL_DEVICE_PROPERTIES_H_
#define PLUGIN_SYCL_DEVICE_PROPERTIES_H_

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include "../../src/common/common.h"               // for HumanMemUnit

namespace xgboost {
namespace sycl {

class DeviceProperties {
  void GetL2Size(const ::sycl::device& device) {
    l2_size = device.get_info<::sycl::info::device::global_mem_cache_size>();
    LOG(INFO) << "Detected L2 Size = " << ::xgboost::common::HumanMemUnit(l2_size);
    l2_size_per_eu = static_cast<float>(l2_size) / max_compute_units;
  }

  void GetSRAMSize(const ::sycl::device& device) {
    auto arch =
      device.get_info<::sycl::ext::oneapi::experimental::info::device::architecture>();
    switch (arch) {
      case ::sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc: {
        LOG(INFO) << "Xe-HPC (Ponte Vecchio) Architecture. L1 friendly optimization enabled.";
        l1_size = 512 * 1024;
        size_t registers_size = 64 * 1024;
        sram_size_per_eu = l1_size  / eu_per_core + registers_size;
        break;
      }
      default:
        sram_size_per_eu = 0;
        l1_size = 64 * 1024; //device.get_info<::sycl::info::device::local_mem_size>();
    }
    LOG(INFO) << "Detected L1 Size = " << ::xgboost::common::HumanMemUnit(l1_size);
  }

 public:
  bool is_gpu;
  bool usm_host_allocations;
  bool host_unified_memory;
  size_t max_compute_units;
  size_t max_work_group_size;
  size_t min_sub_group_size;
  size_t max_sub_group_size;
  size_t eu_per_core;
  size_t n_cores;
  size_t local_mem_size = 0;
  float sram_size_per_eu = 0;
  size_t l2_size = 0;
  size_t l1_size = 0;
  float l2_size_per_eu = 0;

  DeviceProperties():
    is_gpu(false) {}

  explicit DeviceProperties(const ::sycl::device& device):
    is_gpu(device.is_gpu()),
    usm_host_allocations(device.has(::sycl::aspect::usm_host_allocations)),
    host_unified_memory(device.get_info<::sycl::info::device::host_unified_memory>()),
    max_compute_units(device.get_info<::sycl::info::device::max_compute_units>()),
    max_work_group_size(1024),
    min_sub_group_size(device.get_info<::sycl::info::device::sub_group_sizes>().front()),
    max_sub_group_size(device.get_info<::sycl::info::device::sub_group_sizes>().back()),
    eu_per_core(device.get_info<::sycl::ext::intel::info::device::gpu_eu_count_per_subslice>()),
    n_cores(max_compute_units / eu_per_core),
    local_mem_size(device.get_info<::sycl::info::device::local_mem_size>()) {
      LOG(INFO) << "Detected " << n_cores << " cores";
      LOG(INFO) << "Detected " << max_compute_units << " EUs";
      LOG(INFO) << "Detected max_work_group_size = " << max_work_group_size;
      LOG(INFO) << "Detected min_sub_group_size = " << min_sub_group_size;
      LOG(INFO) << "Detected max_sub_group_size = " << max_sub_group_size;
      LOG(INFO) << "Detected usm_host_allocations = " << usm_host_allocations;
      LOG(INFO) << "Detected host_unified_memory = " << host_unified_memory;
      LOG(INFO) << "Detected local_mem_size = "
                << ::xgboost::common::HumanMemUnit(local_mem_size);
      GetL2Size(device);
      if (is_gpu) {
        GetSRAMSize(device);
      }
    }
};

}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_DEVICE_PROPERTIES_H_

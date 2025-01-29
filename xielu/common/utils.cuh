#ifndef UTILS_CUH
#define UTILS_CUH

#include <nvtx3/nvToolsExt.h>
#include <torch/extension.h>

#ifdef __CUDA_ARCH__
#define DEVICE __device__
#else
#define DEVICE
#endif

template <class T>
DEVICE inline T *shared_array(unsigned int n_elements, void *&ptr,
                              unsigned int *space) noexcept {
  const unsigned long long inptr = reinterpret_cast<unsigned long long>(ptr);
  const unsigned long long end = inptr + n_elements * sizeof(T);
  if (space)
    *space += static_cast<unsigned int>(end - inptr);
  ptr = reinterpret_cast<void *>(end);
  return reinterpret_cast<T *>(inptr);
}

template <typename scalar_t, int num_dims>
using Accessor =
    torch::PackedTensorAccessor32<scalar_t, num_dims, torch::RestrictPtrTraits>;

template <typename scalar_t, int num_dims>
inline Accessor<scalar_t, num_dims> get_accessor(const torch::Tensor &tensor) {
  return tensor
      .packed_accessor32<scalar_t, num_dims, torch::RestrictPtrTraits>();
};

const uint32_t colors[] = {0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff,
                           0xff00ffff, 0xffff0000, 0xffffffff};
const int num_colors = sizeof(colors) / sizeof(uint32_t);

#define PUSH_RANGE(name, cid)                                                  \
  {                                                                            \
    int color_id = cid;                                                        \
    color_id = color_id % num_colors;                                          \
    nvtxEventAttributes_t eventAttrib = {0};                                   \
    eventAttrib.version = NVTX_VERSION;                                        \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                          \
    eventAttrib.colorType = NVTX_COLOR_ARGB;                                   \
    eventAttrib.color = colors[color_id];                                      \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                         \
    eventAttrib.message.ascii = name;                                          \
    nvtxRangePushEx(&eventAttrib);                                             \
  }

#define POP_RANGE nvtxRangePop();

#endif // UTILS_HPP
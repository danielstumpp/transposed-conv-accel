

namespace CFG
{
    // note: in_size + 2pad - k must be multiple of stride

    constexpr int in_channels = 1;
    constexpr int out_channels = 2;
    constexpr int in_size = 5;
    constexpr int kernel_size = 4;
    constexpr int stride = 3;
    constexpr int pad = 1;
    constexpr int test = in_size + 2*pad - kernel_size;

    constexpr int out_size = stride*(in_size - 1) + kernel_size - 2*pad;
} // namespace CFG


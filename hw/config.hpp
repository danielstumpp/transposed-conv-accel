

namespace CFG
{
    constexpr int in_channels = 1;
    constexpr int out_channels = 1;
    constexpr int in_size = 3;
    constexpr int kernel_size = 3;
    constexpr int stride = 2;
    constexpr int pad = 1;

    constexpr int out_size = stride*(in_size - 1) + kernel_size - 2*pad;
} // namespace CFG

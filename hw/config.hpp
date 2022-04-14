

namespace CFG
{
    // note: in_size + 2pad - k must be multiple of stride
    // i.e. a = 0 must hold for the current implementation

    constexpr int in_channels = 1;
    constexpr int out_channels = 1;
    constexpr int in_size = 4;
    constexpr int kernel_size = 4;
    constexpr int stride = 3;
    constexpr int pad = 3;
    constexpr int a = (in_size + 2*pad - kernel_size) % stride;
    constexpr int test = in_size + 2 *pad - kernel_size;

    constexpr int out_size = stride * (in_size - 1) + kernel_size - 2 * pad;

} // namespace CFG


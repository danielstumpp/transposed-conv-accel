#pragma once

namespace CFG
{
    // note: in_size + 2pad - k must be multiple of stride
    // i.e. a = 0 must hold for the current implementation

    const int in_channels = 3;
    const int out_channels = 2;
    const int in_size = 4;
    const int kernel_size = 4;
    const int stride = 3;
    const int pad = 3;
    const int a = (in_size + 2*pad - kernel_size) % stride;
    const int test = in_size + 2 *pad - kernel_size;

    const int out_size = stride * (in_size - 1) + kernel_size - 2 * pad;

} // namespace CFG



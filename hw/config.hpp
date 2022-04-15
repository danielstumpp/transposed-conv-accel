#pragma once

namespace CFG
{
    // note: in_size + 2pad - k must be multiple of stride
    // i.e. a = 0 must hold for the current implementation

    const int in_channels = 256;
    const int out_channels = 128;
    const int in_size = 100;
    const int kernel_size = 2;
    const int stride = 2;
    const int pad = 0;
    const int a = (in_size + 2*pad - kernel_size) % stride;
    const int test = in_size + 2 *pad - kernel_size;

    const int out_size = stride * (in_size - 1) + kernel_size - 2 * pad;

    const int cTile = 2;
} // namespace CFG



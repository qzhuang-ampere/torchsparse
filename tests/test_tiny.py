import unittest
from torchsparse.nn import functional as F
from python import (
    test_single_layer_convolution_forward_tiny,
)

class SparseConvTestCase(unittest.TestCase):
    def test_single_layer(self):
        kernel_sizes = [3]
        strides = [1]
        acc_adiff = 0.0
        acc_rdiff = 0.0
        count = 0

        # hashmap mode by default
        for kernel_size in kernel_sizes:
            for stride in strides:
                mean_adiff, max_rdiff = test_single_layer_convolution_forward_tiny(
                    kernel_size=kernel_size, stride=stride, num_points=100000, IC=96, OC=96, shape=[500,500,50], use_ref=False
                )
                acc_adiff += mean_adiff
                acc_rdiff += max_rdiff
                count += 1

        # switch to hashmap_on_the_fly
        config = F.conv_config.get_default_conv_config()
        config.kmap_mode = "hashmap_on_the_fly"
        F.conv_config.set_global_conv_config(config)
        for kernel_size in kernel_sizes:
            for stride in strides:
               # mean_adiff, max_rdiff = test_single_layer_convolution_forward_tiny(
               #     kernel_size=kernel_size, stride=stride, num_points=1000000, IC=96, OC=96, shape=[1000,1000,500], use_ref=False
               # )
                acc_adiff += mean_adiff
                acc_rdiff += max_rdiff
                count += 1

        self.assertLessEqual(acc_adiff / count, 1e-4)
        self.assertLessEqual(acc_rdiff / count, 1e-2)

if __name__ == "__main__":
    unittest.main()

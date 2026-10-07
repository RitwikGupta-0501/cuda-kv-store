import torch
import warpkv_torch
import unittest

class TestWarpKVTorch(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.engine = warpkv_torch.Engine(4096)
        
    def test_zero_copy_insert_lookup(self):
        # 1024 keys
        keys = torch.arange(1, 1025, dtype=torch.int64, device='cuda')
        values = keys * 10
        
        # Zero-copy insert
        self.engine.insert_batch(keys, values)
        
        # Zero-copy lookup
        result = self.engine.lookup_batch(keys)
        
        # Validate
        self.assertTrue(torch.equal(result, values))
        
    def test_missing_keys(self):
        keys = torch.arange(1, 100, dtype=torch.int64, device='cuda')
        values = keys * 10
        self.engine.insert_batch(keys, values)
        
        # Look up non-existent keys
        missing_keys = torch.arange(1000, 1100, dtype=torch.int64, device='cuda')
        result = self.engine.lookup_batch(missing_keys)
        
        # NOT_FOUND is 0xFFFFFFFFFFFFFFFF, which is -1 in signed int64
        self.assertTrue(torch.all(result == -1))
        
    def test_zero_copy_delete(self):
        keys = torch.arange(1, 100, dtype=torch.int64, device='cuda')
        values = keys * 10
        self.engine.insert_batch(keys, values)
        
        self.engine.delete_batch(keys)
        
        result = self.engine.lookup_batch(keys)
        self.assertTrue(torch.all(result == -1))

if __name__ == '__main__':
    unittest.main()

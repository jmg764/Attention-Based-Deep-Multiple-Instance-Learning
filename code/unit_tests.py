import unittest
import torch 
import model_def 

class RunTests(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super(RunTests, self).__init__()

  # Ensure shape of dataset is correct
  def test_equal(self, train_loader):
    data, _ = iter(train_loader).next()
    self.assertEqual(torch.Size((16, 3, 128, 128)), data.shape)
  
  # Testing if dataset works with the dataloader
  def test_single_process_dataloader(self, data):
    with self.subTest(split='train'):
        self._check_dataloader(data, shuffle=True, num_workers=0)
    with self.subTest(split='test'):
        self._check_dataloader(data, shuffle=False, num_workers=0)

  def _check_dataloader(self, data, shuffle, num_workers):
    loader = torch.data_utils.DataLoader(data, batch_size=1, shuffle=shuffle, num_workers=num_workers)
    for _ in loader:
        pass

  # Ensure that there aren't any dead sub-graphs (i.e. any learnable parameters that aren't used 
  # in the forward pass, backward pass, or both)
  def test_all_parameters_updated(self):
    net = model_def.Attention()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0005)

    loss, _ = net.calculate_objective(torch.randn(16, 3, 128, 128), torch.tensor([[1]]))
    
    loss.backward()
    optim.step()
    
    for param_name, param in net.named_parameters():
        if param.requires_grad:
            with self.subTest(name=param_name):
                self.assertIsNotNone(param.grad)
                self.assertNotEqual(0., torch.sum(param.grad ** 2))

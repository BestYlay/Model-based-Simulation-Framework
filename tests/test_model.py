import unittest

import torch

from survival_simulation.model import SurvivalTransformer, cox_ph_loss
from survival_simulation.training import resolve_device


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = SurvivalTransformer(
            vocab_sizes={"kind": 4},
            embedding_dims={"kind": 3},
            num_numerical_features=2,
            d_model=8,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=16,
            dropout=0.0,
        ).eval()

    def test_forward_shape_and_padding_invariance(self):
        numerical = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]], [[5.0, 6.0], [0.0, 0.0], [0.0, 0.0]]]
        )
        categorical = {"kind": torch.tensor([[1, 2, 0], [3, 0, 0]])}
        mask = torch.tensor([[False, False, True], [False, True, True]])

        first = self.model(numerical, categorical, mask)
        changed_padding = numerical.clone()
        changed_padding[mask] = 999.0
        second = self.model(changed_padding, categorical, mask)

        self.assertEqual(tuple(first.shape), (2,))
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_cox_loss_is_finite_for_extreme_scores(self):
        scores = torch.tensor([1000.0, 0.0, -1000.0], requires_grad=True)
        times = torch.tensor([1.0, 2.0, 3.0])
        events = torch.tensor([1.0, 1.0, 1.0])

        loss = cox_ph_loss(scores, times, events)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_higher_risk_for_earlier_events_has_lower_loss(self):
        times = torch.tensor([1.0, 2.0, 3.0])
        events = torch.ones(3)
        aligned = cox_ph_loss(torch.tensor([2.0, 1.0, 0.0]), times, events)
        reversed_order = cox_ph_loss(torch.tensor([0.0, 1.0, 2.0]), times, events)
        self.assertLess(aligned.item(), reversed_order.item())

    def test_auto_device_is_available(self):
        device = resolve_device("auto")
        self.assertIn(device.type, {"cpu", "cuda"})


if __name__ == "__main__":
    unittest.main()

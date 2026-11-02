import unittest
from pathlib import Path

import torch

from gamefeed.models import SharedBottomRanker, TwoTower, grouped_recall_loss, multitask_loss


class ModelTests(unittest.TestCase):
    def test_shared_bottom_shapes_and_loss(self) -> None:
        model = SharedBottomRanker(input_dim=12)
        outputs = model(torch.randn(5, 12))
        self.assertEqual(outputs["long_view"].shape, (5,))
        loss = multitask_loss(outputs, torch.tensor([0, 1, 0, 1, 1]), torch.tensor([0, 0, 0, 1, 0]))
        self.assertTrue(torch.isfinite(loss))

    def test_grouped_loss_masks_padding(self) -> None:
        logits = torch.tensor([[2.0, 1.0, 100.0]])
        mask = torch.tensor([[True, True, False]])
        expected = torch.nn.functional.cross_entropy(torch.tensor([[2.0, 1.0]]) / 0.1, torch.tensor([0]))
        self.assertAlmostEqual(grouped_recall_loss(logits, mask, 0.1).item(), expected.item(), places=6)

    def test_save_load_is_stable_in_eval_mode(self) -> None:
        torch.manual_seed(1)
        model = TwoTower(num_users=5, num_items=8, num_tags=4, dropout=0.2)
        model.eval()
        inputs = (
            torch.tensor([2]),
            torch.tensor([[2, 3, 0]]),
            torch.tensor([[[2, 3], [2, 0], [0, 0]]]),
            torch.tensor([[0.1, 0.2, 0.0]]),
            torch.tensor([[True, True, False]]),
            torch.tensor([0.5]),
        )
        with torch.no_grad():
            before = model.encode_user(*inputs)
        path = Path("artifacts") / "test_model_state.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(model.state_dict(), path)
            restored = TwoTower(num_users=5, num_items=8, num_tags=4, dropout=0.2)
            restored.load_state_dict(torch.load(path, weights_only=True))
            restored.eval()
            with torch.no_grad():
                after = restored.encode_user(*inputs)
        finally:
            path.unlink(missing_ok=True)
        self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()


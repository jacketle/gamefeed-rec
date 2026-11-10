import unittest
import torch
from gamefeed.sequence_model import SequenceRanker


class SequenceModelTests(unittest.TestCase):
    def test_author_zero_fallback_gradient_and_reload(self):
        import io
        model = SequenceRanker(2,8,8,2,'B0',num_authors=4)
        batch = self.batch(); batch['author'] = torch.tensor([0,2])
        model(batch).sum().backward()
        self.assertEqual(model.author.weight.grad[0].abs().sum().item(),0)
        self.assertGreater(model.author.weight.grad[2].abs().sum().item(),0)
        model.eval(); expected = model(batch).detach()
        with torch.no_grad(): model.author.weight[0].fill_(100)
        self.assertTrue(torch.equal(model(batch).detach(),expected))
        buffer = io.BytesIO(); torch.save(model.state_dict(),buffer); buffer.seek(0)
        restored = SequenceRanker(2,8,8,2,'B0',num_authors=4).eval()
        restored.load_state_dict(torch.load(buffer,weights_only=True))
        self.assertTrue(torch.equal(restored(batch).detach(),expected))

    def test_item_l2_only_penalizes_distinct_used_ids(self):
        model=SequenceRanker(2,8,8,2,'B1')
        b=self.batch(); b['history'][0,0]=4
        b['history'][1,0]=7  # Invalid position must not participate.
        penalty=model.item_l2(b)
        self.assertTrue(torch.allclose(penalty,model.item.weight[[2,3,4]].square().mean()))
        penalty.backward()
        self.assertEqual(model.item.weight.grad[[0,1,5,6,7]].abs().sum().item(),0)
        self.assertGreater(model.item.weight.grad[4].abs().sum().item(),0)
        model.mode='B0'
        self.assertTrue(torch.allclose(model.item_l2(b),model.item.weight[[2,3]].square().mean()))
        b['item'][:]=1
        self.assertEqual(model.item_l2(b).item(),0)

    def test_paired_seeds_share_initial_common_weights(self):
        torch.manual_seed(2026)
        mean=SequenceRanker(2,8,8,2,'B1')
        torch.manual_seed(2026)
        attention=SequenceRanker(2,8,8,2,'B2')
        for name,p in mean.named_parameters():
            self.assertTrue(torch.equal(p,dict(attention.named_parameters())[name]),name)

    def batch(self):
        return dict(item=torch.tensor([2,3]),tags=torch.tensor([[2,0],[3,0]]),
            history=torch.tensor([[2,0],[0,0]]),history_tags=torch.tensor([[[2,0],[0,0]],[[0,0],[0,0]]]),
            mask=torch.tensor([[True,False],[False,False]]),user=torch.tensor([1,1]),
            tab=torch.tensor([1,1]),numeric=torch.zeros(2,4))

    def test_empty_history_is_finite_and_padding_ignored(self):
        for mode in ['B0','B1','B2']:
            model=SequenceRanker(2,8,8,2,mode)
            b=self.batch(); y=model(b)
            self.assertTrue(torch.isfinite(y).all())
            self.assertTrue(model.trained_items[2])
            self.assertFalse(model.trained_items[0])
            y.sum().backward()
            model.eval(); a=model(b)
            b['history'][~b['mask']]=7
            self.assertTrue(torch.allclose(a,model(b)))

    def test_untrained_id_uses_same_zero_fallback(self):
        model=SequenceRanker(2,8,8,2,'B2').eval()
        b=self.batch(); a=model(b)
        b['item']=torch.tensor([6,7])
        self.assertTrue(torch.allclose(a,model(b)))

    def test_history_id_mask_preserves_candidate_tags_and_positions(self):
        model=SequenceRanker(2,8,8,2,'B1',history_id_dropout=1.0)
        b=self.batch(); b['history'][0,0]=7
        calls=[]
        original=model.encode
        def capture(ids,tags):
            calls.append((ids.clone(),tags.clone()))
            return original(ids,tags)
        model.encode=capture
        prediction=model(b)
        self.assertTrue(torch.equal(calls[0][0],b['item']))
        self.assertTrue(torch.equal(calls[1][1],b['history_tags']))
        self.assertEqual(calls[1][0][0,0].item(),1)
        self.assertEqual(b['history'][0,0].item(),7)
        self.assertTrue(b['mask'][0,0])
        self.assertFalse(model.trained_items[7])
        prediction.sum().backward()
        self.assertEqual(model.item.weight.grad[7].abs().sum().item(),0)
        model.eval(); model.trained_items[7]=True; calls.clear()
        model(b)
        self.assertEqual(calls[1][0][0,0].item(),7)

    def test_mask_probability_zero_preserves_output_and_initialization(self):
        torch.manual_seed(2026); a=SequenceRanker(2,8,8,2,'B1')
        torch.manual_seed(2026); b=SequenceRanker(2,8,8,2,'B1',history_id_dropout=.5)
        b.history_id_dropout=0
        self.assertTrue(torch.equal(a(self.batch()),b(self.batch())))

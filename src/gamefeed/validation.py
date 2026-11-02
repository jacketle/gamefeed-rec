"""Small-data fit check and a separately trained no-history ablation."""
from dataclasses import replace
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from .datasets import RecallCollator
from .features import FeatureSpace
from .training import _build_model, _batch_loss, train_two_tower
from .utils import seed_everything, write_json


def small_fit(config):
    seed_everything(config['seed'])
    paths = config['resolved_paths']
    groups = pd.read_parquet(paths.processed_dir / 'recall_groups.parquet')
    sample = groups[groups.split.eq('train')].sample(100, random_state=config['seed'])
    events = pd.read_parquet(paths.processed_dir / 'events.parquet')
    items = pd.read_parquet(paths.processed_dir / 'items.parquet')
    features = FeatureSpace.fit(events[events.split.eq('train')], items)
    batch = RecallCollator(features, items, 20)(sample.to_dict('records'))
    model = _build_model(config, features)
    # Disable dropout for this deterministic memorization diagnostic only.
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    initial = float(_batch_loss(model, batch, .1).detach())
    for _ in range(150):
        optimizer.zero_grad()
        loss = _batch_loss(model, batch, .1)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        users = model.encode_user(*(batch[k] for k in ('user_ids','history_item_ids','history_tag_ids','history_durations','history_mask','history_len_scaled')))
        items_encoded = model.item_encoder(*(batch[k] for k in ('candidate_item_ids','candidate_tag_ids','candidate_durations')))
        scores = torch.einsum('bd,bcd->bc', users, items_encoded).masked_fill(~batch['candidate_mask'], -torch.inf)
        accuracy = float(scores.argmax(1).eq(0).float().mean())
        final = float(_batch_loss(model, batch, .1))
    result = dict(samples=100, steps=150, initial_loss=initial, final_loss=final,
                  positive_top1_accuracy=accuracy, passed=accuracy >= .95 and final < initial * .1)
    write_json(paths.report_dir / 'small_fit.json', result)
    return result


def run_validation(config):
    result = {'small_fit': small_fit(config)}
    if not result['small_fit']['passed']:
        raise RuntimeError('Small fit failed; inspect small_fit.json before continuing')
    for mode in ('with_history', 'without_history'):
        local = deepcopy(config)
        local['resolved_paths'] = replace(config['resolved_paths'],
            report_dir=config['resolved_paths'].report_dir / mode,
            artifact_dir=config['resolved_paths'].artifact_dir / mode)
        local['resolved_paths'].report_dir.mkdir(parents=True, exist_ok=True)
        local['ablation_no_history'] = mode == 'without_history'
        local['evaluation_splits'] = ['dev']
        result[mode] = train_two_tower(local)
    write_json(config['resolved_paths'].report_dir / 'validation.json', result)
    return result

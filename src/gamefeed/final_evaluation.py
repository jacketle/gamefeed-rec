"""Explain saved fusion results and evaluate the final ten recommendations."""
from collections import Counter
import numpy as np
import pandas as pd
import torch

from .datasets import ExposureCollator
from .demo import _load_ranker, author_capped_rerank, author_key, caption_key
from .features import FeatureSpace
from .training import _ranker_forward
from .utils import write_json


def recall(ids, positives, k):
    return len(set(ids[:k]) & set(positives)) / len(set(positives))


def observed_ndcg(ids, positives, k=10):
    """IDCG uses all observed positives, including ones missing from candidates."""
    positives = set(positives)
    if not positives:
        raise ValueError('NDCG needs at least one positive')
    discounts = 1 / np.log2(np.arange(k) + 2)
    dcg = sum(discounts[i] for i, v in enumerate(ids[:k]) if v in positives)
    return float(dcg / discounts[:min(k, len(positives))].sum())


def paired_user_interval(frame, column='delta', repeats=2000, seed=2026):
    """Resample whole users; preserve the original equal-weight query mean."""
    grouped = frame.groupby('user_id')[column].agg(['sum', 'count'])
    sums, counts = grouped['sum'].to_numpy(), grouped['count'].to_numpy()
    rng = np.random.default_rng(seed)
    values = np.empty(repeats)
    for i in range(repeats):
        indices = rng.integers(0, len(grouped), len(grouped))
        values[i] = sums[indices].sum() / counts[indices].sum()
    return dict(estimate=float(frame[column].mean()), lower=float(np.quantile(values, .025)),
                upper=float(np.quantile(values, .975)), positive_fraction=float((values > 0).mean()),
                users=len(grouped), queries=len(frame), repeats=repeats, seed=seed)


def history_group(n):
    return '无历史' if n == 0 else ('1—5条历史' if n <= 5 else '6—20条历史')


def load_routes(paths, split):
    frames = {name: pd.read_parquet(paths.report_dir / f'{name}_{split}_recommendations.parquet')
              .sort_values(['user_id', 'date']).reset_index(drop=True)
              for name in ('selected', 'itemcf', 'popularity')}
    base = frames['selected']
    for frame in frames.values():
        assert not frame[['user_id', 'date']].duplicated().any()
        assert frame[['user_id', 'date']].equals(base[['user_id', 'date']])
        assert all(set(a) == set(b) for a, b in zip(frame.positive_ids, base.positive_ids))
    return frames


def fusion_attribution(paths, items, queries, split, output):
    frames = load_routes(paths, split)
    hist = queries.set_index(['user_id', 'date']).history_len.to_dict()
    games = set(items.loc[items.is_game, 'video_id'])
    rows, changed = [], []
    for a, b, p in zip(frames['selected'].itertuples(), frames['itemcf'].itertuples(), frames['popularity'].itertuples()):
        positive = set(a.positive_ids)
        fused, cf = set(a.recommended_ids[:100]), set(b.recommended_ids[:100])
        gained, lost = (fused-cf) & positive, (cf-fused) & positive
        hit_a, hit_b = len(fused & positive), len(cf & positive)
        game_pos = positive & games
        rows.append(dict(user_id=a.user_id, date=a.date, positives=len(positive), gained=len(gained), lost=len(lost),
                         itemcf=hit_b/len(positive), selected=hit_a/len(positive), delta=(hit_a-hit_b)/len(positive),
                         newly_hit=int(hit_b == 0 and hit_a > 0), lost_all=int(hit_b > 0 and hit_a == 0),
                         history_group=history_group(hist[(a.user_id, a.date)]), game_positives=len(game_pos),
                         game_gained=len(gained & games), game_lost=len(lost & games),
                         game_delta=(len(gained & games)-len(lost & games))/len(game_pos) if game_pos else np.nan))
        for kind, ids in [('新增命中', gained), ('丢失命中', lost)]:
            for video in sorted(ids):
                pop = list(p.recommended_ids)
                changed.append(dict(user_id=a.user_id, date=a.date, video_id=video, change=kind,
                                    is_game=video in games, popularity_rank=pop.index(video)+1 if video in pop else None,
                                    in_popular_top10=video in pop[:10]))
    frame = pd.DataFrame(rows)
    # Each changed positive contributes 1 / positive_count / total_queries.
    changes = pd.DataFrame(changed)
    if len(changes):
        den = frame.set_index(['user_id', 'date']).positives.to_dict()
        changes['macro_contribution'] = [
            (1 if r.change=='新增命中' else -1)/den[(r.user_id,r.date)]/len(frame) for r in changes.itertuples()]
        changes = changes.merge(items[['video_id','caption','first_level_category_name']], on='video_id', validate='many_to_one')
    frame.to_parquet(output/f'fusion_{split}_per_query.parquet', index=False)
    changes.to_csv(output/f'fusion_{split}_changed_hits.csv', index=False, encoding='utf-8-sig')
    users = frame.groupby('user_id').delta.mean()
    def summary(f):
        return dict(queries=len(f), users=int(f.user_id.nunique()), itemcf=float(f.itemcf.mean()), selected=float(f.selected.mean()),
                    delta=float(f.delta.mean()), improved=int((f.delta>0).sum()), harmed=int((f.delta<0).sum()),
                    unchanged=int((f.delta==0).sum()), gained=int(f.gained.sum()), lost=int(f.lost.sum()),
                    newly_hit=int(f.newly_hit.sum()), lost_all=int(f.lost_all.sum()))
    result = summary(frame)
    result['users_by_mean_change'] = dict(improved=int((users>0).sum()), harmed=int((users<0).sum()), unchanged=int((users==0).sum()))
    result['paired_user_interval'] = paired_user_interval(frame)
    result['history_groups'] = {name: summary(f) for name,f in frame.groupby('history_group')}
    gf = frame[frame.game_positives.gt(0)]
    result['game'] = dict(queries=len(gf), gained=int(gf.game_gained.sum()), lost=int(gf.game_lost.sum()),
                          delta=float(gf.game_delta.mean()), paired_user_interval=paired_user_interval(gf, 'game_delta'))
    result['gained_from_popular_top10'] = int(changes.loc[changes.change.eq('新增命中'), 'in_popular_top10'].sum()) if len(changes) else 0
    result['largest_improvements'] = frame.nlargest(3,'delta')[['user_id','date','positives','gained','lost','delta']].assign(date=lambda f:f.date.astype(str)).to_dict('records')
    result['largest_losses'] = frame.nsmallest(3,'delta')[['user_id','date','positives','gained','lost','delta']].assign(date=lambda f:f.date.astype(str)).to_dict('records')
    return result


def score_candidates(config, queries, items, frames, split, output):
    """Fixed scene tab=0 and weight=.7, declared before reading stage results."""
    paths = config['resolved_paths']
    models, collators = {}, {}
    for task in ('long_view', 'is_like'):
        directory = paths.artifact_dir / f'single_{task}_ranker'
        features = FeatureSpace.load(directory/'feature_space.json')
        models[task] = _load_ranker(config, features, directory/'model.pt')
        collators[task] = ExposureCollator(features, items, int(config['data']['history_length']))
    histories = queries.set_index(['user_id','date']).history_ids.to_dict()
    result = []
    base = list(frames.itertuples())
    with torch.inference_mode():
        for start in range(0, len(base), 32):
            chunk = base[start:start+32]
            records = [dict(user_id=q.user_id, video_id=v, tab=0, history_ids=histories[(q.user_id,q.date)],
                            history_len=len(histories[(q.user_id,q.date)]), long_view=0, is_like=0)
                       for q in chunk for v in q.recommended_ids[:100]]
            probs = {}
            for task in models:
                values=[]
                for pos in range(0,len(records),512):
                    batch=collators[task](records[pos:pos+512])
                    values.extend(torch.sigmoid(_ranker_forward(models[task],batch)[task]).tolist())
                probs[task]=values
            offset=0
            for q in chunk:
                n=len(q.recommended_ids[:100])
                result.append(dict(user_id=q.user_id,date=q.date,
                                   long_probability=probs['long_view'][offset:offset+n],
                                   like_probability=probs['is_like'][offset:offset+n]))
                offset+=n
            if start % 1024 == 0:
                print(f'{split}: scored {min(start+32,len(base))}/{len(base)} groups', flush=True)
    scores=pd.DataFrame(result)
    scores.to_parquet(output/f'{split}_candidate_scores.parquet',index=False)
    return scores


def final_lists(config, items, queries, split, output):
    frames=load_routes(config['resolved_paths'],split)['selected']
    scores=score_candidates(config,queries,items,frames,split,output)
    assert scores[['user_id','date']].equals(frames[['user_id','date']])
    metadata=items.set_index('video_id').to_dict('index')
    rows=[]
    for q,p in zip(frames.itertuples(),scores.itertuples()):
        records=[]
        for v,pl,pi in zip(q.recommended_ids[:100],p.long_probability,p.like_probability):
            m=metadata[v]
            text=m.get('caption')
            if pd.isna(text) or not str(text).strip(): text=m.get('show_cover_text','')
            records.append(dict(video_id=v,author_id=m.get('author_id'),caption=text,score=.7*pl+.3*pi))
        ranked=sorted(records,key=lambda r:(-r['score'],str(r['video_id'])))
        stages={'recall':records[:10], 'ranked':ranked[:10],
                'author_cap':author_capped_rerank(records,10,2,deduplicate_caption=False),
                'author_and_caption':author_capped_rerank(records,10,2,deduplicate_caption=True)}
        for stage,chosen in stages.items():
            ids=[r['video_id'] for r in chosen]
            authors=[author_key(r) for r in chosen]
            captions=[caption_key(r) for r in chosen if caption_key(r)]
            excess=sum(max(0,c-2) for c in Counter(authors).values())
            rows.append(dict(user_id=q.user_id,date=q.date,stage=stage,recommended_ids=ids,
                             recall10=recall(ids,q.positive_ids,10),ndcg10=observed_ndcg(ids,q.positive_ids),
                             count=len(ids),repeat_author=len(authors)-len(set(authors)),excess_author=excess,
                             repeat_caption=len(captions)-len(set(captions)),caption_count=len(captions)))
    frame=pd.DataFrame(rows)
    frame.to_parquet(output/f'{split}_final_lists.parquet',index=False)
    summaries={}
    for stage,f in frame.groupby('stage',sort=False):
        total=int(f['count'].sum())
        summaries[stage]=dict(queries=len(f),recall10=float(f.recall10.mean()),ndcg10=float(f.ndcg10.mean()),
            repeat_author_ratio=float(f.repeat_author.sum()/total),
            author_over_cap_list_ratio=float(f.excess_author.gt(0).mean()),
            repeat_caption_ratio=float(f.repeat_caption.sum()/total),
            lists_with_repeated_caption=int(f.repeat_caption.gt(0).sum()),
            coverage=len({v for row in f.recommended_ids for v in row})/len(items),
            fewer_than10=int(f['count'].lt(10).sum()),returned_items=total)
    paired=frame.pivot(index=['user_id','date'],columns='stage',values='recall10').reset_index()
    comparisons={}
    for left,right in [('ranked','recall'),('author_cap','ranked'),('author_and_caption','author_cap')]:
        paired['delta']=paired[left]-paired[right]
        comparisons[left+'_minus_'+right]=paired_user_interval(paired)
    return dict(stages=summaries,recall10_paired_intervals=comparisons)


def run(config):
    torch.set_num_threads(4)
    paths=config['resolved_paths']
    output=paths.report_dir/'final_evaluation'
    output.mkdir(exist_ok=True)
    items=pd.read_parquet(paths.processed_dir/'items_content.parquet')
    queries=pd.read_parquet(paths.processed_dir/'recall_queries.parquet')
    settings=dict(scene_tab=0,watch_weight=.7,like_weight=.3,candidates=100,output_k=10,
        author_limit=2,bootstrap_repeats=2000,bootstrap_seed=2026,
        ndcg_denominator='All observed positives for this user/date, not only recalled positives',
        scene='Fixed tab=0 scenario; no later observed tab used to score candidates',
        interpretation='Offline replay of observed positives; unlabeled candidates have unknown actual feedback',
        test_history='Previously inspected test set; this analysis does not create new held-out data')
    write_json(output/'settings.json',settings)
    result={'settings':settings,'fusion':{},'lists':{}}
    for split in ('dev','test'):
        result['fusion'][split]=fusion_attribution(paths,items,queries,split,output)
        print(split,'fusion',result['fusion'][split]['delta'],flush=True)
        result['lists'][split]=final_lists(config,items,queries,split,output)
        write_json(output/'results.json',result)
    write_markdown(result,output)
    return result


def write_markdown(result,output):
    lines=['# 融合提升与最终推荐列表检查','',
        '固定现有模型、100条候选、场景tab=0、观看达标/点赞权重0.7/0.3。开发与测试分别报告，本次没有根据测试结果调权重。',
        '正例为同一用户同一天日志中观看达标的不同视频。结果衡量已观察正例的找回，不估计未知候选的真实观看率。',
        '每组等权平均；重复抽样时同一用户的全部日期一起抽取，两个方法使用同一批抽中用户。区间反映评价用户变化，不代表重新训练的波动。测试集此前已查看。','',
        '## 1. 融合比ItemCF改善了什么','',
        '| 集合 | ItemCF Recall@100 | 融合 | 差值（百分点） | 用户重复抽样95%区间（百分点） |',
        '|---|---:|---:|---:|---|']
    notes=[]
    for split,r in result['fusion'].items():
        ci=r['paired_user_interval']
        lines.append(f"| {split} | {r['itemcf']:.4%} | {r['selected']:.4%} | {100*r['delta']:.4f} | [{100*ci['lower']:.4f}, {100*ci['upper']:.4f}] |")
        notes.extend(['',f"{split}：{r['improved']}组改善、{r['harmed']}组下降、{r['unchanged']}组不变；新增{r['gained']}次正例命中、丢失{r['lost']}次。其中{r['newly_hit']}组从零命中变成有命中，{r['lost_all']}组丢失全部命中。新增命中来自热门前10条的有{r['gained_from_popular_top10']}次。",''])
    lines.extend(notes)
    r=result['fusion']['test']
    lines.extend(['### 测试集按历史长度','', '| 历史 | 样本数 | 改善组 | 下降组 | Recall差值（百分点） |','|---|---:|---:|---:|---:|'])
    for name,g in r['history_groups'].items():
        lines.append(f"| {name} | {g['queries']} | {g['improved']} | {g['harmed']} | {100*g['delta']:.4f} |")
    game=r['game']
    lines.extend(['',f"游戏正例：{game['queries']}组中新增{game['gained']}次命中、丢失{game['lost']}次，逐组平均Recall差值为{100*game['delta']:.4f}个百分点。整体与游戏组分母不同，分别计算。",'',
                  '## 2. 前10条推荐的变化',''])
    names={'recall':'融合RRF顺序','ranked':'模型排序','author_cap':'排序＋同作者最多2条','author_and_caption':'排序＋作者限制＋文案去重'}
    for split,r in result['lists'].items():
        lines.extend([f'### {split}','', '| 阶段 | Recall@10 | NDCG@10 | 同作者重复占比 | 重复文案占比 | 内容覆盖率 | 不足10条的组数 |','|---|---:|---:|---:|---:|---:|---:|'])
        for stage,g in r['stages'].items():
            lines.append(f"| {names[stage]} | {g['recall10']:.4%} | {g['ndcg10']:.4f} | {g['repeat_author_ratio']:.4%} | {g['repeat_caption_ratio']:.4%} | {g['coverage']:.4%} | {g['fewer_than10']} |")
        lines.append('')
    lines.extend(['','同作者重复占比：每组返回列表中，某作者第一次出现后的额外条数，除以全部返回条数。同作者最多2条仍允许出现一次重复，因此该指标不一定为0。',
        '重复文案占比：同一组中非空文案首次出现后的额外条数，除以全部返回条数；比较时合并连续空白，其他文字保持不变。空文案互不合并。',
        'NDCG的理想排序使用当天全部已观察正例，包括未被召回的正例。覆盖率分母为7583条内容。',
        '作者缺失时每个视频分别计数，避免把未知作者错误视为同一作者。', '',
        '本次开发集的模型排序降低了Recall@10和NDCG@10，因此演示默认保留融合候选的RRF顺序。模型排序、作者限制与文案去重保留为可勾选的实验对照。',
        '作者数量限制在当前模型排序后的前10条没有触发，不能据此声称它提高了多样性；文案去重确实消除了重复文案。', '',
        '## 3. 可核对文件','',
        '- `fusion_*_per_query.parquet`：每个用户和日期的新增/丢失命中与指标变化。',
        '- `fusion_*_changed_hits.csv`：变化涉及的视频、文案、是否游戏、热门名次及对总指标的贡献。',
        '- `*_candidate_scores.parquet`：两个现有模型对每条候选的预测，顺序对应原selected推荐文件。',
        '- `*_final_lists.parquet`：各阶段最终列表及逐组指标。',
        '- `results.json`、`settings.json`：汇总与事先确定的设置。'])
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

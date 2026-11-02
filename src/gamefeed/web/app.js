const $ = id => document.getElementById(id);
const escapeText = value => String(value ?? '').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = x => (100*x).toFixed(2)+'%';
const names={itemcf:'协同过滤',semantic:'语义相似',text:'语义相似',popularity:'热门',selected:'选定方案'};
async function request(url){const res=await fetch(url);const data=await res.json();if(!res.ok)throw new Error(data.error||'请求失败');return data;}
async function recommend(){
 $('run').disabled=true;$('status').textContent='正在计算候选内容的观看达标和点赞得分…';
 try{const [user,date]=$('query').value.split('|');const p=new URLSearchParams({user,date,method:$('method').value,tab:$('tab').value,rank:$('rank').checked?'1':'0',game:$('game').checked?'1':'0'});const d=await request('/api/recommend?'+p);
 $('history-count').textContent=d.history.length+'条';$('result-count').textContent=d.displayed_count+' / '+d.candidate_count+'条候选';
 $('history').innerHTML=d.history.length?d.history.map(r=>`<div class="history-item"><small>${escapeText(r.category)} · #${escapeText(r.video_id)}</small>${escapeText(r.caption)}</div>`).join(''):'<p class="empty">这名用户暂无此前观看达标记录。</p>';
 $('results').innerHTML=d.recommendations.length?d.recommendations.map((r,i)=>`<article class="card"><div class="card-head"><span>${String(i+1).padStart(2,'0')} · ${escapeText(r.category)}</span><span>#${escapeText(r.video_id)}</span></div><h3>${escapeText(r.caption)}</h3><p class="source">召回来源：${r.channels.map(x=>escapeText(names[x]||x)).join(' / ')}</p><details><summary>查看模型打分</summary><div class="score-row"><span>观看达标概率</span><strong>${pct(r.long_probability)}</strong></div><div class="score-row"><span>点赞概率</span><strong>${pct(r.like_probability)}</strong></div><div class="score-row"><span>演示综合分</span><strong>${r.score.toFixed(4)}</strong></div><p class="source">召回名次 ${r.recall_rank} · 作者 ${escapeText(r.author_id)}</p></details></article>`).join(''):'<p class="empty">这100条候选中没有符合筛选条件的内容，请关闭游戏筛选或切换方案。</p>';
 $('status').textContent=`用户 ${d.user_id} · ${d.date} · 打分与重排 ${d.scoring_and_rerank_ms.toFixed(1)} ms。${$('game').checked?'当前仅展示候选中的游戏内容。':''}`;
 }catch(e){$('status').textContent='加载失败：'+e.message;}finally{$('run').disabled=false;}
}
async function init(){try{const d=await request('/api/options');$('query').innerHTML=d.queries.map(q=>`<option value="${escapeText(q.user_id+'|'+q.date)}">${escapeText(q.user_id)} · ${q.date} · 历史${q.history_len}条 / 游戏${q.game_history}条</option>`).join('');$('tab').innerHTML=d.tabs.map(t=>`<option>${t}</option>`).join('');
 const s=d.summary;$('metrics').innerHTML=[['测试集 Recall@100',pct(s.selected_test['recall@100']),'开发集选定方案 · 全体内容'],['ItemCF基线 Recall@100',pct(s.results.itemcf.test['recall@100']),'融合提高0.69个百分点；游戏组融合29.62%、ItemCF29.91%'],['热门基线 Recall@100',pct(s.popularity_test['recall@100']),'融合提高3.10个百分点；739条游戏内容单独分析']].map(([label,value,note])=>`<div class="metric"><small>${label}</small><strong>${value}</strong><small>${note}</small></div>`).join('');await recommend();}catch(e){$('status').textContent='数据加载失败：'+e.message;}}
$('run').addEventListener('click',recommend);init();

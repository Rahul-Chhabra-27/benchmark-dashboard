/* Dedicated explorer: cumulative token usage is distinct from peak root context. */
window.StructuredDashboard = (() => {
  let data, host, loading;
  const state = {campaign:'compare', task:'nq_128k', selection:'comparison', depth:'all', chart:'tradeoff', sort:'score', source:'all', method:'all', pareto:'all'};
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = (n, digits=0) => n == null ? '—' : Number(n).toLocaleString('en-US', {maximumFractionDigits:digits});
  const short = n => n >= 1000 ? `${fmt(n/1000,1)}k` : fmt(n);
  const names = {nq_128k:'NQ',hotpotqa_128k:'HotpotQA',musique_128k:'MuSiQue',qampari_128k:'Qampari',quest_128k:'Quest',pooled_n110:'Pooled · 110 rows',pooled_n51:'Pooled · 51 rows'};
  const selectionNames = {aware:'Question-guided selection',independent:'Question-independent selection',oracle:'Gold-informed diagnostic',historical:'Historical comparison',
    'order-graph':'BM25-graph order · reads every chunk','order-source':'Source order · reads every chunk',vanilla:'Vanilla · whole document',rlm:'Base RLM · agentic',kvzip:'KVzip only · compressed KV, no RLM'};
  const palette = {aware:'#245b83',independent:'#187567',oracle:'#a66a13',historical:'#77668c',
    'order-graph':'#b5452b','order-source':'#4c6a92',vanilla:'#2f7d4a',rlm:'#8a5a9e',kvzip:'#a66a13'};
  const metricName = r => r.metric === 'coverage' ? 'Answer coverage' : 'Subspan EM';
  // Cross-campaign view: every completed Sep 9 / Sep 10 configuration on one axis.
  // All three campaigns score the same 55 test questions per subset with the same
  // model, so their scores and token usage are directly comparable.
  const COMPARE = ['graph-order','query','geometry'];
  const COMPARE_TITLE = 'Compare all · Sep 9 + Sep 10';
  const campaignColors = {'graph-order':'#b5452b',query:'#245b83',geometry:'#187567',reference:'#8a8f98'};
  const campaignTags = {'graph-order':'Sep 10 · order',query:'Sep 10 · selection',geometry:'Sep 9 · geometry'};
  const METHODS = [['all','All methods'],['bm25-bare','Corrected BM25 · question only'],['bm25-task','BM25 · task + question (old query)'],['independent','Question-independent selection'],['order','Whole document · source / graph order'],['reference','References · vanilla / KVzip / base RLM'],['oracle','Oracle diagnostic']];
  const isCompare = () => state.campaign === 'compare';
  const campaignTitle = id => (data.campaigns.find(c => c.id === id) || {}).title || id;
  const methodOf = r => ['source','bm25-graph'].includes(r.arm) ? 'order' : ['vanilla','rlm','kvzip'].includes(r.arm) ? 'reference' : ['bm25-bare','bm25-task','oracle'].includes(r.arm) ? r.arm : 'independent';
  // References (vanilla / KVzip / base RLM) ride in the graph-order campaign but are
  // not graph-order results, so they get their own colour in the compare view.
  const colorKey = r => methodOf(r) === 'reference' ? 'reference' : r.campaign;
  const colorOf = r => isCompare() ? campaignColors[colorKey(r)] : palette[r.selection];
  const labelOf = r => isCompare() ? `${campaignTags[r.campaign]} · ${r.label}` : r.label;
  const RETRIEVAL_EXPLAINER = 'BM25 retrieval (the question-guided rows) is a different use of the same scorer: the document is cut into windows, the QUESTION is the BM25 query, and only the k best-scoring windows are read. "Corrected BM25" queries with the question alone; the old "task + question" query also carried the instruction text, whose words swamped the question and pulled in the wrong windows. BM25-graph ORDERING below never sees the question — it reorders every chunk and all of them are read.';
  const compareCampaign = () => ({id:'compare', title:COMPARE_TITLE,
    explainer:[RETRIEVAL_EXPLAINER, ...((data.campaigns.find(c => c.id === 'graph-order') || {}).explainer || [])],
    description:'Every completed configuration from the Sep 9 chunk-geometry sweep, the Sep 10 query-selection study and the Sep 10 graph-order campaign (with its vanilla, KVzip-only and base-RLM references), on one axis. Same model and the same 55 LOFT-128k test questions per subset, so scores and token usage compare directly across campaigns.',
    note:'"Corrected BM25" is the question-only retrieval query; "task + question" is the old instruction-heavy query it replaced (Sep 9 oracle, read-cap and two-hop cells still use it). Tokens are recorded input + output model tokens per query across root and sub-calls (KVzip: prefilled prompt + generated answer). Graph-order rows are an interim snapshot until that campaign finishes. Sep 9 has no Quest runs. Qampari and Quest are scored on answer coverage, the others on subspan EM. "Best score for its token cost" keeps only configurations that no other one matches or beats on score while using no more tokens.'});
  const campaignRows = () => isCompare() ? data.runs.filter(r => COMPARE.includes(r.campaign)) : data.runs.filter(r => r.campaign === state.campaign);
  function paretoOnly(rows) {
    return rows.filter(r => r.tokens_per_query != null && !rows.some(o => o !== r && o.tokens_per_query != null
      && o.tokens_per_query <= r.tokens_per_query && o.score >= r.score && (o.tokens_per_query < r.tokens_per_query || o.score > r.score)));
  }
  function compareFilters() {
    return select('From','source',[['all','All three campaigns'],...COMPARE.map(id=>[id,campaignTitle(id)])])
      + select('Method','method',METHODS)
      + select('Show','pareto',[['all','Every configuration'],['pareto','Best score for its token cost']]);
  }
  function legend(rows) {
    return isCompare()
      ? [...new Set(rows.map(colorKey))].map(c=>`<span><i style="background:${campaignColors[c]}"></i>${escape(c==='reference'?'References · vanilla / KVzip only / base RLM':campaignTitle(c))}</span>`).join('')
      : [...new Set(rows.map(r=>r.selection))].map(s=>`<span><i style="background:${palette[s]}"></i>${selectionNames[s]}</span>`).join('');
  }
  function visibleRows() {
    let rows = campaignRows().filter(r => r.task === state.task);
    if (isCompare()) {
      rows = rows.filter(r => state.source === 'all' || r.campaign === state.source).filter(r => state.method === 'all' || methodOf(r) === state.method);
      if (state.pareto === 'pareto') rows = paretoOnly(rows);
    } else {
      rows = rows.filter(r => state.selection === 'all' || (state.selection === 'comparison' ? r.arm === 'bm25-bare' || r.arm === 'all' : r.selection === state.selection))
        .filter(r => state.depth === 'all' || (state.depth === 'exhaustive' ? r.arm === 'all' : r.k === Number(state.depth) && r.arm !== 'all'));
    }
    return rows
      .sort((a,b) => state.sort === 'tokens' ? (a.tokens_per_query ?? Infinity)-(b.tokens_per_query ?? Infinity) || b.score-a.score : b.score-a.score || (a.tokens_per_query ?? Infinity)-(b.tokens_per_query ?? Infinity));
  }
  function options(items, selected) { return items.map(([value,label]) => `<option value="${escape(value)}" ${String(value)===selected?'selected':''}>${escape(label)}</option>`).join(''); }
  function select(label, field, items) { return `<label class="sr-field">${label}<select data-field="${field}">${options(items,state[field])}</select></label>`; }
  function syncURL() {
    history.replaceState(null,'',`#structured?${new URLSearchParams(state)}`);
  }
  function render() {
    const campaign = isCompare() ? compareCampaign() : data.campaigns.find(c => c.id === state.campaign);
    const all = campaignRows(), tasks = [...new Set(all.map(r => r.task))];
    if (!tasks.includes(state.task)) state.task = tasks[0];
    const historical = !['query','geometry','compare'].includes(state.campaign);
    if (historical) { state.selection='all'; state.depth='all'; }
    const rows = visibleRows(), taskRows = all.filter(r => r.task === state.task);
    const label = metricName(taskRows[0]), costKnown = taskRows.some(r => r.tokens_per_query != null);
    const depths = [...new Set(taskRows.filter(r=>r.arm!=='all').map(r=>r.k).filter(k=>k!=null))].sort((a,b)=>a-b);
    syncURL();
    host.innerHTML = `
      <div class="sr-heading"><div><span class="sr-eyebrow">LOFT 128K · ${escape(data.model)}</span><h2>Structured RLM</h2><p>How context selection changes answer quality and token use.</p></div><a class="sr-link" href="data/structured_results.csv" download>Download all results ↓</a></div>
      <div class="sr-definitions"><div><b>Question-guided</b><span>BM25 ranks windows using the question.</span></div><div><b>Question-independent selection</b><span>Lead, stride, random or all windows. Reading still uses the question.</span></div><div><b>Process once, answer later</b><span>All five notes indexes are built. Answering and held-out evaluation are pending.</span></div></div>
      <section class="card sr-study"><div class="sr-filter-grid">
        ${select('Campaign','campaign',[['compare',COMPARE_TITLE],...data.campaigns.map(c=>[c.id,c.title])])}
        ${select('Subset','task',tasks.map(t=>[t,names[t]||t]))}
        ${isCompare() ? compareFilters() : ''}${!historical && !isCompare() ? select('Selection','selection',[['comparison','Corrected BM25 + exhaustive'],['all','All methods'],['aware','Question-guided'],['independent','Question-independent'],['oracle','Oracle diagnostic']]) : ''}
        ${!historical && !isCompare() ? select('Read depth','depth',[['all','All depths'],...depths.map(k=>[String(k),`${k} windows`]),...(taskRows.some(r=>r.arm==='all')?[['exhaustive','Every window']]:[])]) : ''}
      </div><p class="sr-description">${escape(campaign.description)}</p>${(campaign.explainer||[]).length?`<details class="sr-explainer"><summary>How BM25 retrieval and BM25-graph ordering work</summary>${campaign.explainer.map(p=>`<p>${escape(p)}</p>`).join('')}</details>`:''}<div class="sr-meta"><span>${all.length} completed results</span><span>${fmt(all.reduce((n,r)=>n+r.n,0))} example-runs</span><span>Snapshot · ${data.updated}</span></div><p class="sr-caveat">${escape(campaign.note)}</p>${campaign.task_notes&&campaign.task_notes[state.task]?`<p class="sr-caveat"><b>${escape(names[state.task]||state.task)}:</b> ${escape(campaign.task_notes[state.task])}</p>`:''}</section>
      <section class="card"><div class="sr-chart-header"><div><h3>${escape(names[state.task]||state.task)} <span>· ${escape(label)}</span></h3><p class="sr-muted">${rows.length} configurations in this selection</p></div><div class="sr-chart-controls" role="group" aria-label="Chart measure">${[['score','Score'],['tokens','Tokens / query'],['tradeoff','Score vs tokens']].map(([value,title])=>`<button class="sr-button ${state.chart===value?'active':''}" data-chart="${value}" aria-pressed="${state.chart===value}">${title}</button>`).join('')}</div></div>
      <div class="sr-chart" id="sr-chart">${chart(rows,label)}</div><div class="sr-legend">${legend(rows)}</div>
      <p class="sr-muted">Select a bar, point or configuration to inspect the run. ${state.chart==='tradeoff'?'Token axis uses a logarithmic scale.':''}</p></section>
      <section class="card"><div class="sr-chart-header"><h3>Results and usage</h3><div class="sr-table-actions">${select('Sort by','sort',[['score','Highest score'],['tokens','Fewest tokens']])}<button class="sr-button" id="sr-export">Export selection ↓</button></div></div>
      <p class="sr-muted">${escape(data.token_definition)} Root context is reported separately. ${costKnown?'Latency reflects shared-server load; it is not a controlled speed comparison.':'Usage is unavailable in this historical summary; it is never inferred from context length.'}</p>
      <div class="sr-table-scroll"><table class="sr-table"><thead><tr><th>Configuration</th><th>${escape(label)}</th><th>Samples</th><th>Tokens / query</th><th>Tokens / run</th><th>Input / output<br><small>tokens per query</small></th><th>Root context<br><small>mean peak tokens</small></th><th>Map calls / query</th><th>Latency / query</th><th>Predictions</th></tr></thead><tbody>${rows.map(r=>`<tr><td><button class="sr-run" data-run="${r.id}">${escape(r.label)}</button><small>${isCompare()?escape(campaignTitle(r.campaign))+' · ':''}${selectionNames[r.selection]}</small><small class="sr-mobile-usage">${r.tokens_per_query==null?'Token use unavailable':fmt(r.tokens_per_query)+' tokens / query'}</small></td><td class="num"><b>${fmt(r.score,2)}%</b></td><td class="num">${r.n}</td><td class="num">${fmt(r.tokens_per_query)}</td><td class="num">${fmt(r.total_tokens)}</td><td class="num">${r.prompt_tokens_per_query==null?'—':fmt(r.prompt_tokens_per_query)+' / '+fmt(r.completion_tokens_per_query)}</td><td class="num">${fmt(r.root_peak_tokens)}</td><td class="num">${fmt(r.map_calls,2)}</td><td class="num">${r.latency_s==null?'—':fmt(r.latency_s,1)+' s'}</td><td>${r.prediction_url?`<a href="${r.prediction_url}" download>CSV ↓</a>`:'—'}</td></tr>`).join('')||'<tr><td colspan="10" class="empty">No completed configurations match these filters. Try All methods or All depths.</td></tr>'}</tbody></table></div></section>
      <dialog class="sr-dialog" id="sr-dialog" aria-labelledby="sr-detail-title"></dialog>`;
    host.querySelectorAll('[data-field]').forEach(el=>el.addEventListener('change',()=>{
      state[el.dataset.field]=el.value;
      if(el.dataset.field==='campaign') {state.selection=['query','geometry'].includes(state.campaign)?'comparison':'all';state.depth='all';if(isCompare())state.chart='tradeoff';}
      if (!host.hidden) render();
    }));
    host.querySelectorAll('[data-chart]').forEach(el=>el.addEventListener('click',()=>{state.chart=el.dataset.chart;render();}));
    host.querySelectorAll('[data-run]').forEach(el=>{
      el.addEventListener('click',()=>details(el.dataset.run));
      if(el.tagName.toLowerCase()!=='button') el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();details(el.dataset.run);}});
    });
    host.querySelector('#sr-export').onclick = () => exportRows(rows);
  }
  function chart(rows,label) {
    if (!rows.length) return '<p class="empty">No completed results match this selection.</p>';
    const valid = rows.filter(r=>state.chart==='score'||r.tokens_per_query!=null);
    if (!valid.length) return '<p class="empty">Token usage was not recorded in this historical summary. Select a newer campaign to compare usage.</p>';
    if(state.chart==='tradeoff') {
      const W=1000,H=410,L=75,R=35,T=30,B=65;
      const logs=valid.map(r=>Math.log10(Math.max(1,r.tokens_per_query)));
      const min=Math.floor(Math.min(...logs)),max=Math.max(min+1,Math.ceil(Math.max(...logs)));
      const x=n=>L+(Math.log10(Math.max(1,n))-min)/(max-min)*(W-L-R),y=n=>H-B-n/100*(H-T-B);
      let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${escape(label)} versus mean tokens per query"><title>${escape(label)} versus token usage</title>`;
      for(let score=0;score<=100;score+=20) svg+=`<line x1="${L}" x2="${W-R}" y1="${y(score)}" y2="${y(score)}" class="sr-grid"/><text x="${L-12}" y="${y(score)+4}" text-anchor="end">${score}%</text>`;
      for(let power=min;power<=max;power++) svg+=`<text x="${x(10**power)}" y="${H-B+25}" text-anchor="middle">${short(10**power)}</text>`;
      valid.forEach(r=>{const tip=`${labelOf(r)}: ${fmt(r.score,2)}%, ${fmt(r.tokens_per_query)} tokens/query`;svg+=`<circle cx="${x(r.tokens_per_query)}" cy="${y(r.score)}" r="7" fill="${colorOf(r)}" stroke="white" stroke-width="2" tabindex="0" role="button" aria-label="${escape(tip)}" data-run="${r.id}"><title>${escape(tip)}</title></circle>`;});
      return svg+`<text x="${W/2}" y="${H-13}" text-anchor="middle">Mean model tokens per query · logarithmic scale</text></svg>`;
    }
    const mobile=window.innerWidth<650;
    const W=mobile?640:(isCompare()?1280:1120),L=mobile?20:(isCompare()?600:440),R=105,T=30,rowH=mobile?78:36,H=T+valid.length*rowH+40,plotW=W-L-R;
    const value=r=>state.chart==='tokens'?r.tokens_per_query:r.score;
    const max=state.chart==='tokens'?Math.max(1,...valid.map(value)):100;
    let svg=`<svg viewBox="0 0 ${W} ${H}" ${mobile?'style="font-size:24px"':''} role="img" aria-label="${state.chart==='tokens'?'Mean tokens per query':escape(label)} by configuration"><title>${state.chart==='tokens'?'Token usage':escape(label)} by configuration</title>`;
    for(let i=0;i<=4;i++){const v=max*i/4,x=L+plotW*i/4;svg+=`<line x1="${x}" x2="${x}" y1="15" y2="${H-30}" class="sr-grid"/><text x="${x}" y="${H-9}" text-anchor="middle">${state.chart==='tokens'?short(v):fmt(v)+'%'}</text>`;}
    valid.forEach((r,i)=>{
      const v=value(r),y=T+i*rowH+(mobile?25:0),tip=`${labelOf(r)}: ${fmt(r.score,2)}%, ${fmt(r.tokens_per_query)} tokens/query`;
      const text=labelOf(r),caption=mobile&&text.length>43?text.slice(0,42)+'…':text;
      svg+=`<g data-run="${r.id}" tabindex="0" role="button" aria-label="${escape(tip)}"><title>${escape(tip)}</title><text x="${mobile?L:L-14}" y="${mobile?y-12:y+16}" text-anchor="${mobile?'start':'end'}">${escape(caption)}</text><rect x="${L}" y="${y}" width="${Math.max(1,v/max*plotW)}" height="24" rx="4" fill="${colorOf(r)}"/><text x="${L+v/max*plotW+9}" y="${y+20}" font-weight="700">${state.chart==='tokens'?short(v):fmt(v,1)+'%'}</text></g>`;
    });
    return svg+'</svg>';
  }
  function details(id) {
    const r=data.runs.find(r=>r.id===id), dialog=host.querySelector('#sr-dialog');
    const fields=[['Selection',selectionNames[r.selection]],['Reader',r.reader||(r.campaign==='query'||r.campaign==='geometry'?'Question-conditioned map → reduce':'Historical arm; see campaign note')],['Input / output tokens per query',r.prompt_tokens_per_query==null?'—':fmt(r.prompt_tokens_per_query)+' / '+fmt(r.completion_tokens_per_query)],['Samples',r.n],['Score',fmt(r.score,2)+'% · '+metricName(r)],['Tokens / query',fmt(r.tokens_per_query)],['Tokens / run',fmt(r.total_tokens)],['Mean peak root context',fmt(r.root_peak_tokens)+' tokens'],['Window / overlap',r.window?`${fmt(r.window)} / ${fmt(r.overlap)} characters`:'—'],['Read cap',r.read_chars?fmt(r.read_chars)+' characters':'Full selected span'],['Finding cap',r.finding_chars?fmt(r.finding_chars)+' characters':'—'],['Map grouping',r.granularity||'—'],['Document coverage',r.document_coverage==null?'—':fmt(r.document_coverage*100,1)+'%'],['Abstentions / unfinished / errors',`${fmt(r.abstained)} / ${fmt(r.unfinished)} / ${fmt(r.errors)}`]];
    dialog.innerHTML=`<div class="sr-chart-header"><h3 id="sr-detail-title">${escape(r.label)}</h3><button class="sr-button" id="sr-close" autofocus>Close</button></div><dl class="sr-detail-grid">${fields.map(([key,val])=>`<div><dt>${escape(key)}</dt><dd>${escape(val)}</dd></div>`).join('')}</dl>${r.run_name?`<details><summary>Source run identifier</summary><code class="sr-run-id">${escape(r.run_name)}</code></details>`:''}${r.prediction_url?`<p><a href="${r.prediction_url}" download>Download all ${r.n} predictions and per-query token counts ↓</a></p><h4>First three predictions</h4>${r.preview.map(p=>`<details class="sr-pred"><summary>${escape(p.id)} · ${fmt(p.tokens)} tokens</summary><p><b>Reference:</b> ${escape(p.answers)}</p><pre>${escape(p.prediction)}</pre></details>`).join('')}`:'<p class="sr-muted">Per-example artifacts are unavailable in this historical summary export.</p>'}`;
    dialog.querySelector('#sr-close').onclick=()=>dialog.close();
    dialog.showModal();
  }
  function exportRows(rows) {
    const fields=['campaign','task','label','selection','metric','score','n','tokens_per_query','total_tokens','prompt_tokens_per_query','completion_tokens_per_query','root_peak_tokens','latency_s','k','window','read_chars','finding_chars','hops','document_coverage','map_calls','abstained','unfinished','errors'];
    const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';
    const csv=[fields.map(quote).join(','),...rows.map(r=>fields.map(f=>quote(r[f])).join(','))].join('\r\n');
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
    const a=document.createElement('a');a.href=url;a.download=`structured-${state.campaign}-${state.task}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  async function mount(element) {
    host=element;
    if(data) {render();return;}
    host.innerHTML='<div class="card" role="status">Loading Structured RLM results…</div>';
    try {
      loading ||= fetch('data/structured_results.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json();});
      data=await loading;
      const saved=new URLSearchParams(location.hash.split('?')[1]||'');
      for(const key of Object.keys(state)) if(saved.has(key)) state[key]=saved.get(key);
      if(state.campaign!=='compare'&&!data.campaigns.some(c=>c.id===state.campaign))state.campaign='compare';
      if(!['all',...COMPARE].includes(state.source))state.source='all';
      if(!METHODS.some(([value])=>value===state.method))state.method='all';
      if(!['all','pareto'].includes(state.pareto))state.pareto='all';
      if(!['comparison','all','aware','independent','oracle'].includes(state.selection))state.selection='comparison';
      if(!['score','tokens','tradeoff'].includes(state.chart))state.chart='score';
      if (!host.hidden) render();
    } catch(error) {
      loading=null;host.innerHTML='<div class="card"><p>Could not load the results. Reload to retry, or <a href="data/structured_results.csv">download the CSV</a>.</p></div>';
      console.error(error);
    }
  }
  let resizeTimer;
  window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(data&&host&&!host.hidden)render();},150);});
  return {mount};
})();

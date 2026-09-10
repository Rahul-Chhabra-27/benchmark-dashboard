/* Dedicated explorer: cumulative token usage is distinct from peak root context. */
window.StructuredDashboard = (() => {
  let data, host, loading;
  const state = {campaign:'query', task:'nq_128k', selection:'comparison', depth:'all', chart:'score', sort:'score'};
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = (n, digits=0) => n == null ? '—' : Number(n).toLocaleString('en-US', {maximumFractionDigits:digits});
  const short = n => n >= 1000 ? `${fmt(n/1000,1)}k` : fmt(n);
  const names = {nq_128k:'NQ',hotpotqa_128k:'HotpotQA',musique_128k:'MuSiQue',qampari_128k:'Qampari',quest_128k:'Quest',pooled_n110:'Pooled · 110 rows',pooled_n51:'Pooled · 51 rows'};
  const selectionNames = {aware:'Question-guided selection',independent:'Question-independent selection',oracle:'Gold-informed diagnostic',historical:'Historical comparison'};
  const palette = {aware:'#245b83',independent:'#187567',oracle:'#a66a13',historical:'#77668c'};
  const metricName = r => r.metric === 'coverage' ? 'Answer coverage' : 'Subspan EM';
  const campaignRows = () => data.runs.filter(r => r.campaign === state.campaign);
  function visibleRows() {
    return campaignRows().filter(r => r.task === state.task)
      .filter(r => state.selection === 'all' || (state.selection === 'comparison' ? r.arm === 'bm25-bare' || r.arm === 'all' : r.selection === state.selection))
      .filter(r => state.depth === 'all' || (state.depth === 'exhaustive' ? r.arm === 'all' : r.k === Number(state.depth) && r.arm !== 'all'))
      .sort((a,b) => state.sort === 'tokens' ? (a.tokens_per_query ?? Infinity)-(b.tokens_per_query ?? Infinity) || b.score-a.score : b.score-a.score || (a.tokens_per_query ?? Infinity)-(b.tokens_per_query ?? Infinity));
  }
  function options(items, selected) { return items.map(([value,label]) => `<option value="${escape(value)}" ${String(value)===selected?'selected':''}>${escape(label)}</option>`).join(''); }
  function select(label, field, items) { return `<label class="sr-field">${label}<select data-field="${field}">${options(items,state[field])}</select></label>`; }
  function syncURL() {
    history.replaceState(null,'',`#structured?${new URLSearchParams(state)}`);
  }
  function render() {
    const campaign = data.campaigns.find(c => c.id === state.campaign);
    const all = campaignRows(), tasks = [...new Set(all.map(r => r.task))];
    if (!tasks.includes(state.task)) state.task = tasks[0];
    const historical = !['query','geometry'].includes(state.campaign);
    if (historical) { state.selection='all'; state.depth='all'; }
    const rows = visibleRows(), taskRows = all.filter(r => r.task === state.task);
    const label = metricName(taskRows[0]), costKnown = taskRows.some(r => r.tokens_per_query != null);
    const depths = [...new Set(taskRows.filter(r=>r.arm!=='all').map(r=>r.k).filter(k=>k!=null))].sort((a,b)=>a-b);
    syncURL();
    host.innerHTML = `
      <div class="sr-heading"><div><span class="sr-eyebrow">LOFT 128K · ${escape(data.model)}</span><h2>Structured RLM</h2><p>How context selection changes answer quality and token use.</p></div><a class="sr-link" href="data/structured_results.csv" download>Download all results ↓</a></div>
      <div class="sr-definitions"><div><b>Question-guided</b><span>BM25 ranks windows using the question.</span></div><div><b>Question-independent selection</b><span>Lead, stride, random or all windows. Reading still uses the question.</span></div><div><b>Process once, answer later</b><span>All five notes indexes are built. Answering and held-out evaluation are pending.</span></div></div>
      <section class="card sr-study"><div class="sr-filter-grid">
        ${select('Campaign','campaign',data.campaigns.map(c=>[c.id,c.title]))}
        ${select('Subset','task',tasks.map(t=>[t,names[t]||t]))}
        ${!historical ? select('Selection','selection',[['comparison','Corrected BM25 + exhaustive'],['all','All methods'],['aware','Question-guided'],['independent','Question-independent'],['oracle','Oracle diagnostic']]) : ''}
        ${!historical ? select('Read depth','depth',[['all','All depths'],...depths.map(k=>[String(k),`${k} windows`]),...(taskRows.some(r=>r.arm==='all')?[['exhaustive','Every window']]:[])]) : ''}
      </div><p class="sr-description">${escape(campaign.description)}</p><div class="sr-meta"><span>${all.length} completed results</span><span>${fmt(all.reduce((n,r)=>n+r.n,0))} example-runs</span><span>Snapshot · ${data.updated}</span></div><p class="sr-caveat">${escape(campaign.note)}</p></section>
      <section class="card"><div class="sr-chart-header"><div><h3>${escape(names[state.task]||state.task)} <span>· ${escape(label)}</span></h3><p class="sr-muted">${rows.length} configurations in this selection</p></div><div class="sr-chart-controls" role="group" aria-label="Chart measure">${[['score','Score'],['tokens','Tokens / query'],['tradeoff','Score vs tokens']].map(([value,title])=>`<button class="sr-button ${state.chart===value?'active':''}" data-chart="${value}" aria-pressed="${state.chart===value}">${title}</button>`).join('')}</div></div>
      <div class="sr-chart" id="sr-chart">${chart(rows,label)}</div><div class="sr-legend">${[...new Set(rows.map(r=>r.selection))].map(s=>`<span><i style="background:${palette[s]}"></i>${selectionNames[s]}</span>`).join('')}</div>
      <p class="sr-muted">Select a bar, point or configuration to inspect the run. ${state.chart==='tradeoff'?'Token axis uses a logarithmic scale.':''}</p></section>
      <section class="card"><div class="sr-chart-header"><h3>Results and usage</h3><div class="sr-table-actions">${select('Sort by','sort',[['score','Highest score'],['tokens','Fewest tokens']])}<button class="sr-button" id="sr-export">Export selection ↓</button></div></div>
      <p class="sr-muted">${escape(data.token_definition)} Root context is reported separately. ${costKnown?'Latency reflects shared-server load; it is not a controlled speed comparison.':'Usage is unavailable in this historical summary; it is never inferred from context length.'}</p>
      <div class="sr-table-scroll"><table class="sr-table"><thead><tr><th>Configuration</th><th>${escape(label)}</th><th>Samples</th><th>Tokens / query</th><th>Tokens / run</th><th>Root context<br><small>mean peak tokens</small></th><th>Map calls / query</th><th>Latency / query</th><th>Predictions</th></tr></thead><tbody>${rows.map(r=>`<tr><td><button class="sr-run" data-run="${r.id}">${escape(r.label)}</button><small>${selectionNames[r.selection]}</small><small class="sr-mobile-usage">${r.tokens_per_query==null?'Token use unavailable':fmt(r.tokens_per_query)+' tokens / query'}</small></td><td class="num"><b>${fmt(r.score,2)}%</b></td><td class="num">${r.n}</td><td class="num">${fmt(r.tokens_per_query)}</td><td class="num">${fmt(r.total_tokens)}</td><td class="num">${fmt(r.root_peak_tokens)}</td><td class="num">${fmt(r.map_calls,2)}</td><td class="num">${r.latency_s==null?'—':fmt(r.latency_s,1)+' s'}</td><td>${r.prediction_url?`<a href="${r.prediction_url}" download>CSV ↓</a>`:'—'}</td></tr>`).join('')||'<tr><td colspan="9" class="empty">No completed configurations match these filters. Try All methods or All depths.</td></tr>'}</tbody></table></div></section>
      <dialog class="sr-dialog" id="sr-dialog" aria-labelledby="sr-detail-title"></dialog>`;
    host.querySelectorAll('[data-field]').forEach(el=>el.addEventListener('change',()=>{
      state[el.dataset.field]=el.value;
      if(el.dataset.field==='campaign') {state.selection=['query','geometry'].includes(state.campaign)?'comparison':'all';state.depth='all';}
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
      valid.forEach(r=>{const tip=`${r.label}: ${fmt(r.score,2)}%, ${fmt(r.tokens_per_query)} tokens/query`;svg+=`<circle cx="${x(r.tokens_per_query)}" cy="${y(r.score)}" r="7" fill="${palette[r.selection]}" stroke="white" stroke-width="2" tabindex="0" role="button" aria-label="${escape(tip)}" data-run="${r.id}"><title>${escape(tip)}</title></circle>`;});
      return svg+`<text x="${W/2}" y="${H-13}" text-anchor="middle">Mean model tokens per query · logarithmic scale</text></svg>`;
    }
    const mobile=window.innerWidth<650;
    const W=mobile?640:1120,L=mobile?20:440,R=105,T=30,rowH=mobile?78:36,H=T+valid.length*rowH+40,plotW=W-L-R;
    const value=r=>state.chart==='tokens'?r.tokens_per_query:r.score;
    const max=state.chart==='tokens'?Math.max(1,...valid.map(value)):100;
    let svg=`<svg viewBox="0 0 ${W} ${H}" ${mobile?'style="font-size:24px"':''} role="img" aria-label="${state.chart==='tokens'?'Mean tokens per query':escape(label)} by configuration"><title>${state.chart==='tokens'?'Token usage':escape(label)} by configuration</title>`;
    for(let i=0;i<=4;i++){const v=max*i/4,x=L+plotW*i/4;svg+=`<line x1="${x}" x2="${x}" y1="15" y2="${H-30}" class="sr-grid"/><text x="${x}" y="${H-9}" text-anchor="middle">${state.chart==='tokens'?short(v):fmt(v)+'%'}</text>`;}
    valid.forEach((r,i)=>{
      const v=value(r),y=T+i*rowH+(mobile?25:0),tip=`${r.label}: ${fmt(r.score,2)}%, ${fmt(r.tokens_per_query)} tokens/query`;
      const caption=mobile&&r.label.length>43?r.label.slice(0,42)+'…':r.label;
      svg+=`<g data-run="${r.id}" tabindex="0" role="button" aria-label="${escape(tip)}"><title>${escape(tip)}</title><text x="${mobile?L:L-14}" y="${mobile?y-12:y+16}" text-anchor="${mobile?'start':'end'}">${escape(caption)}</text><rect x="${L}" y="${y}" width="${Math.max(1,v/max*plotW)}" height="24" rx="4" fill="${palette[r.selection]}"/><text x="${L+v/max*plotW+9}" y="${y+20}" font-weight="700">${state.chart==='tokens'?short(v):fmt(v,1)+'%'}</text></g>`;
    });
    return svg+'</svg>';
  }
  function details(id) {
    const r=data.runs.find(r=>r.id===id), dialog=host.querySelector('#sr-dialog');
    const fields=[['Selection',selectionNames[r.selection]],['Reader',r.campaign==='query'||r.campaign==='geometry'?'Question-conditioned map → reduce':'Historical arm; see campaign note'],['Samples',r.n],['Score',fmt(r.score,2)+'% · '+metricName(r)],['Tokens / query',fmt(r.tokens_per_query)],['Tokens / run',fmt(r.total_tokens)],['Mean peak root context',fmt(r.root_peak_tokens)+' tokens'],['Window / overlap',r.window?`${fmt(r.window)} / ${fmt(r.overlap)} characters`:'—'],['Read cap',r.read_chars?fmt(r.read_chars)+' characters':'Full selected span'],['Finding cap',r.finding_chars?fmt(r.finding_chars)+' characters':'—'],['Map grouping',r.granularity||'—'],['Document coverage',r.document_coverage==null?'—':fmt(r.document_coverage*100,1)+'%'],['Abstentions / unfinished / errors',`${fmt(r.abstained)} / ${fmt(r.unfinished)} / ${fmt(r.errors)}`]];
    dialog.innerHTML=`<div class="sr-chart-header"><h3 id="sr-detail-title">${escape(r.label)}</h3><button class="sr-button" id="sr-close" autofocus>Close</button></div><dl class="sr-detail-grid">${fields.map(([key,val])=>`<div><dt>${escape(key)}</dt><dd>${escape(val)}</dd></div>`).join('')}</dl>${r.run_name?`<details><summary>Source run identifier</summary><code class="sr-run-id">${escape(r.run_name)}</code></details>`:''}${r.prediction_url?`<p><a href="${r.prediction_url}" download>Download all ${r.n} predictions and per-query token counts ↓</a></p><h4>First three predictions</h4>${r.preview.map(p=>`<details class="sr-pred"><summary>${escape(p.id)} · ${fmt(p.tokens)} tokens</summary><p><b>Reference:</b> ${escape(p.answers)}</p><pre>${escape(p.prediction)}</pre></details>`).join('')}`:'<p class="sr-muted">Per-example artifacts are unavailable in this historical summary export.</p>'}`;
    dialog.querySelector('#sr-close').onclick=()=>dialog.close();
    dialog.showModal();
  }
  function exportRows(rows) {
    const fields=['campaign','task','label','selection','metric','score','n','tokens_per_query','total_tokens','root_peak_tokens','latency_s','k','window','read_chars','finding_chars','hops','document_coverage','map_calls','abstained','unfinished','errors'];
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
      if(!data.campaigns.some(c=>c.id===state.campaign))state.campaign='query';
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

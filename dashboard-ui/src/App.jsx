import React, { useEffect, useMemo, useState } from 'react';
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, LabelList, Legend,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import {
  Activity, ArrowDownRight, ArrowUpRight, BarChart3, CheckCircle2,
  ChevronRight, CircleHelp, Disc3, Headphones, Layers3, Menu, Music2,
  RefreshCcw, Search, Sparkles, TrendingUp, X,
} from 'lucide-react';

const API = '/api';
const fmt = n => Number(n ?? 0).toLocaleString('en-IN');
const score = n => Number(n ?? 0).toFixed(2);
const COLORS = ['#83e0ba', '#9c92ff', '#f2bb81'];
const menu = [
  ['overview', 'Overview', BarChart3],
  ['genres', 'Genre intelligence', Disc3],
  ['tracks', 'Top tracks & artists', Music2],
  ['models', 'ML comparison', Activity],
  ['charts', 'Chart gallery', Layers3],
];

function Panel({title, subtitle, action, children, className = ''}) {
  return <section className={`panel ${className}`}><header className="panel-header"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</header>{children}</section>;
}
function Metric({icon: Icon, label, value, note, accent = 'mint'}) {
  return <div className={`metric ${accent}`}><div className="metric-top"><span>{label}</span><div className="metric-icon"><Icon size={18}/></div></div><strong>{value}</strong><div className="metric-note">{note}</div></div>;
}
function Loading() {return <div className="loading"><Disc3 size={28} className="spin"/><p>Loading results from your Spark pipeline…</p></div>}
function Empty({message}) {return <div className="empty"><CircleHelp size={20}/>{message}</div>}
function ChartTooltip({active, payload, label}) {
  if (!active || !payload?.length) return null;
  return <div className="tooltip"><strong>{label}</strong>{payload.map((p,i)=><span key={i}>{p.name || 'Value'}: {typeof p.value === 'number' ? score(p.value) : p.value}</span>)}</div>;
}
function HorizontalBars({data, keyName='avgPopularity', label='Avg popularity'}) {
  return <ResponsiveContainer width="100%" height={Math.max(280, data.length*39)}><BarChart data={data} layout="vertical" margin={{top:5,right:28,left:16,bottom:0}}><CartesianGrid horizontal={false} stroke="#263047" strokeDasharray="3 6"/><XAxis type="number" tick={{fill:'#8797b3',fontSize:11}} domain={[0,'dataMax + 10']}/><YAxis type="category" width={84} dataKey="name" tick={{fill:'#cbd6e9',fontSize:12}}/><Tooltip content={<ChartTooltip/>}/><Bar dataKey={keyName} name={label} fill="#81dec0" radius={[0,6,6,0]} barSize={13}/></BarChart></ResponsiveContainer>;
}
function Overview({data, models}) {
  const dataset=data.dataset;
  const gbt=models.find(x=>x.model.toLowerCase().includes('boost'));
  const total=data.popularityDistribution.reduce((n,r)=>n+r.count,0);
  return <>
    <div className="stats">
      <Metric icon={Disc3} label="Cleaned records" value={fmt(dataset.cleanedRows)} note="Genre-assignment rows"/>
      <Metric icon={Music2} label="Unique tracks" value={fmt(dataset.uniqueTracks)} note="Distinct Spotify track IDs" accent="violet"/>
      <Metric icon={Layers3} label="Genre labels" value={fmt(dataset.genreCount)} note="Genres represented" accent="gold"/>
      <Metric icon={TrendingUp} label="GBT test R²" value={gbt ? score(gbt.r2) : '—'} note={gbt ? 'Measured on held-out test set' : 'Waiting for model results'} accent="cyan"/>
    </div>
    <div className="two-col">
      <Panel title="Popularity landscape" subtitle="Unique-track distribution (not repeated genre rows)" action={<span className="tiny-pill">{fmt(total)} tracks</span>}>
        <div className="chart-area"><ResponsiveContainer width="100%" height={280}><BarChart data={data.popularityDistribution} margin={{top:12,right:10,bottom:0,left:0}}><CartesianGrid stroke="#243049" vertical={false} strokeDasharray="3 7"/><XAxis dataKey="category" tickLine={false} axisLine={false} tick={{fill:'#c1cde0',fontSize:12}}/><YAxis width={50} tickLine={false} axisLine={false} tick={{fill:'#8190ab',fontSize:11}} tickFormatter={n=>n>=1000?`${Math.round(n/1000)}k`:n}/><Tooltip formatter={v=>[fmt(v),'Tracks']} cursor={{fill:'#ffffff08'}} contentStyle={{background:'#161d30',border:'1px solid #35415a',color:'#e8eefb'}}/><Bar dataKey="count" radius={[7,7,0,0]} maxBarSize={74}>{data.popularityDistribution.map((r,i)=><Cell fill={COLORS[i]} key={r.category}/>)}</Bar></BarChart></ResponsiveContainer></div>
        <div className="chart-foot">Distribution recalculated from unique tracks by the exporter.</div>
      </Panel>
      <Panel title="Genre spotlight" subtitle="Average popularity by genre · minimum 25 tracks"><HorizontalBars data={data.genres.slice(0,6)}/></Panel>
    </div>
    <div className="two-col">
      <Panel title="Audio features vs popularity" subtitle="Pearson correlations on unique tracks"><div className="correlation-list">{data.correlations.slice(0,7).map(row=><div className="corr-row" key={row.feature}><span>{row.feature}</span><div className="corr-track"><div className={`corr-fill ${row.value<0?'negative':''}`} style={{width:`${Math.abs(row.value)*100}%`}}/></div><b>{row.value.toFixed(4)}</b></div>)}</div><p className="muted-note">Bar length reflects the correlation magnitude (0–1); weak relationships appear short.</p></Panel>
      <Panel title="Pipeline health" subtitle="What this dashboard is actually showing"><div className="pipeline"><div><CheckCircle2 size={19}/><span>Apache Spark processing</span><em>Offline batch</em></div><div><CheckCircle2 size={19}/><span>Track-level deduplication</span><em>Applied</em></div><div><CheckCircle2 size={19}/><span>Charts and SQL analyses</span><em>Exported</em></div><div><CheckCircle2 size={19}/><span>ML results</span><em>Read from CSV</em></div></div><div className="notice">This dashboard reads outputs from your existing Spark run. It does not pretend to train models in the browser.</div></Panel>
    </div>
  </>;
}
function Genres({data}) {return <><div className="section-copy"><h2>Genre intelligence</h2><p>Compare genre-level popularity using the complete cleaned genre-assignment dataset.</p></div><div className="two-col"><Panel title="Average genre popularity" subtitle="Top 12 genres · at least 25 genre-assignment rows"><HorizontalBars data={data.genres}/></Panel><Panel title="Genre statistics" subtitle="Exact counts from Spark groupBy aggregation"><div className="table-wrap"><table><thead><tr><th>Genre</th><th>Tracks</th><th>Avg popularity</th></tr></thead><tbody>{data.genres.map(row=><tr key={row.name}><td>{row.name}</td><td>{fmt(row.tracks)}</td><td><span className="score-pill">{score(row.avgPopularity)}</span></td></tr>)}</tbody></table></div></Panel></div></>}
function Tracks({data}) {
  const [filter, setFilter] = useState('');
  const shown=data.topTracks.filter(x=>(`${x.name} ${x.artists}`).toLowerCase().includes(filter.toLowerCase()));
  return <><div className="section-copy"><h2>Tracks & artists</h2><p>Track rankings use unique Spotify track IDs. Artist statistics split collaborative credits and require at least five tracks.</p></div><div className="two-col"><Panel title="Most popular tracks" subtitle="Unique songs sorted by popularity" action={<div className="search-box"><Search size={15}/><input value={filter} onChange={e=>setFilter(e.target.value)} placeholder="Find track or artist"/></div>}><div className="tracks">{shown.length?shown.map((row,i)=><div className="track" key={row.id}><div className="track-number">{i+1}</div><div className="track-cover"><Music2 size={18}/></div><div className="track-desc"><strong>{row.name}</strong><small>{row.artists}</small></div><div className="track-score">{row.popularity}</div></div>):<Empty message="No tracks match this search."/>}</div></Panel><Panel title="Top artists by average popularity" subtitle="At least five unique tracks per credited artist"><HorizontalBars data={data.artists}/></Panel></div></>;
}
function Models({models}) {
  if(!models.length) return <Empty message="No model results found. Run your Spark training pipeline to generate model_comparison.csv."/>;
  const best = [...models].sort((a,b)=>a.rmse-b.rmse)[0];
  return <><div className="section-copy"><h2>Model lab</h2><p>Compare held-out test metrics exported by your Spark MLlib pipeline. Lower RMSE/MAE and higher R² are preferable.</p></div><div className="stats model-stats"><Metric icon={ArrowDownRight} label="Lowest test RMSE" value={score(best.rmse)} note={best.model}/><Metric icon={TrendingUp} label="Test R² of lowest-RMSE model" value={best.r2.toFixed(4)} note="Predictive power remains limited" accent="violet"/><Metric icon={Layers3} label="Models evaluated" value={models.length} note="Same dataset and train/test split" accent="gold"/></div><div className="two-col"><Panel title="Test-set RMSE" subtitle="Lower means smaller typical prediction errors"><ResponsiveContainer width="100%" height={290}><BarChart data={models} margin={{top:20,right:12,left:0,bottom:12}}><CartesianGrid stroke="#243049" vertical={false} strokeDasharray="3 7"/><XAxis dataKey="model" interval={0} tick={{fontSize:10,fill:'#c2cce0'}} tickFormatter={n=>n.includes('Gradient')?'GBT':n.includes('Forest')?'Random Forest':'Linear'}/><YAxis tick={{fill:'#8a9ab6',fontSize:11}}/><Tooltip content={<ChartTooltip/>}/><Bar dataKey="rmse" name="RMSE" fill="#9d91ff" radius={[7,7,0,0]} barSize={48}><LabelList dataKey="rmse" position="top" formatter={v=>v.toFixed(2)} fill="#e0e7fa" fontSize={12}/></Bar></BarChart></ResponsiveContainer></Panel><Panel title="Test-set R²" subtitle="Relative variance explained"><ResponsiveContainer width="100%" height={290}><BarChart data={models} margin={{top:20,right:12,left:0,bottom:12}}><CartesianGrid stroke="#243049" vertical={false} strokeDasharray="3 7"/><XAxis dataKey="model" interval={0} tick={{fontSize:10,fill:'#c2cce0'}} tickFormatter={n=>n.includes('Gradient')?'GBT':n.includes('Forest')?'Random Forest':'Linear'}/><YAxis tick={{fill:'#8a9ab6',fontSize:11}}/><Tooltip content={<ChartTooltip/>}/><Bar dataKey="r2" name="R²" fill="#7dddbb" radius={[7,7,0,0]} barSize={48}><LabelList dataKey="r2" position="top" formatter={v=>v.toFixed(3)} fill="#e0e7fa" fontSize={12}/></Bar></BarChart></ResponsiveContainer></Panel></div><Panel title="Full model comparison" subtitle="Numbers loaded directly from outputs/results/model_comparison.csv"><div className="table-wrap"><table><thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead><tbody>{models.map(x=><tr key={x.model}><td>{x.model}</td><td>{x.rmse.toFixed(4)}</td><td>{x.mae.toFixed(4)}</td><td>{x.r2.toFixed(4)}</td></tr>)}</tbody></table></div><p className="muted-note">These metrics indicate test-split performance, not guaranteed accuracy on future songs.</p></Panel></>;
}
function Charts({charts}) {return <><div className="section-copy"><h2>Chart gallery</h2><p>These PNGs are served directly from your existing <code>outputs/charts</code> folder.</p></div>{charts.length?<div className="gallery">{charts.map(item=><Panel title={item.name} key={item.filename}><a href={item.url} target="_blank" rel="noreferrer"><img loading="lazy" src={item.url} alt={item.name}/></a></Panel>)}</div>:<Empty message="No PNG charts found. Run your Spark project to generate outputs/charts first."/>}</>}
export default function App() {
  const [tab,setTab] = useState('overview');
  const [data,setData] = useState(null);
  const [models,setModels] = useState([]);
  const [charts,setCharts] = useState([]);
  const [error,setError] = useState('');
  const [busy,setBusy] = useState(true);
  const [open,setOpen] = useState(false);
  async function refresh(){
    setBusy(true);setError('');
    try {
      const [r, m, c] = await Promise.all([fetch(`${API}/dashboard`),fetch(`${API}/models`),fetch(`${API}/charts`)]);
      if(!r.ok){const body=await r.json().catch(()=>({detail:'Dashboard unavailable'}));throw new Error(body.detail||'Dashboard data unavailable')}
      const d=await r.json();const mm=m.ok?await m.json():[];const cc=c.ok?await c.json():[];
      setData(d);setModels(mm);setCharts(cc);
    } catch(e){setError(e.message)} finally {setBusy(false)}
  }
  useEffect(()=>{refresh()},[]);
  const current=menu.find(x=>x[0]===tab);
  return <div className="app">
    <aside className={`sidebar ${open?'open':''}`}>
      <div className="logo"><div className="logo-mark"><Activity size={22}/></div><div><b>BEYOND<span>THE</span>BEAT</b><small>SPOTIFY DATA INTELLIGENCE</small></div><button className="mobile-close" onClick={()=>setOpen(false)} aria-label="Close menu"><X/></button></div>
      <div className="nav-label">WORKSPACE</div>
      <nav>{menu.map(([id,label,Icon])=><button key={id} className={`nav-item ${tab===id?'active':''}`} onClick={()=>{setTab(id);setOpen(false)}}><Icon size={18}/><span>{label}</span>{tab===id&&<ChevronRight size={15} className="nav-arrow"/>}</button>)}</nav>
      <div className="side-bottom"><div className="side-orbit"><Sparkles size={17}/><div><strong>Powered by PySpark</strong><span>Real analysis. Real metrics.</span></div></div><p>Semester VII · Big Data Analytics</p></div>
    </aside>
    {open&&<div className="backdrop" onClick={()=>setOpen(false)}/>}
    <main className="main"><header className="topbar"><div className="top-left"><button className="mobile-menu" onClick={()=>setOpen(true)} aria-label="Open menu"><Menu/></button><span className="breadcrumb">Workspace <ChevronRight size={13}/> <b>{current?.[1]}</b></span></div><div className="top-actions"><div className="status"><span className={`status-dot ${error?'down':''}`}/>{error?'Data unavailable':'Local Spark snapshot'}</div><button className="refresh" onClick={refresh} disabled={busy}><RefreshCcw size={15} className={busy?'spin':''}/> Refresh data</button></div></header>
      <div className="content"><div className="hero"><div className="hero-copy"><div className="eyebrow"><span className="live-dot"/>DATA ANALYTICS DASHBOARD</div><h1>See the story <span>behind the sound.</span></h1><p>Explore Spotify listening data through genre intelligence, audio-feature analytics and Spark MLlib experiments.</p>{data&&<div className="hero-tags"><span><CheckCircle2 size={14}/> {fmt(data.dataset.uniqueTracks)} unique tracks</span><span><Activity size={14}/> Spark MLlib</span><span>Snapshot: {new Date(data.generatedAt).toLocaleString()}</span></div>}</div><div className="hero-disc"><div className="disc-ring"><div className="disc-inner"><Disc3 size={46}/></div></div></div></div>
      {busy?<Loading/>:error?<div className="error"><CircleHelp size={24}/><div><h3>Connect your real project data</h3><p>{error}</p><small>Check the README. Run the export function in main.py, then restart/refresh the dashboard.</small></div><button onClick={refresh}>Retry</button></div>:<>
        {tab==='overview'&&<Overview data={data} models={models}/>}
        {tab==='genres'&&<Genres data={data}/>}
        {tab==='tracks'&&<Tracks data={data}/>}
        {tab==='models'&&<Models models={models}/>}
        {tab==='charts'&&<Charts charts={charts}/>}
      </>}
      <footer className="footer">Beyond the Beat <span>•</span> Built with React, FastAPI & Apache Spark <span>•</span> Batch analytics dashboard</footer></div>
    </main>
  </div>;
}

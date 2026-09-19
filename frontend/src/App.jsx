function Advisor() {
  const [cells, setCells] = useState([]);
  const [selectedId, setSelectedId] = useState('');
  const [prompt, setPrompt] = useState('');
  const [answer, setAnswer] = useState(null);
  const [advisorError, setAdvisorError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const cellState = useAsync(() => fetchAllCells(), []);
  const [cellSearch, setCellSearch] = useState('');
  const filteredCells = useMemo(() => {
    const term = cellSearch.trim().toLowerCase();
    if (!term) return cells.slice(0, 12);
    return cells.filter((cell) => cell.cell_id.toLowerCase().includes(term)).slice(0, 12);
  }, [cells, cellSearch]);
  useEffect(() => {
    if (cellState.data) {
      setCells(cellState.data);
      setSelectedId(cellState.data[0]?.cell_id || '');
    }
  }, [cellState.data]);
  const detail = useAsync(() => selectedId ? api.cell(selectedId) : Promise.resolve(null), [selectedId]);
  const selected = detail.data;
  useEffect(() => {
    setAnswer(null);
    setAdvisorError(null);
  }, [selectedId]);
  const ask = async (question = prompt) => {
    const trimmedQuestion = question.trim();
    if (!selectedId || !trimmedQuestion || submitting) return;
    setPrompt(trimmedQuestion);
    setSubmitting(true);
    setAdvisorError(null);
    try {
      setAnswer(await api.advisor({ cellId: selectedId, question: trimmedQuestion }));
    } catch (error) {
      setAdvisorError(error.message || 'AI Advisor is temporarily unavailable.');
    } finally {
      setSubmitting(false);
    }
  };
  const questions = ['Why is this cell high risk?', 'What cooling action should be considered?', 'What do these environmental values mean?', 'Explain this location in simple terms.'];
  return <div className="page advisor-page"><SectionHeading eyebrow="Grounded intelligence" title="Ask UrbanCool" copy="AI responses are grounded in the selected UrbanCool cell's satellite-derived metrics and cooling recommendations." /><div className="advisor-layout"><div className="advisor-chat panel"><div className="chat-head"><span className="status-avatar"><BrainCircuit size={19} /></span><div><strong>UrbanCool advisor</strong><small>Grounded AI Advisor · selected-cell evidence</small></div><span className="online-tag">READY</span></div><div className="chat-body"><div className="message assistant"><span className="message-avatar"><Sparkles size={14} /></span><div>Choose any real UrbanCool cell and ask about its observed signals or attached recommendations.</div></div>{answer && <div className="message assistant"><span className="message-avatar"><Sparkles size={14} /></span><div><small className="answer-meta">{answer.provider} · grounded in {answer.cell_id}</small><div className="advisor-answer">{answer.answer}</div></div></div>}{submitting && <div className="message assistant advisor-thinking"><span className="message-avatar"><Sparkles size={14} /></span><div>Reviewing the selected cell's evidence…</div></div>}{advisorError && <div className="advisor-error"><CircleHelp size={16} /><span>{advisorError}</span></div>}<div className="prompt-grid">{questions.map((item) => <button key={item} disabled={submitting || !selectedId} onClick={() => ask(item)}>{item}<ArrowRight size={14} /></button>)}</div></div><div className="chat-input"><input value={prompt} disabled={submitting || !selectedId} onChange={(event) => setPrompt(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') ask(); }} placeholder="Ask about this location…" /><button disabled={submitting || !selectedId || !prompt.trim()} onClick={() => ask()} aria-label="Send prompt">{submitting ? <RefreshCw className="spin-icon" size={17} /> : <ArrowRight size={17} />}</button></div></div><div className="advisor-context panel"><div className="panel-kicker"><MapPinned size={16} /> GROUNDED IN SELECTED CELL</div><h3>Choose a real cell</h3><select value={selectedId} onChange={(event) => setSelectedId(event.target.value)}><option value="">Select cell</option>{cells.map((cell) => <option key={cell.cell_id} value={cell.cell_id}>{cell.cell_id} · {format(cell.risk.score)}</option>)}</select>{detail.loading ? <LoadingState /> : selected && <div className="context-readout"><div className={`risk-chip ${selected.risk.class.toLowerCase()}`}>{selected.risk.class} risk</div><div className="context-score">{format(selected.risk.score)}<small>official baseline score</small></div><div className="context-metrics"><MiniMetric label="LST" value={`${format(selected.metrics.lst_median_c)}°C`} /><MiniMetric label="NDVI" value={format(selected.metrics.ndvi_median, 2)} /><MiniMetric label="NDBI" value={format(selected.metrics.ndbi_median, 2)} /></div><Link to={`/cell/${selected.cell_id}`} className="text-link">Open full analysis <ArrowRight size={15} /></Link></div>}</div></div></div>;
}
import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, NavLink, Navigate, Route, Routes, useLocation, useNavigate, useParams } from 'react-router-dom';
import {
  Activity, ArrowRight, BarChart3, BookOpen, BrainCircuit, ChevronRight, CircleHelp,
  Compass, Droplets, ExternalLink, Flame, Globe2, Leaf, Menu, Map as MapIcon, MapPinned,
  MousePointer2, Navigation, PanelLeftClose, RefreshCw, Search, ShieldCheck, Sparkles,
  Thermometer, Trees, X, Zap,
} from 'lucide-react';
import { CircleMarker, MapContainer, TileLayer, Tooltip, useMap } from 'react-leaflet';
import { api, fetchAllCells } from './services/api';

const navGroups = [
  { label: 'Overview', items: [{ to: '/dashboard', label: 'Dashboard', icon: Compass }, { to: '/map', label: 'Heat Map', icon: MapIcon }, { to: '/insights', label: 'Insights', icon: BarChart3 }, { to: '/recommendations', label: 'Recommendations', icon: Leaf }, { to: '/advisor', label: 'AI Advisor', icon: BrainCircuit }] },
  { label: 'Trust & Context', items: [{ to: '/methodology', label: 'Methodology', icon: BookOpen }] },
];

const riskColors = { Low: '#55d6be', Medium: '#f0b35b', High: '#ff6f78' };
const format = (value, digits = 1) => Number(value).toFixed(digits);

function Reveal({ children, className = '', delay = 0, once = true }) {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const element = ref.current;
    if (!element) return undefined;
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setVisible(true);
        if (once) observer.unobserve(element);
      } else if (!once) setVisible(false);
    }, { threshold: 0.12, rootMargin: '0px 0px -8% 0px' });
    observer.observe(element);
    return () => observer.disconnect();
  }, [once]);
  return <div ref={ref} className={`reveal ${visible ? 'is-visible' : ''} ${className}`} style={{ '--reveal-delay': `${delay}ms` }}>{children}</div>;
}

function useCountUp(value, duration = 900) {
  const [displayValue, setDisplayValue] = useState(0);
  useEffect(() => {
    const target = Number(value);
    if (!Number.isFinite(target)) return undefined;
    let frame;
    const settle = window.setTimeout(() => setDisplayValue(target), duration + 80);
    const start = performance.now();
    const tick = (now) => {
      const progress = Math.min(1, (now - start) / duration);
      const eased = 1 - ((1 - progress) ** 3);
      setDisplayValue(target * eased);
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => { cancelAnimationFrame(frame); window.clearTimeout(settle); };
  }, [value, duration]);
  return displayValue;
}

function usePointerParallax() {
  const ref = useRef(null);
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return undefined;
    let frame;
    const handleMove = (event) => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const x = (event.clientX / window.innerWidth - 0.5) * 8;
        const y = (event.clientY / window.innerHeight - 0.5) * 8;
        ref.current?.style.setProperty('--pointer-x', `${x}px`);
        ref.current?.style.setProperty('--pointer-y', `${y}px`);
      });
    };
    window.addEventListener('pointermove', handleMove, { passive: true });
    return () => { cancelAnimationFrame(frame); window.removeEventListener('pointermove', handleMove); };
  }, []);
  return ref;
}

function App() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const location = useLocation();
  useEffect(() => setDrawerOpen(false), [location.pathname]);
  return (
    <div className="app-shell">
      <button className="mobile-menu" aria-label="Open navigation" onClick={() => setDrawerOpen(true)}><Menu size={20} /></button>
      <aside className={`sidebar ${drawerOpen ? 'is-open' : ''}`}>
        <div className="brand"><span className="brand-mark"><Thermometer size={18} /></span><span>URBAN<span>COOL</span></span></div>
        <div className="brand-subtitle">Urban heat intelligence</div>
        <button className="drawer-close" onClick={() => setDrawerOpen(false)} aria-label="Close navigation"><X size={18} /></button>
        <nav>
          {navGroups.map((group) => <div className="nav-group" key={group.label}><div className="nav-label">{group.label}</div>{group.items.map(({ to, label, icon: Icon }) => <NavLink key={to} to={to} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}><Icon size={17} /><span>{label}</span>{label === 'AI Advisor' && <span className="nav-badge">BETA</span>}</NavLink>)}</div>)}
        </nav>
        <div className="sidebar-footer"><div className="live-dot" /> <span>Earth Engine-derived dataset</span></div>
      </aside>
      {drawerOpen && <button className="drawer-backdrop" onClick={() => setDrawerOpen(false)} aria-label="Close navigation" />}
      <main className="main-shell"><PageHeader /><div className="route-stage"><Routes><Route path="/" element={<Home />} /><Route path="/dashboard" element={<Dashboard />} /><Route path="/map" element={<HeatMap />} /><Route path="/cell/:cellId" element={<CellAnalysis />} /><Route path="/insights" element={<Insights />} /><Route path="/recommendations" element={<Recommendations />} /><Route path="/advisor" element={<Advisor />} /><Route path="/methodology" element={<Methodology />} /><Route path="*" element={<Navigate to="/" replace />} /></Routes></div></main>
    </div>
  );
}

function PageHeader() {
  const location = useLocation();
  const titles = { '/dashboard': 'Overview', '/map': 'Heat map', '/insights': 'Insights', '/recommendations': 'Recommendations', '/advisor': 'AI advisor', '/methodology': 'Methodology' };
  if (location.pathname === '/') return null;
  const title = titles[location.pathname] || 'Cell analysis';
  return <header className="topbar"><div><h1>{title}</h1></div><div className="topbar-status"><span className="live-dot" /> Dataset snapshot <span className="status-divider" /> <span className="muted">Landsat 8 / 9 · 2022–2025</span></div></header>;
}

function Home() {
  const parallaxRef = usePointerParallax();
  return <div ref={parallaxRef} className="home-page"><div className="home-orbit orbit-one" /><div className="home-orbit orbit-two" /><div className="home-grid" /><header className="home-nav"><Link to="/" className="brand"><span className="brand-mark"><Thermometer size={18} /></span><span>URBAN<span>COOL</span></span></Link><Link to="/methodology" className="text-link">How it works <ArrowRight size={15} /></Link></header><section className="hero"><div className="hero-copy"><div className="eyebrow hero-eyebrow"><span className="live-dot" /> Satellite intelligence / Coimbatore</div><h1>See where your city is heating up.<em> Discover how to cool it down.</em></h1><p>UrbanCool turns Earth observation into a clearer view of urban heat risk, then connects each location to practical, explainable cooling actions.</p><div className="hero-actions"><Link to="/dashboard" className="button button-primary">Explore UrbanCool <ArrowRight size={17} /></Link><Link to="/map" className="button button-ghost">Explore the heat map <MapPinned size={17} /></Link></div><div className="hero-proof"><ShieldCheck size={16} /><span>Real Landsat-derived signals</span><span className="proof-divider" /><span>500 m grid analysis</span></div></div><div className="hero-visual"><div className="visual-label"><span>LIVE OBSERVATION</span><strong>urban heat field</strong></div><div className="visual-scan" /><div className="heat-rings"><div /><div /><div /><div /></div><div className="visual-pin pin-a"><span>48.7°</span></div><div className="visual-pin pin-b"><span>36.2°</span></div><div className="visual-pin pin-c"><span>42.9°</span></div><div className="visual-coordinates">11°01' N / 76°57' E</div></div></section><section className="capability-row"><Capability icon={Globe2} title="Real Earth Observation" copy="Landsat signals, processed transparently." /><Capability icon={Flame} title="Heat Risk Intelligence" copy="A sharper view of where exposure concentrates." /><Capability icon={Trees} title="Cooling Recommendations" copy="Location-specific actions, clearly explained." /><Capability icon={Sparkles} title="AI-Assisted Insights" copy="A grounded foundation for future intelligence." /></section></div>;
}

function Capability({ icon: Icon, title, copy }) { return <div className="capability"><Icon size={19} /><div><strong>{title}</strong><span>{copy}</span></div></div>; }

function useAsync(loader, deps = []) {
  const [state, setState] = useState({ loading: true, data: null, error: null });
  const reload = () => { setState({ loading: true, data: null, error: null }); loader().then((data) => setState({ loading: false, data, error: null })).catch((error) => setState({ loading: false, data: null, error })); };
  useEffect(reload, deps); // eslint-disable-line react-hooks/exhaustive-deps
  return { ...state, reload };
}

function AsyncState({ state, children }) { if (state.loading) return <LoadingState />; if (state.error) return <ErrorState error={state.error} retry={state.reload} />; return children(state.data); }
function LoadingState() { return <div className="loading-state"><div className="spinner" /><span>Connecting to UrbanCool data…</span></div>; }
function ErrorState({ error, retry }) { return <div className="error-state"><CircleHelp size={22} /><strong>Data connection unavailable</strong><span>{error?.message || 'The API did not respond.'}</span><button className="button button-ghost" onClick={retry}><RefreshCw size={15} /> Retry</button></div>; }
function StatCard({ label, value, suffix = '', accent = 'cyan', decimal = 0 }) { const animatedValue = useCountUp(value); return <div className={`stat-card accent-${accent}`}><span>{label}</span><strong>{typeof value === 'number' ? animatedValue.toLocaleString(undefined, { maximumFractionDigits: decimal }) : value}{suffix}</strong><div className="stat-line" /></div>; }
function SectionHeading({ eyebrow, title, copy, action }) { return <div className="section-heading"><div><span className="eyebrow">{eyebrow}</span><h2>{title}</h2>{copy && <p>{copy}</p>}</div>{action}</div>; }

function Dashboard() {
  const state = useAsync(api.statistics, []);
  return <AsyncState state={state}>{(stats) => <div className="page dashboard-page"><SectionHeading eyebrow="Operational overview" title="Coimbatore Urban Heat Intelligence" copy="A snapshot of Earth Engine-derived cells across the study area." action={<Link className="button button-primary" to="/map">Open heat map <ArrowRight size={16} /></Link>} /><div className="stat-grid">{[[stats.total_cells, 'cyan', 'Cells analyzed'], [stats.high_risk_cells, 'coral', 'High risk cells'], [stats.medium_risk_cells, 'amber', 'Medium risk cells'], [stats.low_risk_cells, 'mint', 'Low risk cells'], [stats.average_lst, 'coral', 'Average surface temp', '°C', 1], [stats.average_ndvi, 'mint', 'Average NDVI', '', 2], [stats.average_ndbi, 'blue', 'Average NDBI', '', 2]].map(([value, accent, label, suffix = '', decimal = 0], index) => <Reveal key={label} delay={index * 90}><StatCard label={label} value={value} suffix={suffix} decimal={decimal} accent={accent} /></Reveal>)}</div><div className="dashboard-grid"><Reveal><RiskDistribution stats={stats} /></Reveal><Reveal delay={130}><div className="signal-panel"><div className="panel-kicker"><Activity size={16} /> PRODUCT FLOW</div><h3>From satellite signal to cooling action.</h3><p>Observed signals move through risk analysis and deterministic recommendation logic.</p><ProductFlow /></div></Reveal></div></div>}</AsyncState>;
}
function RiskDistribution({ stats }) { const total = stats.total_cells; return <div className="panel risk-panel"><div className="panel-kicker"><Flame size={16} /> RISK DISTRIBUTION</div><div className="risk-panel-head"><div><h3>Heat-risk footprint</h3><p>Percentile-based baseline classes</p></div><span className="panel-number">{(stats.high_risk_cells / total * 100).toFixed(1)}% <small>high</small></span></div><div className="risk-bar"><span style={{ width: `${stats.low_risk_cells / total * 100}%`, background: riskColors.Low }} /><span style={{ width: `${stats.medium_risk_cells / total * 100}%`, background: riskColors.Medium }} /><span style={{ width: `${stats.high_risk_cells / total * 100}%`, background: riskColors.High }} /></div><div className="legend-list"><Legend color={riskColors.Low} label="Low" value={stats.low_risk_cells} /><Legend color={riskColors.Medium} label="Medium" value={stats.medium_risk_cells} /><Legend color={riskColors.High} label="High" value={stats.high_risk_cells} /></div></div>; }
function Legend({ color, label, value }) { return <div className="legend-item"><span className="legend-dot" style={{ background: color }} /><span>{label}</span><strong>{value.toLocaleString()}</strong></div>; }
function ProductFlow() { return <div className="product-flow">{['Satellite-derived data', 'Environmental signals', 'Heat-risk analysis', 'Machine learning', 'Cooling recommendations'].map((item, index) => <div className="flow-step" key={item} style={{ '--flow-delay': `${index * 110}ms` }}><span>{String(index + 1).padStart(2, '0')}</span><strong>{item}</strong>{index < 4 && <ChevronRight size={15} />}</div>)}</div>; }

function HeatMap() {
  const [riskClass, setRiskClass] = useState('');
  const [selected, setSelected] = useState(null);
  const [query, setQuery] = useState('');
  const state = useAsync(() => fetchAllCells({ riskClass: riskClass || undefined }), [riskClass]);
  useEffect(() => setSelected(null), [riskClass]);
  return <AsyncState state={state}>{(cells) => <div className="page map-page"><SectionHeading eyebrow="Spatial intelligence" title="Heat map" copy="Every point is a real 500 m cell from the Coimbatore study area." action={<div className="map-count"><span className="live-dot" /> {cells.length.toLocaleString()} dataset cells</div>} /><div className="map-toolbar"><div className="search-box"><Search size={17} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search by cell ID" /></div><select value={riskClass} onChange={(event) => setRiskClass(event.target.value)}><option value="">All risk classes</option><option value="High">High risk</option><option value="Medium">Medium risk</option><option value="Low">Low risk</option></select><span className="toolbar-note"><MousePointer2 size={15} /> Select a cell for details</span></div><div className="map-layout"><HeatMapCanvas cells={cells} query={query} selected={selected} onSelect={setSelected} /><MapPreview cell={selected} /></div></div>}</AsyncState>;
}
function HeatMapCanvas({ cells, query, selected, onSelect }) { const mapCells = useMemo(() => { const matching = query ? cells.filter((cell) => cell.cell_id.toLowerCase().includes(query.toLowerCase())) : cells; const limit = 1760; if (matching.length <= limit) return matching; const stride = Math.ceil(matching.length / limit); return matching.filter((_, index) => index % stride === 0); }, [cells, query]); const center = [10.84, 77.07]; return <div className="map-wrap"><MapContainer center={center} zoom={9} scrollWheelZoom className="leaflet-map"><TileLayer attribution='&copy; OpenStreetMap contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />{mapCells.map((cell) => <CircleMarker key={cell.cell_id} center={[cell.location.latitude, cell.location.longitude]} radius={selected?.cell_id === cell.cell_id ? 8 : 4.5} bubblingMouseEvents={false} pathOptions={{ color: riskColors[cell.risk.class], fillColor: riskColors[cell.risk.class], fillOpacity: 0.68, weight: selected?.cell_id === cell.cell_id ? 3 : 1 }} eventHandlers={{ click: (event) => { event.originalEvent?.stopPropagation(); onSelect(cell); } }}><Tooltip>{cell.cell_id} · {format(cell.risk.score)} score</Tooltip></CircleMarker>)}</MapContainer><div className="map-legend"><strong>Risk level</strong><Legend color={riskColors.High} label="High" value="" /><Legend color={riskColors.Medium} label="Medium" value="" /><Legend color={riskColors.Low} label="Low" value="" /></div><div className="map-note">{cells.length.toLocaleString()} cells in dataset · {mapCells.length.toLocaleString()} shown on map for performance</div></div>; }
function MapPreview({ cell }) { const detailState = useAsync(() => cell ? api.cell(cell.cell_id) : Promise.resolve(null), [cell?.cell_id]); if (!cell) return <div className="map-preview empty-preview"><MapPinned size={25} /><strong>Select a cell</strong><span>Click any point on the map to inspect its risk signal.</span></div>; if (detailState.loading || !detailState.data) return <div className="map-preview"><LoadingState /></div>; if (detailState.error) return <div className="map-preview"><ErrorState error={detailState.error} retry={detailState.reload} /></div>; const detail = detailState.data; return <div className="map-preview"><div className="preview-top"><span className="eyebrow">SELECTED CELL</span><span className={`risk-chip ${detail.risk.class.toLowerCase()}`}>{detail.risk.class} risk</span></div><h3>{detail.cell_id}</h3><div className="preview-score"><strong>{format(detail.risk.score)}</strong><span>/ 100 baseline score</span></div><div className="mini-metrics"><MiniMetric label="LST" value={`${format(detail.metrics.lst_median_c)}°C`} /><MiniMetric label="NDVI" value={format(detail.metrics.ndvi_median, 2)} /><MiniMetric label="NDBI" value={format(detail.metrics.ndbi_median, 2)} /></div><Link className="button button-primary full-button" to={`/cell/${detail.cell_id}`}>Open full analysis <ArrowRight size={16} /></Link></div>; }
function MiniMetric({ label, value }) { return <div><span>{label}</span><strong>{value}</strong></div>; }

function CellAnalysis() { const { cellId } = useParams(); const state = useAsync(() => api.cell(cellId), [cellId]); return <AsyncState state={state}>{(cell) => <div className="page cell-page"><Link to="/map" className="back-link"><ArrowRight size={15} /> Back to heat map</Link><div className="cell-hero"><div><span className="eyebrow">LOCATION INTELLIGENCE / {cell.cell_id}</span><h2>Cell analysis</h2><p><Navigation size={14} /> {format(cell.location.latitude, 4)}° N, {format(cell.location.longitude, 4)}° E</p></div><span className={`risk-chip ${cell.risk.class.toLowerCase()}`}>{cell.risk.class} risk</span></div><div className="cell-grid"><RiskGauge score={cell.risk.score} /><div className="panel environmental-panel"><div className="panel-kicker"><Activity size={16} /> ENVIRONMENTAL PROFILE</div><h3>What the cell is telling us</h3><div className="environment-grid"><DataMetric icon={Thermometer} label="Median surface temperature" value={`${format(cell.metrics.lst_median_c)} °C`} /><DataMetric icon={Flame} label="90th percentile temperature" value={`${format(cell.metrics.lst_p90_c)} °C`} /><DataMetric icon={Leaf} label="Vegetation index / NDVI" value={format(cell.metrics.ndvi_median, 3)} /><DataMetric icon={BuildingIcon} label="Built-up index / NDBI" value={format(cell.metrics.ndbi_median, 3)} /></div><div className="risk-explanation"><strong>Why is this area at risk?</strong><p>{explainCell(cell)}</p></div></div></div><section className="recommendation-section"><SectionHeading eyebrow="Decision support" title="Recommended cooling actions" copy="Rules are calculated from this cell's observed environmental signals." action={<Link className="button button-ghost" to="/recommendations">Explore all actions <ArrowRight size={15} /></Link>} /><div className="recommendation-grid">{cell.recommendations.map((recommendation) => <RecommendationCard key={recommendation.rank} recommendation={recommendation} />)}</div></section></div>}</AsyncState>; }
function RiskGauge({ score }) { return <div className="panel gauge-panel"><div className="panel-kicker"><Zap size={16} /> BASELINE HEAT RISK</div><div className="gauge" style={{ '--score': `${score * 3.6}deg` }}><div className="gauge-inner"><strong>{format(score)}</strong><span>out of 100</span></div></div><p>Transparent index built from LST, NDBI, and vegetation deficit. It is a decision-support signal, not a guaranteed outcome.</p></div>; }
function DataMetric({ icon: Icon, label, value }) { return <div className="data-metric"><Icon size={18} /><span>{label}</span><strong>{value}</strong></div>; }
function BuildingIcon(props) { return <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M4 21V5l8-2v18M12 21h8V9l-8-2M7 8h2M7 12h2M7 16h2M15 12h2M15 16h2M15 20h2" /></svg>; }
function explainCell(cell) { const { metrics, risk } = cell; const signals = [`baseline score ${format(risk.score)}`, `median LST ${format(metrics.lst_median_c)} °C`, `NDVI ${format(metrics.ndvi_median, 3)}`, `NDBI ${format(metrics.ndbi_median, 3)}`]; if (risk.class === 'High') return `${cell.cell_id} is classified High because its observed signals combine into a high relative baseline score: ${signals.join(', ')}.`; if (risk.class === 'Medium') return `${cell.cell_id} is classified Medium in the dataset-relative baseline. Its observed profile is ${signals.join(', ')}.`; return `${cell.cell_id} is classified Low in the dataset-relative baseline. Its observed profile is ${signals.join(', ')}.`; }
function RecommendationCard({ recommendation }) { return <details className="recommendation-card"><summary><div className="rec-rank">0{recommendation.rank}</div><div className="rec-body"><div className="rec-head"><span className="eyebrow">{recommendation.priority} PRIORITY</span><strong>{recommendation.intervention}</strong></div><span className="recommendation-summary">Priority {format(recommendation.score)} · {recommendation.priority}</span></div><div className="rec-score"><strong>{format(recommendation.score)}</strong><span>priority score</span></div></summary><p className="recommendation-reason">{recommendation.reason}</p></details>; }

function Insights() { const state = useAsync(async () => { const [stats, cells] = await Promise.all([api.statistics(), fetchAllCells()]); const step = Math.max(1, Math.floor(cells.length / 100)); const detailCells = cells.filter((_, index) => index % step === 0).slice(0, 180); const details = await Promise.all(detailCells.map((cell) => api.cell(cell.cell_id))); return { stats, details }; }, []); return <AsyncState state={state}>{({ stats, details }) => { const counts = Array.from({ length: 28 }, (_, index) => details.filter((cell) => cell.metrics.lst_median_c >= 22 + index && cell.metrics.lst_median_c < 23 + index).length); const maxCount = Math.max(1, ...counts); return <div className="page insights-page"><SectionHeading eyebrow="Observed signals" title="Urban heat insights" copy="A compact analytical view grounded in the Earth Engine-derived API dataset." /><div className="insight-kpis"><InsightKpi label="Average surface temp" value={`${format(stats.average_lst)} °C`} sub="Across all cells" icon={Thermometer} /><InsightKpi label="Average NDVI" value={format(stats.average_ndvi, 3)} sub="Vegetation signal" icon={Leaf} /><InsightKpi label="Average NDBI" value={format(stats.average_ndbi, 3)} sub="Built-up signal" icon={BuildingIcon} /></div><div className="insights-grid"><div className="panel distribution-panel"><div className="panel-kicker"><BarChart3 size={16} /> TEMPERATURE DISTRIBUTION</div><h3>Surface temperature field</h3><p>Sampled from real cell-detail responses for a responsive view.</p><div className="bars">{counts.map((count, index) => <span key={index} style={{ height: `${Math.max(4, count / maxCount * 100)}%` }} title={`${22 + index}°C`} />)}</div><div className="chart-axis"><span>22°C</span><span>30°C</span><span>40°C</span><span>50°C</span></div></div><div className="panel signal-story"><div className="panel-kicker"><Sparkles size={16} /> SYSTEM READING</div><h3>How UrbanCool understands heat</h3><Flow /></div></div><div className="panel honest-panel"><ShieldCheck size={20} /><div><strong>Honest by design</strong><p>The current API exposes operational statistics and cell-level metrics. Deeper model artifacts remain documented in the repository rather than being presented as invented dashboard data.</p></div><Link to="/methodology" className="text-link">Read methodology <ArrowRight size={15} /></Link></div></div>; }}</AsyncState>; }
function InsightKpi({ icon: Icon, label, value, sub }) { return <div className="insight-kpi"><Icon size={19} /><span>{label}</span><strong>{value}</strong><small>{sub}</small></div>; }
function Flow() { return <div className="flow">{['Earth Engine', 'Landsat 8 / 9', 'LST · NDVI · NDBI', 'Spatial analysis', 'Machine learning', 'Cooling actions'].map((item, index) => <div className="flow-step" key={item}><span>{String(index + 1).padStart(2, '0')}</span><strong>{item}</strong>{index < 5 && <ChevronRight size={15} />}</div>)}</div>; }

function Recommendations() { const state = useAsync(() => fetchAllCells({ riskClass: 'High', limit: 120 }), []); const [filter, setFilter] = useState('All'); return <AsyncState state={state}>{(cells) => <div className="page recommendations-page"><SectionHeading eyebrow="Decision support" title="Cooling recommendations" copy="Explore actions attached to real high-risk cells from the UrbanCool database." /><div className="category-tabs">{['All', 'Tree / Vegetation Cover', 'Cool / Reflective Roofs', 'Shaded Public / Pedestrian Areas', 'Green Corridors / Connected Vegetation'].map((category) => <button className={filter === category ? 'active' : ''} key={category} onClick={() => setFilter(category)}>{category === 'All' ? 'All actions' : category.replace(' / ', ' · ')}</button>)}</div><div className="recommendation-explorer">{cells.slice(0, 36).map((cell) => <RecommendationListItem key={cell.cell_id} cellId={cell.cell_id} filter={filter} />)}</div><div className="disclaimer"><ShieldCheck size={18} /><span>UrbanCool provides decision support. Recommendations do not guarantee a specific temperature reduction and should be evaluated against local feasibility, land ownership, cost, and community priorities.</span></div></div>}</AsyncState>; }
function RecommendationListItem({ cellId, filter }) { const state = useAsync(() => api.recommendations(cellId), [cellId]); return <AsyncState state={state}>{(recommendations) => { const shown = filter === 'All' ? recommendations.slice(0, 1) : recommendations.filter((rec) => rec.intervention === filter); if (!shown.length) return null; return <div className="recommendation-list-item"><div><span className="eyebrow">{cellId}</span><strong>Cooling opportunity</strong></div>{shown.map((rec) => <div className="list-rec" key={rec.rank}><span className={`priority-dot ${rec.priority.toLowerCase()}`} /><span>{rec.intervention}</span><strong>{format(rec.score)}</strong><small>{rec.priority}</small></div>)}<Link to={`/cell/${cellId}`} className="icon-link" aria-label={`Open ${cellId}`}><ExternalLink size={16} /></Link></div>; }}</AsyncState>; }

function LegacyAdvisor() { const [cells, setCells] = useState([]); const [selectedId, setSelectedId] = useState(''); const [prompt, setPrompt] = useState(''); const cellState = useAsync(() => fetchAllCells({ riskClass: 'High' }), []); useEffect(() => { if (cellState.data) { setCells(cellState.data.slice(0, 120)); setSelectedId(cellState.data[0]?.cell_id || ''); } }, [cellState.data]); const detail = useAsync(() => selectedId ? api.cell(selectedId) : Promise.resolve(null), [selectedId]); const selected = detail.data; const answer = selected ? explainAdvisor(selected, prompt) : null; return <div className="page advisor-page"><SectionHeading eyebrow="Grounded intelligence / prototype" title="Ask UrbanCool" copy="A prepared interface for future AI assistance, grounded today in real cell metrics and recommendations." /><div className="advisor-layout"><div className="advisor-chat panel"><div className="chat-head"><span className="status-avatar"><BrainCircuit size={19} /></span><div><strong>UrbanCool advisor</strong><small>Data-backed prototype · no LLM connected</small></div><span className="online-tag">READY</span></div><div className="chat-body"><div className="message assistant"><span className="message-avatar"><Sparkles size={14} /></span><div>Choose a real high-risk location and ask about the signals. I will answer from its observed metrics and recommendation records.</div></div>{answer && <div className="message assistant"><span className="message-avatar"><Sparkles size={14} /></span><div>{answer}</div></div>}<div className="prompt-grid">{['Why is this area high risk?', 'What cooling action should be prioritized?', 'What environmental factors contribute to heat?', 'How can vegetation help this area?'].map((item) => <button key={item} onClick={() => setPrompt(item)}>{item}<ArrowRight size={14} /></button>)}</div></div><div className="chat-input"><input value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Ask about this location…" /><button onClick={() => setPrompt(prompt || 'Why is this area high risk?')} aria-label="Send prompt"><ArrowRight size={17} /></button></div></div><div className="advisor-context panel"><div className="panel-kicker"><MapPinned size={16} /> LOCATION CONTEXT</div><h3>Choose a real cell</h3><select value={selectedId} onChange={(event) => setSelectedId(event.target.value)}><option value="">Select high-risk cell</option>{cells.map((cell) => <option key={cell.cell_id} value={cell.cell_id}>{cell.cell_id} · {format(cell.risk.score)}</option>)}</select>{detail.loading ? <LoadingState /> : selected && <div className="context-readout"><div className={`risk-chip ${selected.risk.class.toLowerCase()}`}>{selected.risk.class} risk</div><div className="context-score">{format(selected.risk.score)}<small>baseline score</small></div><div className="context-metrics"><MiniMetric label="LST" value={`${format(selected.metrics.lst_median_c)}°C`} /><MiniMetric label="NDVI" value={format(selected.metrics.ndvi_median, 2)} /><MiniMetric label="NDBI" value={format(selected.metrics.ndbi_median, 2)} /></div><Link to={`/cell/${selected.cell_id}`} className="text-link">Open full analysis <ArrowRight size={15} /></Link></div>}</div></div></div>; }
function explainAdvisor(cell, prompt) { if (prompt.includes('cooling action')) return `The current top recommendation is ${cell.recommendations[0].intervention} with a ${format(cell.recommendations[0].score)} priority score. The engine bases this on the cell's heat, vegetation, and built-up signals.`; if (prompt.includes('vegetation')) return `The observed NDVI is ${format(cell.metrics.ndvi_median, 3)}. The recommendation engine treats relatively low vegetation as a stronger opportunity for tree cover and connected green corridors.`; if (prompt.includes('environmental')) return `This cell records ${format(cell.metrics.lst_median_c)} °C median LST, NDVI ${format(cell.metrics.ndvi_median, 3)}, and NDBI ${format(cell.metrics.ndbi_median, 3)}.`; return explainCell(cell); }

function Methodology() { return <div className="page methodology-page"><SectionHeading eyebrow="Provenance & responsibility" title="Methodology" copy="UrbanCool is designed to make the path from observation to action visible." /><div className="method-flow panel"><Flow /></div><div className="method-grid"><MethodBlock icon={Globe2} title="Data sources" copy="Google Earth Engine provides Landsat 8 and Landsat 9 Collection 2 Level-2 observations for March–May 2022–2025. QA masking, official scale factors, and a 500 m grid are applied before analysis." /><MethodBlock icon={BarChart3} title="Spatial analysis & machine learning" copy="LST, NDVI, and NDBI signals are summarized per 500 m cell. The existing Random Forest model predicts median LST from NDVI, NDBI, latitude, and longitude, with spatial holdout evaluation." /><MethodBlock icon={Trees} title="Recommendation logic" copy="Four intervention categories receive deterministic 0–100 scores from dataset-relative heat, vegetation, and built-up signals. The top three are exposed through the API." /><MethodBlock icon={BrainCircuit} title="Grounded AI Advisor" copy="The AI Advisor sends the selected cell ID to the backend, which retrieves real metrics and deterministic recommendations from SQLite before asking the configured LLM to explain them. The LLM does not calculate or alter the official risk score. Responses use processed satellite-derived data, are not real-time, and cannot establish local cost, ownership, feasibility, planning, or community conditions." /></div><div className="limitations panel"><span className="eyebrow">LIMITATIONS</span><h3>Useful signals, not universal truths.</h3><p>UrbanCool uses a bounded Coimbatore district study area and relative percentiles. It does not account for implementation cost, land ownership, local planning constraints, community priorities, or every factor that shapes lived heat exposure. Any action should be reviewed by local experts and communities.</p></div></div>; }
function MethodBlock({ icon: Icon, title, copy }) { return <div className="method-block"><span className="method-icon"><Icon size={20} /></span><h3>{title}</h3><p>{copy}</p></div>; }

export default App;
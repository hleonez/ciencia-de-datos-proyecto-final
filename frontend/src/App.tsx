import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  Activity, ArrowDownToLine, BarChart3, Check, ChevronRight, CircleAlert, Copy,
  Clock3, Database, FileSpreadsheet, Gauge, Layers3, LoaderCircle, Play,
  RefreshCw, Settings2, ShieldCheck, Sparkles, Square, Target, Trash2, TrendingUp,
} from "lucide-react";
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { AcfChart, DecompositionPanel, DistributionPanel } from "./components/EdaCharts";
import { MetricCard } from "./components/MetricCard";
import { ComparisonPanel, FullSeriesChart, ResidualsChart, RunHistory } from "./components/ResultsCharts";
import { UploadPanel } from "./components/UploadPanel";
import { api } from "./services/api";
import { buildSampleFile } from "./sampleData";
import type { Dataset, Eda, RunRecord, TrainingJob, TrainingResults, TrainConfig } from "./types";

type Section = "dashboard" | "explore" | "train" | "results";

const navigation: Array<{ id: Section; label: string; icon: typeof Database }> = [
  { id: "dashboard", label: "Resumen", icon: BarChart3 },
  { id: "explore", label: "Exploración", icon: Activity },
  { id: "train", label: "Modelo LSTM", icon: Settings2 },
  { id: "results", label: "Resultados", icon: TrendingUp },
];

const initialConfig: TrainConfig = {
  date_column: "",
  target_column: "",
  feature_columns: [],
  lookback: 24,
  epochs: 50,
  batch_size: 32,
  units: 64,
  dropout: 0.2,
  train_ratio: 0.7,
  validation_ratio: 0.15,
  test_ratio: 0.15,
  missing_strategy: "forward_fill",
  patience: 8,
  forecast_steps: 12,
};

const formatNumber = (value: number | null | undefined, digits = 3) =>
  value == null || !Number.isFinite(value)
    ? "—"
    : new Intl.NumberFormat("es", { maximumFractionDigits: digits }).format(value);

function App() {
  const [section, setSection] = useState<Section>("dashboard");
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [eda, setEda] = useState<Eda | null>(null);
  const [edaLoading, setEdaLoading] = useState(false);
  const [seriesColumn, setSeriesColumn] = useState("");
  const [edaDateColumn, setEdaDateColumn] = useState("");
  const [edaPeriod, setEdaPeriod] = useState(0);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const jobConfigRef = useRef<TrainConfig>(initialConfig);
  const [config, setConfig] = useState<TrainConfig>(initialConfig);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<TrainingJob | null>(null);
  const [results, setResults] = useState<TrainingResults | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const numericColumns = useMemo(
    () => dataset?.columns.filter((column) => column.data_type === "numeric") ?? [],
    [dataset],
  );
  const dateColumns = useMemo(
    () => dataset?.columns.filter((column) => column.data_type === "date") ?? [],
    [dataset],
  );

  useEffect(() => {
    if (!dataset) return;
    let active = true;
    setEdaLoading(true);
    api.eda(dataset.id, seriesColumn, edaDateColumn, edaPeriod)
      .then((response) => { if (active) setEda(response); })
      .catch((reason) => { if (active) setError(reason instanceof Error ? reason.message : "No se pudo calcular el análisis exploratorio."); })
      .finally(() => { if (active) setEdaLoading(false); });
    return () => { active = false; };
  }, [dataset, seriesColumn, edaDateColumn, edaPeriod]);

  useEffect(() => {
    if (!jobId) return;
    let active = true;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const status = await api.job(jobId);
        if (!active) return;
        setJob(status);
        if (status.status === "completed") {
          const finished = await api.results(jobId);
          if (!active) return;
          const used = jobConfigRef.current;
          setResults(finished);
          setRuns((current) => [...current, {
            id: jobId,
            finishedAt: new Date().toLocaleTimeString("es"),
            target: used.target_column,
            predictors: used.feature_columns.length,
            lookback: used.lookback,
            epochs: finished.losses.train.length,
            units: used.units,
            metrics: finished.metrics,
            baseline: finished.baseline_metrics,
          }]);
          setSection("results");
          return;
        }
        if (status.status === "cancelled") {
          setNotice("El entrenamiento se canceló. No se generaron resultados.");
          return;
        }
        if (status.status === "failed") {
          setError(status.error || "El entrenamiento no se pudo completar.");
          return;
        }
        timer = window.setTimeout(() => { void poll(); }, 1200);
      } catch (reason) {
        if (active) setError(reason instanceof Error ? reason.message : "No se pudo consultar el entrenamiento.");
      }
    };
    void poll();
    return () => {
      active = false;
      if (timer) window.clearTimeout(timer);
    };
  }, [jobId]);

  function handleUploaded(next: Dataset) {
    if (dataset) {
      void api.deleteDataset(dataset.id).catch((reason) => {
        setNotice(`El archivo anterior no se pudo eliminar: ${reason instanceof Error ? reason.message : "error desconocido"}`);
      });
    }
    setDataset(next);
    setEda(null);
    setResults(null);
    setJob(null);
    setJobId(null);
    setError("");
    setNotice("");
    const numericNames = next.columns.filter((column) => column.data_type === "numeric");
    // Identifier-like columns (one value per row, or named "...id") carry no signal for a forecast.
    const measures = numericNames.filter((column) =>
      !(next.row_count > 10 && column.unique_count === next.row_count) && !/(^|[\s_-])id$/i.test(column.name));
    const numbers = (measures.length ? measures : numericNames).map((column) => column.name);
    setSeriesColumn(numbers[0] ?? "");
    setEdaDateColumn("");
    setEdaPeriod(0);
    setRuns([]);
    setConfig({
      ...initialConfig,
      date_column: next.columns.find((column) => column.data_type === "date")?.name ?? "",
      target_column: numbers[0] ?? "",
      // The target's own history is the strongest predictor of a time series, so it is included by default.
      feature_columns: numbers,
    });
    setSection("dashboard");
  }

  async function removeDataset() {
    if (!dataset) return;
    setBusy(true);
    setError("");
    try {
      await api.deleteDataset(dataset.id);
      setDataset(null);
      setEda(null);
      setResults(null);
      setJob(null);
      setJobId(null);
      setRuns([]);
      setConfig(initialConfig);
      setNotice("");
      setSection("dashboard");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "No se pudo eliminar el archivo.");
    } finally {
      setBusy(false);
    }
  }

  async function startTraining() {
    if (!dataset) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const response = await api.startTraining(dataset.id, config);
      jobConfigRef.current = config;
      setJobId(response.job_id);
      setJob({
        id: response.job_id,
        dataset_id: dataset.id,
        status: response.status,
        current_epoch: 0,
        total_epochs: config.epochs,
        train_loss: null,
        validation_loss: null,
        loss_history: [],
        validation_loss_history: [],
        error: null,
      });
      setResults(null);
      setSection("train");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "No se pudo iniciar el entrenamiento.");
    } finally {
      setBusy(false);
    }
  }

  async function cancelTraining() {
    if (!jobId) return;
    try {
      await api.cancelJob(jobId);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "No se pudo cancelar el entrenamiento.");
    }
  }

  function updateConfig<K extends keyof TrainConfig>(key: K, value: TrainConfig[K]) {
    setConfig((current) => ({ ...current, [key]: value }));
  }

  const canTrain = Boolean(
    dataset && dateColumns.length && numericColumns.length && config.date_column &&
    config.target_column && config.feature_columns.length && !busy &&
    !(job && ["queued", "running"].includes(job.status)),
  );

  return (
    <div className="min-h-screen bg-canvas">
      <aside className="fixed inset-y-0 left-0 z-20 hidden w-64 flex-col border-r border-slate-200/80 bg-white lg:flex">
        <div className="flex h-[76px] items-center gap-3 border-b border-slate-100 px-6">
          <div className="grid h-10 w-10 place-items-center rounded-xl bg-blue-600 text-white shadow-sm shadow-blue-200">
            <Activity size={21} />
          </div>
          <div>
            <p className="font-bold tracking-tight text-ink">DataLab</p>
            <p className="text-[11px] font-medium text-slate-400">ANÁLISIS INTELIGENTE</p>
          </div>
        </div>
        <div className="px-4 pt-7">
          <p className="mb-3 px-3 text-[10px] font-bold uppercase tracking-[0.16em] text-slate-400">Espacio de trabajo</p>
          <nav className="space-y-1">
            {navigation.map(({ id, label, icon: Icon }) => {
              const disabled = id !== "dashboard" && !dataset;
              return (
                <button
                  key={id}
                  type="button"
                  disabled={disabled}
                  onClick={() => setSection(id)}
                  className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm font-medium transition ${
                    section === id ? "bg-blue-50 text-blue-700" : "text-slate-500 hover:bg-slate-50 hover:text-ink"
                  } disabled:cursor-not-allowed disabled:opacity-40`}
                >
                  <Icon size={18} />
                  {label}
                  {section === id && <ChevronRight className="ml-auto" size={15} />}
                </button>
              );
            })}
          </nav>
        </div>
        <div className="mt-auto p-4">
          <div className="rounded-2xl bg-slate-50 p-4">
            <div className="mb-3 flex items-center gap-2 text-xs font-semibold text-slate-600">
              <ShieldCheck size={16} className="text-emerald-600" /> Procesamiento privado
            </div>
            <p className="text-xs leading-relaxed text-slate-500">Tus archivos se procesan temporalmente en el servidor y no se usan para entrenar modelos compartidos.</p>
          </div>
        </div>
      </aside>

      <div className="lg:pl-64">
        <header className="sticky top-0 z-10 flex min-h-[76px] items-center justify-between border-b border-slate-200/80 bg-white/90 px-4 backdrop-blur sm:px-8">
          <div className="flex items-center gap-3">
            <div className="grid h-9 w-9 place-items-center rounded-lg bg-blue-600 text-white lg:hidden"><Activity size={19} /></div>
            <div>
              <p className="text-sm font-bold text-ink lg:hidden">DataLab</p>
              <p className="hidden text-sm font-semibold text-ink lg:block">Plataforma de análisis LSTM</p>
              <p className="text-xs text-muted">{dataset ? dataset.filename : "Espacio de análisis"}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {dataset && (
              <>
                <span className="hidden items-center gap-1.5 rounded-full bg-emerald-50 px-3 py-1.5 text-xs font-semibold text-emerald-700 sm:inline-flex">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" /> Dataset validado
                </span>
                <button type="button" className="secondary-button !px-3 !py-2" onClick={() => setSection("dashboard")} title="Cargar otro archivo">
                  <RefreshCw size={15} /><span className="hidden sm:inline">Nuevo análisis</span>
                </button>
              </>
            )}
          </div>
        </header>

        <main className="mx-auto max-w-[1440px] px-4 py-7 sm:px-8 sm:py-9">
          <nav className="mb-6 grid grid-cols-4 gap-1 rounded-xl border border-slate-200 bg-white p-1 lg:hidden">
            {navigation.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                type="button"
                disabled={id !== "dashboard" && !dataset}
                onClick={() => setSection(id)}
                className={`flex flex-col items-center justify-center gap-1 rounded-lg px-1 py-2 text-[10px] font-semibold sm:flex-row sm:text-xs ${
                  section === id ? "bg-blue-50 text-blue-700" : "text-slate-500"
                } disabled:opacity-40`}
              ><Icon size={16} />{label}</button>
            ))}
          </nav>
          {error && (
            <div role="alert" className="mb-6 flex items-start gap-3 rounded-xl border border-red-100 bg-red-50 px-4 py-3 text-sm text-red-700">
              <CircleAlert className="mt-0.5 shrink-0" size={17} />
              <span className="flex-1">{error}</span>
              <button type="button" onClick={() => setError("")} className="font-semibold">Cerrar</button>
            </div>
          )}
          {notice && <p className="mb-5 rounded-xl bg-amber-50 px-4 py-3 text-sm text-amber-800">{notice}</p>}

          {section === "dashboard" && (
            <Dashboard
              dataset={dataset}
              results={results}
              onUploaded={handleUploaded}
              onNavigate={setSection}
              onRemove={removeDataset}
              removing={busy && !jobId}
            />
          )}
          {section === "explore" && dataset && (
            <ExploreSection
              dataset={dataset}
              eda={eda}
              loading={edaLoading}
              seriesColumn={seriesColumn}
              onSeriesColumn={setSeriesColumn}
              dateColumn={edaDateColumn || eda?.series_date_column || ""}
              onDateColumn={setEdaDateColumn}
              onPeriod={setEdaPeriod}
            />
          )}
          {section === "train" && dataset && (
            <TrainSection
              dataset={dataset}
              numericColumns={numericColumns.map((column) => column.name)}
              dateColumns={dateColumns.map((column) => column.name)}
              config={config}
              job={job}
              canTrain={canTrain}
              busy={busy}
              onConfig={updateConfig}
              onStart={startTraining}
              onCancel={cancelTraining}
              onResults={() => setSection("results")}
            />
          )}
          {section === "results" && dataset && (
            <ResultsSection
              results={results}
              runs={runs}
              jobId={jobId}
              target={config.target_column}
              onTrain={() => setSection("train")}
            />
          )}
        </main>
      </div>
    </div>
  );
}

function PageHeading({ eyebrow, title, description, action }: {
  eyebrow: string; title: string; description: string; action?: ReactNode;
}) {
  return (
    <div className="mb-7 flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
      <div>
        <p className="mb-2 text-xs font-bold uppercase tracking-[0.15em] text-blue-600">{eyebrow}</p>
        <h1 className="text-2xl font-extrabold tracking-tight text-ink sm:text-[30px]">{title}</h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">{description}</p>
      </div>
      {action}
    </div>
  );
}

function Dashboard({ dataset, results, onUploaded, onNavigate, onRemove, removing }: {
  dataset: Dataset | null;
  results: TrainingResults | null;
  onUploaded: (dataset: Dataset) => void;
  onNavigate: (section: Section) => void;
  onRemove: () => void;
  removing: boolean;
}) {
  if (!dataset) {
    return (
      <>
        <PageHeading eyebrow="Tu espacio de trabajo" title="Analiza tus datos. Predice lo que sigue." description="Carga una serie temporal, explora sus patrones y entrena un modelo LSTM con tus propios datos." />
        <div className="grid gap-6 xl:grid-cols-[1.4fr_0.8fr]">
          <div className="card">
            <div className="mb-5 flex items-center justify-between gap-4">
              <div><h2 className="section-title">Comienza con tus datos</h2><p className="mt-1 text-sm text-muted">Elige un archivo CSV o Excel para iniciar el análisis.</p></div>
              <span className="hidden rounded-xl bg-blue-50 p-3 text-blue-600 sm:grid"><Database size={20} /></span>
            </div>
            <UploadPanel onUploaded={onUploaded} />
            <SampleDatasetButton onUploaded={onUploaded} />
          </div>
          <div className="space-y-4">
            <div className="card">
              <div className="mb-4 flex items-center gap-2"><Sparkles size={18} className="text-blue-600" /><h2 className="section-title">Tu flujo de análisis</h2></div>
              <div className="space-y-4">
                {[
                  ["01", "Carga y explora", "Valida tu archivo y revisa estadísticas, calidad y correlaciones."],
                  ["02", "Configura tu modelo", "Selecciona la fecha, el target y las variables de entrada."],
                  ["03", "Entrena y evalúa", "Sigue el progreso y compara predicciones con datos no vistos."],
                ].map(([number, title, text]) => (
                  <div key={number} className="flex gap-3">
                    <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-slate-100 text-xs font-bold text-slate-500">{number}</span>
                    <div><p className="text-sm font-semibold text-ink">{title}</p><p className="mt-0.5 text-xs leading-relaxed text-muted">{text}</p></div>
                  </div>
                ))}
              </div>
            </div>
            <div className="rounded-2xl border border-blue-100 bg-gradient-to-br from-blue-50 to-indigo-50 p-5">
              <p className="text-sm font-semibold text-blue-900">Privacidad desde el diseño</p>
              <p className="mt-2 text-xs leading-relaxed text-blue-800/75">Los datos se procesan en Python y se eliminan al borrar el análisis o vencer su tiempo de retención.</p>
            </div>
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <PageHeading
        eyebrow="Resumen del análisis"
        title="Tu espacio de trabajo"
        description="El conjunto de datos está listo. Explora su calidad o configura un modelo para comenzar."
        action={<button type="button" className="secondary-button" onClick={onRemove} disabled={removing}><Trash2 size={16} />{removing ? "Eliminando…" : "Eliminar archivo"}</button>}
      />
      <div className="mb-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Registros" value={formatNumber(dataset.row_count, 0)} hint="Filas detectadas" icon={Database} />
        <MetricCard label="Variables" value={formatNumber(dataset.column_count, 0)} hint="Columnas detectadas" icon={Layers3} tone="violet" />
        <MetricCard label="Valores faltantes" value={formatNumber(dataset.null_count, 0)} hint="En todo el conjunto" icon={CircleAlert} tone="amber" />
        <MetricCard label="Duplicados" value={formatNumber(dataset.duplicate_count, 0)} hint="Filas repetidas" icon={Copy} tone="green" />
      </div>
      <div className="grid gap-6 xl:grid-cols-[1fr_0.7fr]">
        <div className="card">
          <div className="mb-5 flex items-start justify-between gap-3">
            <div><h2 className="section-title">Conjunto cargado</h2><p className="mt-1 text-sm text-muted">Validación completada · {(dataset.size_bytes / 1024 / 1024).toFixed(2)} MB</p></div>
            <span className="grid h-10 w-10 place-items-center rounded-xl bg-emerald-50 text-emerald-600"><Check size={19} /></span>
          </div>
          <div className="flex items-center gap-3 rounded-xl border border-slate-100 bg-slate-50/70 p-4">
            <span className="grid h-11 w-11 place-items-center rounded-xl bg-white text-blue-600 shadow-sm"><FileSpreadsheet size={20} /></span>
            <div className="min-w-0 flex-1"><p className="truncate text-sm font-semibold text-ink">{dataset.filename}</p><p className="mt-1 text-xs text-muted">{dataset.row_count.toLocaleString("es")} registros · {dataset.column_count} columnas</p></div>
            <span className="rounded-md bg-emerald-100 px-2 py-1 text-[10px] font-bold uppercase tracking-wide text-emerald-700">Válido</span>
          </div>
          <div className="mt-5 flex flex-wrap gap-2">
            {dataset.columns.slice(0, 9).map((column) => (
              <span key={column.name} className="rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-600">{column.name}<span className="ml-1.5 text-slate-400">{column.data_type === "numeric" ? "123" : column.data_type === "date" ? "fecha" : "texto"}</span></span>
            ))}
            {dataset.columns.length > 9 && <span className="px-2 py-1.5 text-xs text-muted">+{dataset.columns.length - 9} más</span>}
          </div>
          <div className="mt-6 flex flex-wrap gap-3">
            <button type="button" className="primary-button" onClick={() => onNavigate("explore")}><BarChart3 size={16} />Explorar datos</button>
            <button type="button" className="secondary-button" onClick={() => onNavigate("train")}><Settings2 size={16} />Configurar modelo</button>
          </div>
        </div>
        <div className="card">
          <div className="mb-4 flex items-center gap-2"><TrendingUp size={18} className="text-blue-600" /><h2 className="section-title">Último análisis</h2></div>
          {results ? (
            <div>
              <p className="mb-4 text-sm leading-relaxed text-muted">{results.summary}</p>
              <div className="grid grid-cols-2 gap-3">
                <SmallMetric label="MAE" value={formatNumber(results.metrics.mae)} />
                <SmallMetric label="RMSE" value={formatNumber(results.metrics.rmse)} />
                <SmallMetric label="R²" value={formatNumber(results.metrics.r2)} />
                <SmallMetric label="Predicciones" value={formatNumber(results.predictions.length, 0)} />
              </div>
              <button type="button" className="secondary-button mt-5 w-full" onClick={() => onNavigate("results")}>Ver resultados <ChevronRight size={15} /></button>
            </div>
          ) : (
            <div className="flex min-h-40 flex-col items-center justify-center rounded-xl bg-slate-50 px-4 text-center">
              <span className="mb-3 grid h-10 w-10 place-items-center rounded-xl bg-white text-slate-400 shadow-sm"><Clock3 size={18} /></span>
              <p className="text-sm font-semibold text-slate-600">Aún no hay resultados</p>
              <p className="mt-1 max-w-xs text-xs leading-relaxed text-muted">Configura e inicia un entrenamiento para evaluar tu modelo.</p>
            </div>
          )}
        </div>
      </div>
      <div className="mt-6 card">
        <div className="mb-5"><h2 className="section-title">Cambiar conjunto de datos</h2><p className="mt-1 text-sm text-muted">Carga otro archivo CSV o Excel para iniciar un análisis nuevo.</p></div>
        <UploadPanel onUploaded={onUploaded} />
      </div>
      {dataset.preview && <PreviewTable preview={dataset.preview} />}
    </>
  );
}

function SampleDatasetButton({ onUploaded }: { onUploaded: (dataset: Dataset) => void }) {
  const [loading, setLoading] = useState(false);
  const [failure, setFailure] = useState("");
  async function load() {
    setLoading(true);
    setFailure("");
    try {
      onUploaded(await api.upload(buildSampleFile()));
    } catch (reason) {
      setFailure(reason instanceof Error ? reason.message : "No se pudo cargar el ejemplo.");
    } finally {
      setLoading(false);
    }
  }
  return (
    <div className="mt-4 flex flex-col items-center gap-2 text-center">
      <button type="button" className="secondary-button" onClick={() => void load()} disabled={loading}>
        <Sparkles size={16} />{loading ? "Cargando ejemplo…" : "Probar con un dataset de ejemplo"}
      </button>
      <p className="text-xs text-muted">Demanda horaria sintética con ciclo diario, temperatura y humedad (480 registros).</p>
      {failure && <p role="alert" className="rounded-lg bg-red-50 px-4 py-2 text-sm text-red-700">{failure}</p>}
    </div>
  );
}

function SmallMetric({ label, value }: { label: string; value: string }) {
  return <div className="rounded-xl bg-slate-50 p-3"><p className="text-xs text-muted">{label}</p><p className="mt-1 font-semibold text-ink">{value}</p></div>;
}

function PreviewTable({ preview }: { preview: Array<Record<string, string | number | boolean | null>> }) {
  const columns = Object.keys(preview[0] ?? {});
  return (
    <div className="card mt-6">
      <div className="mb-4 flex items-center justify-between"><div><h2 className="section-title">Vista previa</h2><p className="mt-1 text-xs text-muted">Primeros registros recibidos desde el backend</p></div><span className="text-xs text-muted">{preview.length} filas</span></div>
      <div className="overflow-x-auto">
        <table className="data-table min-w-full text-left text-xs">
          <thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
          <tbody>{preview.slice(0, 10).map((row, index) => <tr key={index}>{columns.map((column) => <td key={column}>{row[column] == null ? <span className="text-amber-500">Nulo</span> : String(row[column])}</td>)}</tr>)}</tbody>
        </table>
      </div>
    </div>
  );
}

function ExploreSection({ dataset, eda, loading, seriesColumn, onSeriesColumn, dateColumn, onDateColumn, onPeriod }: {
  dataset: Dataset; eda: Eda | null; loading: boolean; seriesColumn: string;
  onSeriesColumn: (column: string) => void;
  dateColumn: string; onDateColumn: (column: string) => void; onPeriod: (period: number) => void;
}) {
  const numericColumns = dataset.columns.filter((column) => column.data_type === "numeric");
  return (
    <>
      <PageHeading eyebrow="Exploración de datos" title="Conoce tu serie temporal" description="Revisa estadísticas, calidad, correlaciones y la evolución de las variables antes de modelar." />
      <div className="mb-6 grid gap-4 sm:grid-cols-3">
        <MetricCard label="Valores nulos" value={formatNumber(eda?.null_count ?? dataset.null_count, 0)} hint="Registros por completar" icon={CircleAlert} tone="amber" />
        <MetricCard label="Filas duplicadas" value={formatNumber(eda?.duplicate_count ?? dataset.duplicate_count, 0)} hint="Duplicados exactos" icon={Copy} tone="violet" />
        <MetricCard label="Variables numéricas" value={formatNumber(numericColumns.length, 0)} hint="Disponibles para modelar" icon={Gauge} tone="green" />
      </div>
      <div className="grid gap-6 xl:grid-cols-[1.35fr_0.65fr]">
        <div className="card min-w-0">
          <div className="mb-5 flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
            <div><h2 className="section-title">Comportamiento temporal</h2><p className="mt-1 text-sm text-muted">{eda?.series_date_column ? `Ordenado por ${eda.series_date_column}` : "Se requiere una columna de fecha para mostrar la serie."}</p></div>
            <div className="flex flex-wrap gap-2">
              {(eda?.date_columns.length ?? 0) > 1 && (
                <select className="field !w-auto min-w-40" value={dateColumn} onChange={(event) => onDateColumn(event.target.value)} aria-label="Columna de fecha">
                  {eda?.date_columns.map((column) => <option key={column} value={column}>{column}</option>)}
                </select>
              )}
              <select className="field !w-auto min-w-40" value={seriesColumn} onChange={(event) => onSeriesColumn(event.target.value)} aria-label="Variable para el gráfico">
                {numericColumns.map((column) => <option key={column.name} value={column.name}>{column.name}</option>)}
              </select>
            </div>
          </div>
          <div className="h-[300px] w-full">
            {loading ? <ChartLoading /> : eda?.series.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={eda.series} margin={{ top: 10, right: 12, left: 0, bottom: 4 }}>
                  <CartesianGrid stroke="#edf0f5" strokeDasharray="4 4" vertical={false} />
                  <XAxis dataKey="date" tickFormatter={shortDate} tick={{ fontSize: 11, fill: "#8290a5" }} axisLine={false} tickLine={false} minTickGap={35} />
                  <YAxis tick={{ fontSize: 11, fill: "#8290a5" }} axisLine={false} tickLine={false} width={50} />
                  <Tooltip labelFormatter={(value) => new Date(String(value)).toLocaleString("es")} formatter={(value) => [formatNumber(Number(value)), seriesColumn]} />
                  <Line type="monotone" dataKey="value" stroke="#3567e8" strokeWidth={2.2} dot={false} activeDot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : <EmptyChart text={numericColumns.length ? "No hay datos válidos para graficar." : "No se detectaron columnas numéricas."} />}
          </div>
        </div>
        <div className="card">
          <div className="mb-4"><h2 className="section-title">Calidad por variable</h2><p className="mt-1 text-sm text-muted">Valores faltantes y tipos detectados</p></div>
          <div className="max-h-[300px] space-y-3 overflow-y-auto pr-1">
            {dataset.columns.map((column) => (
              <div key={column.name} className="flex items-center justify-between gap-3 border-b border-slate-100 pb-3 last:border-0">
                <div className="min-w-0"><p className="truncate text-sm font-medium text-slate-700">{column.name}</p><p className="mt-0.5 text-xs text-slate-400">{column.data_type === "numeric" ? "Numérica" : column.data_type === "date" ? "Fecha" : "Texto"} · {column.unique_count.toLocaleString("es")} valores únicos</p></div>
                <span className={`shrink-0 rounded-lg px-2 py-1 text-xs font-semibold ${column.null_count || column.invalid_count ? "bg-amber-50 text-amber-700" : "bg-emerald-50 text-emerald-700"}`}>{column.null_count} nulos{column.invalid_count ? ` · ${column.invalid_count} inválidos` : ""}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      {eda && (
        <div className="mt-6 grid gap-6 xl:grid-cols-2">
          <DistributionPanel column={seriesColumn} bins={eda.histograms[seriesColumn]} box={eda.boxplots[seriesColumn]} />
          <AcfChart acf={eda.acf} column={seriesColumn} />
          <div className="xl:col-span-2">
            <DecompositionPanel decomposition={eda.decomposition} detectedPeriod={eda.period} onPeriod={onPeriod} />
          </div>
        </div>
      )}
      <div className="card mt-6">
        <div className="mb-4"><h2 className="section-title">Estadística descriptiva</h2><p className="mt-1 text-sm text-muted">Resumen calculado en el backend para columnas numéricas</p></div>
        {eda?.statistics && Object.keys(eda.statistics).length ? (
          <div className="overflow-x-auto">
            <table className="data-table min-w-full text-left text-xs">
              <thead><tr>{["Variable", "Media", "Mediana", "Desv. estándar", "Mínimo", "Máximo", "Nulos"].map((item) => <th key={item}>{item}</th>)}</tr></thead>
              <tbody>{Object.entries(eda.statistics).map(([name, stats]) => <tr key={name}><td className="font-semibold text-ink">{name}</td><td>{formatNumber(stats.mean)}</td><td>{formatNumber(stats.median)}</td><td>{formatNumber(stats.std)}</td><td>{formatNumber(stats.min)}</td><td>{formatNumber(stats.max)}</td><td>{stats.null_count}</td></tr>)}</tbody>
            </table>
          </div>
        ) : <EmptyChart text={loading ? "Calculando estadísticas…" : "No hay estadísticas numéricas disponibles."} />}
      </div>
      <div className="card mt-6">
        <div className="mb-4"><h2 className="section-title">Matriz de correlación</h2><p className="mt-1 text-sm text-muted">Correlación de Pearson entre variables numéricas</p></div>
        {eda && Object.keys(eda.correlation).length ? <CorrelationTable correlation={eda.correlation} /> : <EmptyChart text="Se requieren al menos dos variables numéricas." />}
      </div>
      <PreviewTable preview={dataset.preview ?? []} />
    </>
  );
}

function CorrelationTable({ correlation }: { correlation: Eda["correlation"] }) {
  const columns = Object.keys(correlation);
  return (
    <div className="overflow-auto">
      <table className="data-table min-w-full text-center text-xs">
        <thead><tr><th>Variable</th>{columns.map((name) => <th key={name}>{name}</th>)}</tr></thead>
        <tbody>{columns.map((row) => <tr key={row}><td className="text-left font-semibold text-ink">{row}</td>{columns.map((column) => {
          const value = correlation[row]?.[column];
          return <td key={column}><span className={`inline-flex min-w-12 justify-center rounded px-1.5 py-1 ${value == null ? "text-slate-300" : value > 0.65 ? "bg-blue-100 text-blue-800" : value < -0.65 ? "bg-violet-100 text-violet-800" : "bg-slate-50 text-slate-600"}`}>{formatNumber(value, 2)}</span></td>;
        })}</tr>)}</tbody>
      </table>
    </div>
  );
}

function TrainSection({ dataset, numericColumns, dateColumns, config, job, canTrain, busy, onConfig, onStart, onCancel, onResults }: {
  dataset: Dataset; numericColumns: string[]; dateColumns: string[];
  config: TrainConfig; job: TrainingJob | null; canTrain: boolean; busy: boolean;
  onConfig: <K extends keyof TrainConfig>(key: K, value: TrainConfig[K]) => void;
  onStart: () => void; onCancel: () => void; onResults: () => void;
}) {
  const dateUnique = dataset.columns.find((column) => column.name === config.date_column)?.unique_count ?? dataset.row_count;
  const repeatedDates = dataset.row_count >= 20 && dateUnique < dataset.row_count * 0.5;
  const ratioTotal = config.train_ratio + config.validation_ratio + config.test_ratio;
  const ratioValid = Math.abs(ratioTotal - 1) < 0.001;
  const running = Boolean(job && ["queued", "running"].includes(job.status));
  const progress = job ? Math.min(100, Math.round((job.current_epoch / job.total_epochs) * 100)) : 0;
  const lossChart = (job?.loss_history ?? []).map((trainLoss, index) => ({
    epoch: index + 1,
    train: trainLoss,
    validation: job?.validation_loss_history[index],
  }));
  return (
    <>
      <PageHeading eyebrow="Configuración y entrenamiento" title="Configura tu modelo LSTM" description={`Define las variables y parámetros para entrenar un modelo con ${dataset.filename}.`} />
      {(!dateColumns.length || !numericColumns.length) && <div role="alert" className="mb-6 rounded-xl border border-amber-100 bg-amber-50 p-4 text-sm text-amber-800">El archivo debe tener una columna temporal detectable y al menos una columna numérica para entrenar.</div>}
      {repeatedDates && <div role="alert" className="mb-6 rounded-xl border border-amber-100 bg-amber-50 p-4 text-sm text-amber-800">La columna «{config.date_column}» solo tiene {dateUnique.toLocaleString("es")} fechas distintas en {dataset.row_count.toLocaleString("es")} filas: el archivo mezcla varias series (por ejemplo, distintos indicadores o zonas). Una LSTM necesita una sola serie por fecha, así que los resultados no serían fiables. Filtra una serie antes de cargar el archivo.</div>}
      <div className="grid items-start gap-6 xl:grid-cols-[1fr_0.8fr]">
        <div className="space-y-6">
          <div className="card">
            <div className="mb-5 flex items-center gap-2"><span className="grid h-8 w-8 place-items-center rounded-lg bg-blue-50 text-sm font-bold text-blue-700">1</span><div><h2 className="section-title">Variables del modelo</h2><p className="text-xs text-muted">Selecciona qué información entra a la red</p></div></div>
            <div className="grid gap-4 sm:grid-cols-2">
              <label><span className="field-label">Columna temporal</span><select className="field" value={config.date_column} onChange={(event) => onConfig("date_column", event.target.value)}>{dateColumns.map((column) => <option key={column} value={column}>{column}</option>)}</select></label>
              <label><span className="field-label">Variable objetivo (target)</span><select className="field" value={config.target_column} onChange={(event) => onConfig("target_column", event.target.value)}>{numericColumns.map((column) => <option key={column} value={column}>{column}</option>)}</select></label>
            </div>
            <div className="mt-5">
              <p className="field-label">Variables predictoras <span className="font-normal text-muted">({config.feature_columns.length} seleccionadas)</span></p>
              <div className="grid gap-2 sm:grid-cols-2">
                {numericColumns.map((column) => {
                  const checked = config.feature_columns.includes(column);
                  return <label key={column} className={`flex cursor-pointer items-center gap-3 rounded-xl border px-3.5 py-3 text-sm transition ${checked ? "border-blue-200 bg-blue-50/60 text-blue-900" : "border-slate-200 text-slate-600 hover:bg-slate-50"}`}>
                    <input type="checkbox" className="h-4 w-4 accent-blue-600" checked={checked} onChange={(event) => onConfig("feature_columns", event.target.checked ? [...config.feature_columns, column] : config.feature_columns.filter((item) => item !== column))} />
                    <span className="flex-1">{column}</span>
                    {column === config.target_column && <span className="text-[10px] text-slate-400">histórico</span>}
                  </label>;
                })}
              </div>
              <p className="mt-2 text-xs text-muted">El target también puede usarse como predictor para aportar sus valores históricos.</p>
            </div>
          </div>
          <div className="card">
            <div className="mb-5 flex items-center gap-2"><span className="grid h-8 w-8 place-items-center rounded-lg bg-violet-50 text-sm font-bold text-violet-700">2</span><div><h2 className="section-title">Hiperparámetros</h2><p className="text-xs text-muted">Ajusta la arquitectura y el entrenamiento</p></div></div>
            <div className="grid gap-x-4 gap-y-4 sm:grid-cols-2">
              <NumberField label="Longitud de secuencia (lookback)" value={config.lookback} min={2} max={500} onChange={(value) => onConfig("lookback", value)} />
              <NumberField label="Épocas máximas" value={config.epochs} min={1} max={300} onChange={(value) => onConfig("epochs", value)} />
              <NumberField label="Tamaño del lote" value={config.batch_size} min={1} max={512} onChange={(value) => onConfig("batch_size", value)} />
              <NumberField label="Unidades LSTM" value={config.units} min={4} max={512} onChange={(value) => onConfig("units", value)} />
              <NumberField label="Dropout" value={config.dropout} min={0} max={0.8} step={0.05} onChange={(value) => onConfig("dropout", value)} />
              <NumberField label="Paciencia (early stopping)" value={config.patience} min={1} max={50} onChange={(value) => onConfig("patience", value)} />
              <NumberField label="Pasos a pronosticar (0 = ninguno)" value={config.forecast_steps} min={0} max={200} onChange={(value) => onConfig("forecast_steps", value)} />
            </div>
            {config.forecast_steps > 0 && !config.feature_columns.includes(config.target_column) && (
              <p role="alert" className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">El pronóstico futuro requiere incluir el target entre las variables predictoras.</p>
            )}
          </div>
          <div className="card">
            <div className="mb-5 flex items-center gap-2"><span className="grid h-8 w-8 place-items-center rounded-lg bg-emerald-50 text-sm font-bold text-emerald-700">3</span><div><h2 className="section-title">Preparación de datos</h2><p className="text-xs text-muted">División cronológica y tratamiento de nulos</p></div></div>
            <div className="grid gap-4 sm:grid-cols-2">
              <NumberField label="Entrenamiento (%)" value={Math.round(config.train_ratio * 100)} min={51} max={89} onChange={(value) => onConfig("train_ratio", value / 100)} />
              <NumberField label="Validación (%)" value={Math.round(config.validation_ratio * 100)} min={5} max={39} onChange={(value) => onConfig("validation_ratio", value / 100)} />
              <NumberField label="Prueba (%)" value={Math.round(config.test_ratio * 100)} min={5} max={39} onChange={(value) => onConfig("test_ratio", value / 100)} />
              <label><span className="field-label">Valores faltantes</span><select className="field" value={config.missing_strategy} onChange={(event) => onConfig("missing_strategy", event.target.value as TrainConfig["missing_strategy"])}><option value="forward_fill">Completar con el último valor</option><option value="interpolate">Interpolación (solo entrenamiento)</option><option value="drop">Eliminar filas incompletas</option></select></label>
            </div>
            <div className={`mt-4 flex items-center justify-between rounded-xl px-4 py-3 text-sm ${ratioValid ? "bg-emerald-50 text-emerald-800" : "bg-amber-50 text-amber-800"}`}><span>Proporción total · {Math.round(ratioTotal * 100)}%</span><span className="font-semibold">{ratioValid ? `Cronológica ${Math.round(config.train_ratio * 100)} / ${Math.round(config.validation_ratio * 100)} / ${Math.round(config.test_ratio * 100)}` : "Debe sumar 100%"}</span></div>
            <p className="mt-3 text-xs leading-relaxed text-muted">El escalado se ajusta únicamente con el segmento de entrenamiento. Validación y prueba conservan su orden temporal.</p>
          </div>
          <button type="button" className="primary-button w-full py-3" disabled={!canTrain || !ratioValid} onClick={onStart}>
            {busy ? <LoaderCircle className="animate-spin" size={17} /> : <Play size={16} fill="currentColor" />}
            Iniciar entrenamiento
          </button>
          {running && (
            <button type="button" className="secondary-button w-full py-3" onClick={onCancel}>
              <Square size={15} />Cancelar entrenamiento
            </button>
          )}
        </div>
        <div className="space-y-6 xl:sticky xl:top-24">
          <div className="card">
            <div className="mb-4 flex items-center gap-2"><Target size={18} className="text-blue-600" /><h2 className="section-title">Arquitectura</h2></div>
            <div className="space-y-2.5">
              {[
                ["Entrada", `${config.lookback} pasos × ${config.feature_columns.length} variables`],
                ["LSTM", `${config.units} unidades`],
                ["Regularización", `Dropout ${config.dropout}`],
                ["Capa densa", "32 neuronas · ReLU"],
                ["Salida", "1 valor · regresión"],
                ["Optimizador / pérdida", "Adam · MSE"],
              ].map(([label, value], index) => <div key={label} className="flex items-center gap-3"><span className={`h-2 w-2 rounded-full ${index === 0 || index === 5 ? "bg-blue-500" : "bg-slate-300"}`} /><p className="flex-1 text-xs text-slate-500">{label}</p><p className="text-right text-xs font-semibold text-slate-700">{value}</p></div>)}
            </div>
          </div>
          {job && (
            <div className="card">
              <div className="mb-4 flex items-start justify-between gap-3"><div><h2 className="section-title">Progreso del entrenamiento</h2><p className="mt-1 text-xs capitalize text-muted">{job.status === "queued" ? "En cola" : job.status === "running" ? "Entrenando modelo" : job.status === "completed" ? "Completado" : job.status === "cancelled" ? "Cancelado" : "Fallido"}</p></div>{running && <LoaderCircle size={18} className="animate-spin text-blue-600" />}{job.status === "completed" && <Check size={19} className="text-emerald-600" />}</div>
              <div className="mb-2 flex justify-between text-xs"><span className="font-medium text-slate-600">Época {job.current_epoch} de {job.total_epochs}</span><span className="text-muted">{progress}%</span></div>
              <div className="h-2 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-blue-600 transition-all duration-500" style={{ width: `${progress}%` }} /></div>
              <div className="mt-4 grid grid-cols-2 gap-3"><SmallMetric label="Pérdida entrenamiento" value={formatNumber(job.train_loss, 5)} /><SmallMetric label="Pérdida validación" value={formatNumber(job.validation_loss, 5)} /></div>
              {lossChart.length > 1 && <div className="mt-5 h-48"><ResponsiveContainer width="100%" height="100%"><LineChart data={lossChart}><CartesianGrid stroke="#edf0f5" strokeDasharray="4 4" vertical={false} /><XAxis dataKey="epoch" tick={{ fontSize: 10 }} /><YAxis tick={{ fontSize: 10 }} width={45} /><Tooltip /><Legend wrapperStyle={{ fontSize: 11 }} /><Line dataKey="train" name="Entrenamiento" stroke="#3567e8" dot={false} /><Line dataKey="validation" name="Validación" stroke="#9b67dc" dot={false} /></LineChart></ResponsiveContainer></div>}
              {job.status === "completed" && <button type="button" className="secondary-button mt-4 w-full" onClick={onResults}>Ver evaluación <ChevronRight size={15} /></button>}
              {job.status === "failed" && job.error && <p className="mt-4 rounded-lg bg-red-50 p-3 text-xs text-red-700">{job.error}</p>}
            </div>
          )}
          {!job && <div className="rounded-2xl border border-blue-100 bg-gradient-to-br from-blue-50 to-indigo-50 p-5"><p className="text-sm font-semibold text-blue-900">Entrenamiento transparente</p><p className="mt-2 text-xs leading-relaxed text-blue-800/75">Se detiene automáticamente cuando la pérdida de validación deja de mejorar y los resultados se calculan sobre el conjunto de prueba.</p></div>}
        </div>
      </div>
    </>
  );
}

function NumberField({ label, value, min, max, step = 1, onChange }: {
  label: string; value: number; min: number; max: number; step?: number; onChange: (value: number) => void;
}) {
  return <label><span className="field-label">{label}</span><input className="field" type="number" value={value} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} /></label>;
}

function ResultsSection({ results, runs, jobId, target, onTrain }: {
  results: TrainingResults | null; runs: RunRecord[]; jobId: string | null; target: string; onTrain: () => void;
}) {
  if (!results) {
    return (
      <>
        <PageHeading eyebrow="Evaluación del modelo" title="Resultados" description="Las métricas y predicciones aparecerán cuando finalice el entrenamiento." />
        <div className="card flex min-h-64 flex-col items-center justify-center text-center"><span className="mb-4 grid h-12 w-12 place-items-center rounded-xl bg-blue-50 text-blue-600"><TrendingUp size={22} /></span><p className="font-semibold text-ink">Todavía no hay resultados</p><p className="mt-1 text-sm text-muted">Inicia un entrenamiento para evaluar las predicciones sobre datos de prueba.</p><button type="button" className="primary-button mt-5" onClick={onTrain}><Settings2 size={16} />Configurar modelo</button></div>
      </>
    );
  }
  const metricCards = [
    { label: "MAE", value: formatNumber(results.metrics.mae), hint: "Error absoluto medio", icon: Activity, tone: "blue" as const },
    { label: "MSE", value: formatNumber(results.metrics.mse), hint: "Error cuadrático medio", icon: Gauge, tone: "violet" as const },
    { label: "RMSE", value: formatNumber(results.metrics.rmse), hint: "Error en unidades del target", icon: Target, tone: "green" as const },
    { label: "R²", value: formatNumber(results.metrics.r2), hint: "Coeficiente de determinación", icon: TrendingUp, tone: "amber" as const },
  ];
  return (
    <>
      <PageHeading eyebrow="Evaluación del modelo" title="Resultados del modelo" description={results.summary} action={jobId ? <a className="primary-button" href={api.predictionsUrl(jobId)}><ArrowDownToLine size={16} />Descargar CSV</a> : undefined} />
      <div className="mb-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {metricCards.map((metric) => <MetricCard key={metric.label} {...metric} />)}
      </div>
      <FullSeriesChart results={results} target={target} />
      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <ComparisonPanel metrics={results.metrics} baseline={results.baseline_metrics} />
        <ResidualsChart results={results} />
        <div className="card min-w-0 xl:col-span-2">
          <div className="mb-4"><h2 className="section-title">Pérdida por época</h2><p className="mt-1 text-sm text-muted">Evolución durante el entrenamiento</p></div>
          <div className="h-[310px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={results.losses.train.map((loss, index) => ({ epoch: index + 1, train: loss, validation: results.losses.validation[index] }))}>
                <CartesianGrid stroke="#edf0f5" strokeDasharray="4 4" vertical={false} />
                <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: "#8290a5" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: "#8290a5" }} axisLine={false} tickLine={false} width={48} />
                <Tooltip formatter={(value) => formatNumber(Number(value), 5)} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line type="monotone" dataKey="train" name="Entrenamiento" stroke="#3567e8" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="validation" name="Validación" stroke="#9b67dc" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
      <div className="card mt-6">
        <div className="mb-4 flex items-center justify-between gap-4"><div><h2 className="section-title">Detalle de predicciones</h2><p className="mt-1 text-sm text-muted">Primeras {Math.min(100, results.predictions.length)} de {results.predictions.length.toLocaleString("es")} observaciones de prueba</p></div>{jobId && <a className="secondary-button !px-3 !py-2 sm:hidden" href={api.predictionsUrl(jobId)} aria-label="Descargar predicciones"><ArrowDownToLine size={16} /></a>}</div>
        <div className="max-h-[440px] overflow-auto">
          <table className="data-table min-w-full text-left text-xs">
            <thead className="sticky top-0"><tr><th>Fecha</th><th>Valor real</th><th>Predicción</th><th>Ingenuo</th><th>Error absoluto</th></tr></thead>
            <tbody>{results.predictions.slice(0, 100).map((row, index) => <tr key={`${row.date}-${index}`}><td>{new Date(row.date).toLocaleString("es")}</td><td>{formatNumber(row.actual, 5)}</td><td className="font-semibold text-blue-700">{formatNumber(row.predicted, 5)}</td><td>{formatNumber(row.naive, 5)}</td><td>{formatNumber(Math.abs(row.actual - row.predicted), 5)}</td></tr>)}</tbody>
          </table>
        </div>
      </div>
      {results.forecast.length > 0 && (
        <div className="card mt-6">
          <div className="mb-4"><h2 className="section-title">Pronóstico futuro</h2><p className="mt-1 text-sm text-muted">{results.forecast.length} pasos posteriores al último dato observado</p></div>
          <div className="max-h-[320px] overflow-auto">
            <table className="data-table min-w-full text-left text-xs">
              <thead className="sticky top-0"><tr><th>Fecha</th><th>Valor pronosticado</th></tr></thead>
              <tbody>{results.forecast.map((row) => <tr key={row.date}><td>{new Date(row.date).toLocaleString("es")}</td><td className="font-semibold text-rose-600">{formatNumber(row.predicted, 5)}</td></tr>)}</tbody>
            </table>
          </div>
        </div>
      )}
      <RunHistory runs={runs} />
    </>
  );
}

function shortDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString("es", { month: "short", day: "numeric" });
}

function ChartLoading() {
  return <div className="flex h-full items-center justify-center gap-2 text-sm text-muted"><LoaderCircle size={17} className="animate-spin" />Calculando datos en el servidor…</div>;
}

function EmptyChart({ text }: { text: string }) {
  return <div className="flex min-h-40 items-center justify-center rounded-xl bg-slate-50 px-4 text-center text-sm text-muted">{text}</div>;
}

export default App;

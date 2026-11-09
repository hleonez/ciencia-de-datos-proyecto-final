import { useMemo } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { fullDate, formatNumber, shortDate } from "../format";
import type { Metrics, RunRecord, TrainingResults } from "../types";

const axisTick = { fontSize: 10, fill: "#8290a5" };

type Row = { date: string; actual?: number; predicted?: number; naive?: number; forecast?: number };

export function improvement(model: Metrics, baseline: Metrics): number | null {
  return baseline.rmse > 0 ? (1 - model.rmse / baseline.rmse) * 100 : null;
}

export function FullSeriesChart({ results, target }: { results: TrainingResults; target: string }) {
  const data = useMemo(() => {
    const rows = new Map<string, Row>();
    for (const point of results.series) rows.set(point.date, { date: point.date, actual: point.actual });
    for (const point of results.predictions) {
      rows.set(point.date, { ...rows.get(point.date), date: point.date, predicted: point.predicted, naive: point.naive });
    }
    const last = results.series[results.series.length - 1];
    if (results.forecast.length && last) {
      rows.set(last.date, { ...rows.get(last.date), date: last.date, forecast: last.actual });
    }
    for (const point of results.forecast) rows.set(point.date, { date: point.date, forecast: point.predicted });
    return [...rows.values()];
  }, [results]);
  const split = results.split_dates;
  const names: Record<string, string> = {
    actual: "Valor real", predicted: "Predicción LSTM", naive: "Modelo ingenuo", forecast: "Pronóstico futuro",
  };
  return (
    <div className="card min-w-0">
      <div className="mb-4">
        <h2 className="section-title">Serie completa, predicción y pronóstico</h2>
        <p className="mt-1 text-sm text-muted">
          Zonas de entrenamiento, validación y prueba · target: {target}
          {results.forecast.length ? ` · ${results.forecast.length} pasos pronosticados` : ""}
        </p>
      </div>
      <div className="h-[340px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 10, left: 4, bottom: 4 }}>
            <CartesianGrid stroke="#edf0f5" strokeDasharray="4 4" vertical={false} />
            <ReferenceArea x1={split.start} x2={split.train_end} fill="#3b72eb" fillOpacity={0.06} label={{ value: "Entrenamiento", fontSize: 10, fill: "#64748b", position: "insideTop" }} />
            <ReferenceArea x1={split.validation_start} x2={split.validation_end} fill="#f59e0b" fillOpacity={0.1} label={{ value: "Validación", fontSize: 10, fill: "#64748b", position: "insideTop" }} />
            <ReferenceArea x1={split.test_start} x2={split.end} fill="#10b981" fillOpacity={0.1} label={{ value: "Prueba", fontSize: 10, fill: "#64748b", position: "insideTop" }} />
            <XAxis dataKey="date" tickFormatter={shortDate} tick={axisTick} axisLine={false} tickLine={false} minTickGap={40} />
            <YAxis tick={axisTick} axisLine={false} tickLine={false} width={48} domain={["auto", "auto"]} />
            <Tooltip labelFormatter={(value) => fullDate(String(value))} formatter={(value, name) => [formatNumber(Number(value)), names[String(name)] ?? name]} />
            <Legend formatter={(value) => names[String(value)] ?? value} wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="actual" stroke="#263b5e" strokeWidth={1.6} dot={false} connectNulls />
            <Line type="monotone" dataKey="naive" stroke="#94a3b8" strokeWidth={1.4} strokeDasharray="4 3" dot={false} connectNulls />
            <Line type="monotone" dataKey="predicted" stroke="#3b72eb" strokeWidth={2} dot={false} connectNulls />
            <Line type="monotone" dataKey="forecast" stroke="#e11d48" strokeWidth={2} strokeDasharray="6 4" dot={false} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </div>
      {results.forecast_note && <p className="mt-3 text-xs leading-relaxed text-muted">{results.forecast_note}</p>}
    </div>
  );
}

export function ResidualsChart({ results }: { results: TrainingResults }) {
  const data = results.predictions.map((row) => ({
    date: row.date,
    lstm: row.actual - row.predicted,
    naive: row.actual - row.naive,
  }));
  const names: Record<string, string> = { lstm: "Error LSTM", naive: "Error ingenuo" };
  return (
    <div className="card min-w-0">
      <div className="mb-4">
        <h2 className="section-title">Residuos en el tiempo</h2>
        <p className="mt-1 text-sm text-muted">Real menos predicho. Un buen modelo oscila alrededor de cero sin patrón.</p>
      </div>
      <div className="h-[310px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 10, left: 4, bottom: 4 }}>
            <CartesianGrid stroke="#edf0f5" strokeDasharray="4 4" vertical={false} />
            <XAxis dataKey="date" tickFormatter={shortDate} tick={axisTick} axisLine={false} tickLine={false} minTickGap={35} />
            <YAxis tick={axisTick} axisLine={false} tickLine={false} width={48} />
            <Tooltip labelFormatter={(value) => fullDate(String(value))} formatter={(value, name) => [formatNumber(Number(value)), names[String(name)] ?? name]} />
            <Legend formatter={(value) => names[String(value)] ?? value} wrapperStyle={{ fontSize: 12 }} />
            <ReferenceLine y={0} stroke="#cbd5e1" />
            <Line type="monotone" dataKey="naive" stroke="#94a3b8" strokeWidth={1.4} strokeDasharray="4 3" dot={false} />
            <Line type="monotone" dataKey="lstm" stroke="#3b72eb" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function ComparisonPanel({ metrics, baseline }: { metrics: Metrics; baseline: Metrics }) {
  const gain = improvement(metrics, baseline);
  const better = gain != null && gain > 0;
  const rows: Array<[string, number | null, number | null]> = [
    ["MAE", metrics.mae, baseline.mae],
    ["RMSE", metrics.rmse, baseline.rmse],
    ["R²", metrics.r2, baseline.r2],
  ];
  return (
    <div className="card">
      <div className="mb-4">
        <h2 className="section-title">¿Es bueno el modelo?</h2>
        <p className="mt-1 text-sm text-muted">Comparación con un modelo ingenuo que repite el valor anterior ("mañana igual que hoy").</p>
      </div>
      <table className="data-table min-w-full text-left text-xs">
        <thead><tr><th>Métrica</th><th>LSTM</th><th>Ingenuo</th></tr></thead>
        <tbody>{rows.map(([label, model, base]) => <tr key={label}><td className="font-semibold text-ink">{label}</td><td className="font-semibold text-blue-700">{formatNumber(model)}</td><td>{formatNumber(base)}</td></tr>)}</tbody>
      </table>
      <p className={`mt-4 rounded-xl px-4 py-3 text-sm ${better ? "bg-emerald-50 text-emerald-800" : "bg-amber-50 text-amber-800"}`}>
        {gain == null
          ? "No se puede comparar: el modelo ingenuo no tiene error."
          : better
            ? `La LSTM reduce el RMSE un ${formatNumber(gain, 1)} % frente al modelo ingenuo.`
            : `La LSTM no supera al modelo ingenuo (RMSE ${formatNumber(gain, 1)} %). Prueba más épocas, otro lookback, incluir el target como predictor o más datos.`}
      </p>
    </div>
  );
}

export function RunHistory({ runs }: { runs: RunRecord[] }) {
  if (runs.length < 2) return null;
  return (
    <div className="card mt-6">
      <div className="mb-4">
        <h2 className="section-title">Historial de entrenamientos</h2>
        <p className="mt-1 text-sm text-muted">Compara configuraciones de esta sesión. Se pierde al recargar la página.</p>
      </div>
      <div className="overflow-x-auto">
        <table className="data-table min-w-full text-left text-xs">
          <thead><tr><th>#</th><th>Hora</th><th>Target</th><th>Predictores</th><th>Lookback</th><th>Épocas</th><th>Unidades</th><th>RMSE</th><th>MAE</th><th>R²</th><th>vs ingenuo</th></tr></thead>
          <tbody>
            {runs.map((run, index) => {
              const gain = improvement(run.metrics, run.baseline);
              return (
                <tr key={run.id}>
                  <td>{index + 1}</td><td>{run.finishedAt}</td><td>{run.target}</td><td>{run.predictors}</td>
                  <td>{run.lookback}</td><td>{run.epochs}</td><td>{run.units}</td>
                  <td className="font-semibold text-blue-700">{formatNumber(run.metrics.rmse)}</td>
                  <td>{formatNumber(run.metrics.mae)}</td><td>{formatNumber(run.metrics.r2)}</td>
                  <td className={gain != null && gain > 0 ? "text-emerald-700" : "text-amber-700"}>{gain == null ? "—" : `${formatNumber(gain, 1)} %`}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

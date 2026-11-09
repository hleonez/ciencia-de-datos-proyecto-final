import { useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { fullDate, formatNumber, shortDate } from "../format";
import type { Boxplot, Eda } from "../types";

const axisTick = { fontSize: 10, fill: "#8290a5" };

export function DistributionPanel({ column, bins, box }: {
  column: string;
  bins: Eda["histograms"][string] | undefined;
  box: Boxplot | null | undefined;
}) {
  const data = (bins ?? []).map((bin) => ({
    label: formatNumber(bin.start, 2),
    range: `${formatNumber(bin.start, 2)} – ${formatNumber(bin.end, 2)}`,
    count: bin.count,
  }));
  return (
    <div className="card">
      <div className="mb-4">
        <h2 className="section-title">Distribución de {column || "la variable"}</h2>
        <p className="mt-1 text-sm text-muted">Histograma y diagrama de caja para detectar sesgo y valores atípicos</p>
      </div>
      {data.length ? (
        <div className="h-[220px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="#edf0f5" strokeDasharray="4 4" vertical={false} />
              <XAxis dataKey="label" tick={axisTick} axisLine={false} tickLine={false} minTickGap={20} />
              <YAxis tick={axisTick} axisLine={false} tickLine={false} width={36} allowDecimals={false} />
              <Tooltip formatter={(value) => [value, "Registros"]} labelFormatter={(_, items) => items[0]?.payload.range ?? ""} />
              <Bar dataKey="count" fill="#3567e8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : <p className="rounded-xl bg-slate-50 px-4 py-8 text-center text-sm text-muted">No hay valores para el histograma.</p>}
      {box && <BoxplotBar box={box} />}
    </div>
  );
}

function BoxplotBar({ box }: { box: Boxplot }) {
  const span = box.max - box.min || 1;
  const pos = (value: number) => `${((value - box.min) / span) * 100}%`;
  return (
    <div className="mt-5">
      <div className="relative mx-2 h-10" role="img" aria-label="Diagrama de caja">
        <div className="absolute top-1/2 h-0.5 -translate-y-1/2 bg-slate-400" style={{ left: pos(box.whisker_low), width: `${((box.whisker_high - box.whisker_low) / span) * 100}%` }} />
        <div className="absolute top-1.5 bottom-1.5 rounded border border-blue-400 bg-blue-100" style={{ left: pos(box.q1), width: `${Math.max(((box.q3 - box.q1) / span) * 100, 0.5)}%` }} />
        <div className="absolute top-1 bottom-1 w-0.5 bg-blue-700" style={{ left: pos(box.median) }} />
        {box.outliers.map((value, index) => (
          <span key={index} className="absolute top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-amber-500" style={{ left: pos(value) }} />
        ))}
      </div>
      <dl className="mt-3 grid grid-cols-2 gap-2 text-xs sm:grid-cols-5">
        {([["Mínimo", box.min], ["Q1", box.q1], ["Mediana", box.median], ["Q3", box.q3], ["Máximo", box.max]] as const).map(([label, value]) => (
          <div key={label} className="rounded-lg bg-slate-50 px-2.5 py-2"><dt className="text-muted">{label}</dt><dd className="font-semibold text-ink">{formatNumber(value)}</dd></div>
        ))}
      </dl>
      <p className="mt-2 text-xs text-muted">
        {box.outlier_count ? `${box.outlier_count} valores atípicos (regla 1,5 × IQR), marcados en ámbar.` : "Sin valores atípicos según la regla 1,5 × IQR."}
      </p>
    </div>
  );
}

export function AcfChart({ acf, column }: { acf: Eda["acf"]; column: string }) {
  const data = acf.values.map((value, lag) => ({ lag, value })).slice(1);
  return (
    <div className="card min-w-0">
      <div className="mb-4">
        <h2 className="section-title">Autocorrelación (ACF)</h2>
        <p className="mt-1 text-sm text-muted">
          Correlación de {column || "la serie"} consigo misma a distintos rezagos. Los picos sugieren un buen valor de lookback; las barras fuera de la banda son significativas.
        </p>
      </div>
      {data.length ? (
        <div className="h-[240px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="#edf0f5" strokeDasharray="4 4" vertical={false} />
              <XAxis dataKey="lag" tick={axisTick} axisLine={false} tickLine={false} />
              <YAxis domain={[-1, 1]} tick={axisTick} axisLine={false} tickLine={false} width={36} />
              <Tooltip formatter={(value) => [formatNumber(Number(value)), "ACF"]} labelFormatter={(lag) => `Rezago ${lag}`} />
              {acf.confidence != null && <ReferenceLine y={acf.confidence} stroke="#9b67dc" strokeDasharray="4 4" />}
              {acf.confidence != null && <ReferenceLine y={-acf.confidence} stroke="#9b67dc" strokeDasharray="4 4" />}
              <ReferenceLine y={0} stroke="#cbd5e1" />
              <Bar dataKey="value" fill="#3567e8" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : <p className="rounded-xl bg-slate-50 px-4 py-8 text-center text-sm text-muted">Se necesitan al menos 6 observaciones válidas para calcular la ACF.</p>}
    </div>
  );
}

export function DecompositionPanel({ decomposition, detectedPeriod, onPeriod }: {
  decomposition: Eda["decomposition"];
  detectedPeriod: number | null;
  onPeriod: (period: number) => void;
}) {
  const [draft, setDraft] = useState("");
  const components = [
    { key: "observed", label: "Observada", color: "#263b5e" },
    { key: "trend", label: "Tendencia", color: "#3567e8" },
    { key: "seasonal", label: "Estacionalidad", color: "#16a34a" },
    { key: "residual", label: "Residuo", color: "#d97706" },
  ] as const;
  return (
    <div className="card">
      <div className="mb-4 flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
        <div>
          <h2 className="section-title">Tendencia y estacionalidad</h2>
          <p className="mt-1 text-sm text-muted">
            {decomposition
              ? `Descomposición aditiva con periodo de ${decomposition.period} observaciones${detectedPeriod ? "" : " (manual)"}.`
              : "No se detectó una estacionalidad clara. Puedes indicar un periodo manualmente."}
          </p>
        </div>
        <form className="flex items-end gap-2" onSubmit={(event) => { event.preventDefault(); const value = Number(draft); if (value >= 2) onPeriod(value); }}>
          <label><span className="field-label">Periodo</span><input className="field !w-24" type="number" min={2} placeholder="auto" value={draft} onChange={(event) => setDraft(event.target.value)} /></label>
          <button type="submit" className="secondary-button !px-3 !py-2" disabled={Number(draft) < 2}>Aplicar</button>
        </form>
      </div>
      {decomposition && (
        <div className="space-y-3">
          {components.map(({ key, label, color }) => (
            <div key={key}>
              <p className="mb-1 text-xs font-semibold text-slate-600">{label}</p>
              <div className="h-[110px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={decomposition.points} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid stroke="#edf0f5" strokeDasharray="4 4" vertical={false} />
                    <XAxis dataKey="date" tickFormatter={shortDate} tick={axisTick} axisLine={false} tickLine={false} minTickGap={40} />
                    <YAxis tick={axisTick} axisLine={false} tickLine={false} width={46} domain={["auto", "auto"]} />
                    <Tooltip labelFormatter={(value) => fullDate(String(value))} formatter={(value) => [formatNumber(Number(value)), label]} />
                    <Line type="monotone" dataKey={key} stroke={color} strokeWidth={1.8} dot={false} connectNulls={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

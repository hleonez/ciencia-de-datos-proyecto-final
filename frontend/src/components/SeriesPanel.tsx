import type { SeriesFilter, SeriesProfile } from "../types";

type Props = {
  profile: SeriesProfile | null;
  filters: SeriesFilter[];
  onFilter: (column: string, value: string, match: "exact" | "prefix") => void;
};

export function SeriesPanel({ profile, filters, onFilter }: Props) {
  if (!profile) {
    return <p className="mb-6 rounded-xl bg-slate-50 px-4 py-3 text-sm text-muted">Revisando si el archivo es una sola serie temporal…</p>;
  }
  const tone = profile.trainable ? "border-emerald-100 bg-emerald-50 text-emerald-900" : "border-amber-100 bg-amber-50 text-amber-900";
  return (
    <div className="card mb-6">
      <h2 className="section-title">Qué serie se va a predecir</h2>
      <p className="mt-1 text-sm text-muted">
        {profile.row_count.toLocaleString("es")} filas · {profile.unique_dates.toLocaleString("es")} fechas distintas
        {profile.suggested_lookback ? ` · lookback sugerido ${profile.suggested_lookback}` : ""}
      </p>
      {filters.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {filters.map((filter) => (
            <button key={filter.column} type="button" className="secondary-button !px-3 !py-1.5 text-xs" onClick={() => onFilter(filter.column, "", "exact")}>
              {filter.match === "prefix" ? `${filter.column} empieza por ${filter.value}` : `${filter.column} = ${filter.value}`} · quitar
            </button>
          ))}
        </div>
      )}
      {profile.categories.length > 0 && (
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          {profile.categories.map((category) => (
            <label key={category.name}>
              <span className="field-label">{category.name}</span>
              <select
                className="field"
                value={(() => { const current = filters.find((item) => item.column === category.name); return current ? `${current.match ?? "exact"}::${current.value}` : ""; })()}
                onChange={(event) => {
                  if (!event.target.value) { onFilter(category.name, "", "exact"); return; }
                  const [match, ...rest] = event.target.value.split("::");
                  onFilter(category.name, rest.join("::"), match === "prefix" ? "prefix" : "exact");
                }}
              >
                <option value="">Todas ({category.unique_count.toLocaleString("es")})</option>
                {category.options.map((option) => (
                  <option key={`${option.match}-${option.value}`} value={`${option.match}::${option.value}`}>
                    {option.match === "prefix" ? `Empieza por ${option.value}` : option.value} · {option.count.toLocaleString("es")}
                  </option>
                ))}
              </select>
              {category.truncated && <span className="mt-1 block text-xs text-muted">Se muestran las 200 categorías más frecuentes.</span>}
            </label>
          ))}
        </div>
      )}
      <p className={`mt-4 rounded-xl border px-4 py-3 text-sm leading-relaxed ${tone}`}>{profile.story}</p>
    </div>
  );
}

/** Builds a deterministic hourly demand series (daily seasonality + trend) for demos. */
export function buildSampleFile(): File {
  const rows = ["fecha,demanda,temperatura,humedad"];
  const start = Date.UTC(2025, 0, 1);
  for (let hour = 0; hour < 24 * 20; hour += 1) {
    const date = new Date(start + hour * 3_600_000).toISOString().slice(0, 19).replace("T", " ");
    const noise = Math.sin(hour * 12.9898) * 43758.5453;
    const jitter = (noise - Math.floor(noise) - 0.5) * 4;
    const demand = 120 + 25 * Math.sin((2 * Math.PI * hour) / 24) + hour * 0.04 + jitter;
    const temperature = 18 + 7 * Math.sin((2 * Math.PI * (hour - 6)) / 24) + jitter / 4;
    const humidity = 60 - 10 * Math.sin((2 * Math.PI * (hour - 6)) / 24) + jitter / 2;
    rows.push(`${date},${demand.toFixed(2)},${temperature.toFixed(2)},${humidity.toFixed(2)}`);
  }
  return new File([rows.join("\n")], "ejemplo_demanda_horaria.csv", { type: "text/csv" });
}

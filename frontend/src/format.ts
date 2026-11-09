export const formatNumber = (value: number | null | undefined, digits = 3) =>
  value == null || !Number.isFinite(value)
    ? "—"
    : new Intl.NumberFormat("es", { maximumFractionDigits: digits }).format(value);

export function shortDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleDateString("es", { month: "short", day: "numeric" });
}

export function fullDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("es");
}

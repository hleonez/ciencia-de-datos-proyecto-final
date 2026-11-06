import { useRef, useState } from "react";
import { FileSpreadsheet, UploadCloud } from "lucide-react";
import type { Dataset } from "../types";
import { api } from "../services/api";

type Props = {
  onUploaded: (dataset: Dataset) => void;
};

export function UploadPanel({ onUploaded }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function upload(file?: File) {
    if (!file) return;
    setError("");
    setLoading(true);
    try {
      onUploaded(await api.upload(file));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "No se pudo cargar el archivo.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          void upload(event.dataTransfer.files[0]);
        }}
        disabled={loading}
        className={`group flex min-h-64 w-full flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 text-center transition ${
          dragging ? "border-blue-500 bg-blue-50" : "border-slate-200 bg-white hover:border-blue-300 hover:bg-slate-50"
        }`}
      >
        <span className="mb-4 grid h-14 w-14 place-items-center rounded-2xl bg-blue-50 text-blue-600">
          {loading ? <span className="h-6 w-6 animate-spin rounded-full border-2 border-blue-600 border-t-transparent" /> : <UploadCloud size={25} />}
        </span>
        <span className="text-base font-semibold text-ink">
          {loading ? "Validando archivo…" : "Arrastra tu archivo aquí"}
        </span>
        <span className="mt-1 text-sm text-muted">o haz clic para buscar en tu equipo</span>
        <span className="mt-5 inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-600">
          <FileSpreadsheet size={15} /> CSV o XLSX · Máximo 20 MB
        </span>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.xlsx"
        className="hidden"
        onChange={(event) => { void upload(event.target.files?.[0]); event.target.value = ""; }}
      />
      {error && <p role="alert" className="mt-3 rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">{error}</p>}
    </div>
  );
}

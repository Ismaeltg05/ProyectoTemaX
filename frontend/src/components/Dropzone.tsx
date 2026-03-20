import React, { useMemo, useState } from "react";

type Props = {
  multiple: boolean;
  accept?: string;
  files: File[];
  onFilesChange: (files: File[]) => void;
  title: string;
  hint?: string;
};

function mergeFiles(existing: File[], incoming: File[], multiple: boolean) {
  if (!multiple) return incoming.length ? [incoming[0]] : [];
  const map = new Map<string, File>();
  for (const f of existing) map.set(`${f.name}-${f.size}-${f.lastModified}`, f);
  for (const f of incoming) map.set(`${f.name}-${f.size}-${f.lastModified}`, f);
  return Array.from(map.values());
}

export default function Dropzone({ multiple, accept = "image/*", files, onFilesChange, title, hint }: Props) {
  const [dragOver, setDragOver] = useState(false);

  const inputId = useMemo(() => `file-input-${Math.random().toString(16).slice(2)}`, []);

  const onPick = (list: FileList | null) => {
    const incoming = list ? Array.from(list) : [];
    onFilesChange(mergeFiles(files, incoming, multiple));
  };

  return (
    <div className="card">
      <div className="cardHeader">
        <h2>{title}</h2>
        {hint ? <p className="muted">{hint}</p> : null}
      </div>

      <div
        className={`dropzone ${dragOver ? "dragover" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          onPick(e.dataTransfer.files);
        }}
      >
        <div className="dropzoneInner">
          <div>
            <strong>Arrastra y suelta</strong> imágenes aquí
          </div>
          <div className="muted">o</div>

          <label className="btn secondary" htmlFor={inputId}>
            Seleccionar archivos
          </label>
          <input
            id={inputId}
            className="hidden"
            type="file"
            accept={accept}
            multiple={multiple}
            onChange={(e) => onPick(e.target.files)}
          />

          <div className="muted small">
            {multiple ? "Puedes seleccionar varias imágenes." : "Solo 1 imagen."}
          </div>
        </div>
      </div>

      <div className="row" style={{ marginTop: 12 }}>
        <span className="pill">{files.length} archivo(s)</span>
        <button
          className="btn secondary"
          disabled={files.length === 0}
          onClick={() => onFilesChange([])}
          type="button"
        >
          Limpiar
        </button>
      </div>
    </div>
  );
}
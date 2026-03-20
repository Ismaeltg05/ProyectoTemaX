import { useEffect, useMemo } from "react";

type Props = {
  files: File[];
};

type Preview = {
  key: string;
  name: string;
  sizeKb: number;
  url: string;
};

export default function PreviewGrid({ files }: Props) {
  const previews = useMemo<Preview[]>(() => {
    return files.map((f) => ({
      key: `${f.name}-${f.size}-${f.lastModified}`,
      name: f.name,
      sizeKb: Math.round(f.size / 1024),
      url: URL.createObjectURL(f)
    }));
  }, [files]);

  useEffect(() => {
    return () => {
      for (const p of previews) URL.revokeObjectURL(p.url);
    };
  }, [previews]);

  if (!files.length) return null;

  return (
    <div className="card">
      <div className="cardHeader">
        <h3>Vista previa</h3>
        <p className="muted">Comprueba las imágenes antes de enviarlas.</p>
      </div>

      <div className="previewGrid">
        {previews.map((p) => (
          <div key={p.key} className="previewItem">
            <img src={p.url} alt={p.name} />
            <div className="previewMeta">
              <div className="previewName" title={p.name}>{p.name}</div>
              <div className="muted small">{p.sizeKb} KB</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
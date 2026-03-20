import { useMemo, useState } from "react";
import {
  getApiBaseUrl,
  healthCheck,
  multipredict,
  predict,
  type MultiPredictResponse,
  type PredictResponse
} from "./api";
import Dropzone from "./components/Dropzone";
import PreviewGrid from "./components/PreviewGrid";
import { MultiPredictResultCard, PredictResultCard } from "./components/ResultView";

type Mode = "predict" | "multipredict";

export default function App() {
  const [mode, setMode] = useState<Mode>("predict");
  const [busy, setBusy] = useState(false);

  const [predictFiles, setPredictFiles] = useState<File[]>([]);
  const [multiFiles, setMultiFiles] = useState<File[]>([]);

  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null);
  const [predictFileName, setPredictFileName] = useState<string>("");
  const [multiResult, setMultiResult] = useState<MultiPredictResponse | null>(null);

  const [error, setError] = useState<string>("");

  const apiBase = useMemo(() => getApiBaseUrl(), []);

  const clearResults = () => {
    setPredictResult(null);
    setMultiResult(null);
    setError("");
  };

  const runHealth = async () => {
    setBusy(true);
    setError("");
    try {
      const data = await healthCheck();
      alert(`API OK: ${JSON.stringify(data)}`);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const onSend = async () => {
    clearResults();
    setBusy(true);
    setError("");

    try {
      if (mode === "predict") {
        const f = predictFiles[0];
        if (!f) throw new Error("Selecciona una imagen para /predict.");
        setPredictFileName(f.name);
        const res = await predict(f);
        setPredictResult(res);
      } else {
        if (!multiFiles.length) throw new Error("Selecciona al menos una imagen para /multipredict.");
        const res = await multipredict(multiFiles);
        setMultiResult(res);
      }
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const disabledSend = busy || (mode === "predict" ? predictFiles.length !== 1 : multiFiles.length === 0);

  return (
    <div className="container">
      <header className="topbar">
        <div>
          <h1>Food Classification</h1>
          <p className="muted">
            API: <span className="mono">{apiBase}</span>
          </p>
        </div>

        <div className="row">
          <button className="btn secondary" onClick={runHealth} disabled={busy} type="button">
            Probar API (/)
          </button>
          <button className="btn" onClick={onSend} disabled={disabledSend} type="button">
            {busy ? "Procesando..." : "Enviar"}
          </button>
        </div>
      </header>

      <nav className="tabs">
        <button
          className={`tab ${mode === "predict" ? "active" : ""}`}
          onClick={() => {
            setMode("predict");
            clearResults();
          }}
          type="button"
        >
          Predict (1)
        </button>

        <button
          className={`tab ${mode === "multipredict" ? "active" : ""}`}
          onClick={() => {
            setMode("multipredict");
            clearResults();
          }}
          type="button"
        >
          MultiPredict (N)
        </button>
      </nav>

      {error ? (
        <div className="card errorCard">
          <div className="cardHeader">
            <h3>Error</h3>
          </div>
          <div className="errorText">{error}</div>
        </div>
      ) : null}

      {mode === "predict" ? (
        <>
          <Dropzone
            multiple={false}
            files={predictFiles}
            onFilesChange={(fs) => {
              setPredictFiles(fs.slice(0, 1));
              clearResults();
            }}
            title="Predict (una imagen)"
            hint="Envía 1 imagen al endpoint /predict. Arrastra y suelta o selecciónala."
          />
          <PreviewGrid files={predictFiles} />
          {predictResult ? <PredictResultCard fileName={predictFileName} result={predictResult} /> : null}
        </>
      ) : (
        <>
          <Dropzone
            multiple={true}
            files={multiFiles}
            onFilesChange={(fs) => {
              setMultiFiles(fs);
              clearResults();
            }}
            title="MultiPredict (varias imágenes)"
            hint="Envía N imágenes al endpoint /multipredict. Se mostrará una predicción por archivo y errores por separado."
          />
          <PreviewGrid files={multiFiles} />
          {multiResult ? <MultiPredictResultCard data={multiResult} /> : null}
        </>
      )}

      <footer className="footer muted">
        Tip: en Docker Compose usa <span className="mono">VITE_API_BASE_URL=http://backend:8000</span>.
      </footer>
    </div>
  );
}
import React from "react";
import type { MultiPredictResponse, PredictResponse } from "../api";

export function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  return (
    <div className="confWrap" title={`${pct}%`}>
      <div className="confBar" style={{ width: `${pct}%` }} />
      <div className="confText">{pct}%</div>
    </div>
  );
}

export function PredictResultCard({ fileName, result }: { fileName: string; result: PredictResponse }) {
  return (
    <div className="card">
      <div className="cardHeader">
        <h3>Resultado (1 imagen)</h3>
        <p className="muted">{fileName}</p>
      </div>

      <div className="resultRow">
        <div>
          <div className="label">Clase</div>
          <div className="value">
            <span className="pill strong">{result.predicted_class}</span>
          </div>
        </div>
        <div>
          <div className="label">Confianza</div>
          <ConfidenceBar value={result.confidence} />
        </div>
      </div>
    </div>
  );
}

export function MultiPredictResultCard({ data }: { data: MultiPredictResponse }) {
  return (
    <div className="card">
      <div className="cardHeader">
        <h3>Resultados (múltiples)</h3>
        <div className="row">
          <span className="pill">Total: {data.count}</span>
          <span className="pill ok">OK: {data.success}</span>
          <span className={`pill ${data.failed ? "err" : ""}`}>Fallidas: {data.failed}</span>
        </div>
      </div>

      {data.results?.length ? (
        <>
          <h4 style={{ marginTop: 8 }}>Predicciones</h4>
          <div className="tableWrap">
            <table>
              <thead>
                <tr>
                  <th>Archivo</th>
                  <th>Clase</th>
                  <th>Confianza</th>
                </tr>
              </thead>
              <tbody>
                {data.results.map((r) => (
                  <tr key={r.filename}>
                    <td className="mono">{r.filename}</td>
                    <td><span className="pill strong">{r.predicted_class}</span></td>
                    <td><ConfidenceBar value={r.confidence} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}

      {data.errors?.length ? (
        <>
          <h4 style={{ marginTop: 16 }} className="errorText">Errores</h4>
          <div className="tableWrap">
            <table>
              <thead>
                <tr>
                  <th>Archivo</th>
                  <th>Detalle</th>
                </tr>
              </thead>
              <tbody>
                {data.errors.map((e, idx) => (
                  <tr key={`${e.filename}-${idx}`}>
                    <td className="mono">{e.filename}</td>
                    <td className="errorText">{e.error}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}
    </div>
  );
}
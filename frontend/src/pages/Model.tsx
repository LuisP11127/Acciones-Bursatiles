import { useJson } from "../api";
import { SeriesChart } from "../components/SeriesChart";
import { Card, ErrorBox, Freshness, Loading, Stat, StatusChip } from "../components/ui";
import { date, dateTime, FEATURE_GROUP_LABELS, int, isNum, num, pct, STRATEGY_LABELS } from "../format";
import type { ModelJson, ModelVersion } from "../types";

const BASELINES: Record<string, string> = {
  statistical_score: "Score estadístico", momentum_3m: "Momentum 3 meses", logistic_regression: "Regresión logística", random: "Azar",
};

function Checks({ v }: { v: ModelVersion }) {
  if (!v.promotion_checks?.length) return null;
  return (
    <div className="table-wrap">
      <table>
        <thead><tr><th>Criterio de promoción</th><th>Resultado</th><th>Detalle</th></tr></thead>
        <tbody>
          {v.promotion_checks.map((c) => (
            <tr key={c.name}>
              <td>{c.name}</td>
              <td>{c.passed ? <StatusChip kind="good">cumple</StatusChip> : <StatusChip kind="critical">no cumple</StatusChip>}</td>
              <td className="secondary">{c.detail}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function VersionDetail({ v, titulo }: { v: ModelVersion; titulo: string }) {
  const agg = v.walk_forward?.aggregate ?? {};
  const bases = v.walk_forward?.baselines ?? {};
  const auc = agg.roc_auc as number | undefined;
  const maxImp = Math.max(...(v.feature_importance ?? []).map((f) => Math.abs(f.auc_drop)), 1e-6);
  return (
    <>
      <Card title={titulo} extra={<>{v.model_version} · {v.status === "production" ? <StatusChip kind="good">en producción</StatusChip> : <StatusChip kind={v.status === "rejected" ? "critical" : "neutral"}>{v.status}</StatusChip>}</>}>
        <div className="stats">
          <Stat label="Versión" value={v.model_version} sub={`features ${v.features_version}`} />
          <Stat label="Fecha de entrenamiento" value={date(v.training_date)} sub={`datos al ${date(v.data_as_of)}`} />
          <Stat label="Periodo de entrenamiento" value={<span style={{ fontSize: "1rem" }}>{date(v.training_period?.start)} → {date(v.training_period?.end)}</span>}
                sub={`validación ${date(v.validation_period?.start)} → ${date(v.validation_period?.end)}`} />
          <Stat label="Arquitectura" value={`${String(v.hyperparameters?.architecture ?? "").toUpperCase()} ${v.hyperparameters?.hidden_size}`}
                sub={`${int(v.n_parameters)} parámetros · secuencia de ${v.lookback} sesiones`} />
          <Stat label="AUC fuera de muestra" value={num(auc, 3)} sub={`${agg.folds ?? "—"} años de test walk-forward`} />
          <Stat label="Tasa base / umbral" value={`${pct(v.base_rate, 1, false)} / ${pct(v.threshold, 1, false)}`} sub="probabilidad mínima para operar" />
        </div>
        {v.label_definition && (
          <p className="muted" style={{ fontSize: "0.82rem", marginTop: 10 }}>
            Etiqueta (solo entrenamiento): {String(v.label_definition.description)} Horizonte {String(v.label_definition.horizon)} sesiones ·
            stop {pct(v.label_definition.stop_loss as number, 0, false)} · take-profit {pct(v.label_definition.take_profit as number, 0, false)} ·
            retorno mínimo {pct(v.label_definition.min_return as number, 1, false)}.
          </p>
        )}
      </Card>
      <Card title="Métricas fuera de muestra vs. benchmarks sencillos">
        <div className="stats">
          {([["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"], ["roc_auc", "ROC-AUC"], ["top_decile_precision", "Precisión decil superior"],
             ["top_decile_lift", "Lift decil superior"], ["brier", "Brier"], ["base_rate", "Tasa base"]] as [string, string][]).map(([k, l]) => (
            <Stat key={k} label={l} value={isNum(agg[k]) ? (["roc_auc", "top_decile_lift", "brier"].includes(k) ? num(agg[k] as number, 3) : pct(agg[k] as number, 1, false)) : "n/d"} />
          ))}
        </div>
        <h3 style={{ marginTop: 14 }}>ROC-AUC: red neuronal frente a alternativas sencillas</h3>
        <div className="bars">
          {[["red_neuronal", auc] as [string, number | undefined], ...Object.entries(bases)].map(([k, val]) => (
            <div className="bar-row" key={k}>
              <span>{k === "red_neuronal" ? <strong>Red neuronal</strong> : BASELINES[k] ?? k}</span>
              <span className="bar-track"><span className="bar-fill" style={{ width: `${isNum(val) ? Math.max(0, (val - 0.4) / 0.3) * 100 : 0}%` }} /></span>
              <span className="num">{num(val as number, 3)}</span>
            </div>
          ))}
        </div>
        <p className="muted" style={{ fontSize: "0.8rem", marginTop: 6 }}>0,5 = azar. Escala de 0,40 a 0,70.</p>
        {v.backtest && (
          <>
            <h3 style={{ marginTop: 14 }}>Resultado simulado (backtest con predicciones fuera de muestra)</h3>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Serie</th><th className="right">Retorno</th><th className="right">Máx. DD</th><th className="right">Sharpe</th><th className="right">Sortino</th><th className="right">Win rate</th><th className="right">Profit factor</th></tr></thead>
                <tbody>
                  {Object.entries(v.backtest).map(([k, m]) => (
                    <tr key={k}><td>{STRATEGY_LABELS[k] ?? k}</td><td className="right num">{pct(m.total_return)}</td><td className="right num">{pct(m.max_drawdown)}</td>
                      <td className="right num">{num(m.sharpe, 2)}</td><td className="right num">{num(m.sortino, 2)}</td>
                      <td className="right num">{pct(m.win_rate, 1, false)}</td><td className="right num">{num(m.profit_factor, 2)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </Card>
      <Card title="Validación walk-forward por año">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Test</th><th>Entrenamiento</th><th>Validación</th><th className="right">Ejemplos test</th><th className="right">AUC</th><th className="right">Precision</th><th className="right">Recall</th><th className="right">F1</th><th className="right">Lift top 10 %</th><th className="right">AUC score estadístico</th></tr></thead>
            <tbody>
              {(v.walk_forward?.folds ?? []).map((f) => (
                <tr key={f.name}>
                  <td><strong>{f.name}</strong></td>
                  <td>{date(f.train_start)} → {date(f.train_end)}</td>
                  <td>{date(f.val_start)} → {date(f.val_end)}</td>
                  <td className="right num">{int(f.test_samples)}</td>
                  <td className="right num">{num(f.metrics.roc_auc, 3)}</td>
                  <td className="right num">{pct(f.metrics.precision, 1, false)}</td>
                  <td className="right num">{pct(f.metrics.recall, 1, false)}</td>
                  <td className="right num">{num(f.metrics.f1, 3)}</td>
                  <td className="right num">{num(f.metrics.top_decile_lift, 2)}</td>
                  <td className="right num">{num(f.baselines_auc.statistical_score, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted" style={{ fontSize: "0.8rem" }}>Entre periodos se purgan sesiones para que ninguna etiqueta (que mira 20 sesiones adelante) cruce al periodo siguiente.</p>
      </Card>
      {v.feature_importance && v.feature_importance.length > 0 && (
        <Card title="Variables más influyentes (importancia por permutación)" extra="Explicación del modelo, no afirmaciones sobre empresas">
          <div className="bars">
            {v.feature_importance.slice(0, 15).map((f) => (
              <div className="bar-row" key={f.feature}>
                <span>{f.label}</span>
                <span className="bar-track"><span className={`bar-fill ${f.auc_drop < 0 ? "neg-fill" : ""}`} style={{ width: `${(Math.abs(f.auc_drop) / maxImp) * 100}%` }} /></span>
                <span className="num">{(f.auc_drop * 100).toFixed(2)}</span>
              </div>
            ))}
          </div>
          <p className="muted" style={{ fontSize: "0.8rem", marginTop: 6 }}>Caída del AUC (×100) al desordenar cada variable en el último año de test.</p>
        </Card>
      )}
      {v.training?.history && (
        <Card title="Entrenamiento del modelo" extra={`${v.training.epochs_run} épocas · mejor ${v.training.best_epoch} · ${v.training.seconds} s · ${int(v.training.train_samples)} ejemplos`}>
          <SeriesChart data={v.training.history.map((h) => ({ epoch: String(h.epoch), val_loss: h.val_loss, train_loss: h.train_loss ?? null }))}
                       xKey="epoch" height={220} yFormat={(x) => x.toFixed(3)}
                       series={[{ key: "train_loss", label: "Pérdida entrenamiento", slot: 0 }, { key: "val_loss", label: "Pérdida validación", slot: 1 }]} />
        </Card>
      )}
      <Card title="Decisión de promoción">
        <Checks v={v} />
        {v.compared_with && <p className="muted" style={{ fontSize: "0.82rem" }}>Comparado con el modelo en producción {v.compared_with}.</p>}
      </Card>
    </>
  );
}

export function Model() {
  const { data, error, loading } = useJson<ModelJson>("model.json");
  if (loading) return <Loading />;
  if (error || !data) return <ErrorBox message={error ?? "Sin datos"} />;
  const le = data.live_evaluation;
  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Modelo IA</h1>
          <Freshness generated={data.generated_at} asOf={data.data_as_of} />
        </div>
      </div>
      <p className="secondary">
        Red neuronal recurrente (PyTorch, {String(data.architecture.architecture ?? "gru").toUpperCase()}) que estima
        P(oportunidad | datos disponibles hasta T) a partir de secuencias de {String(data.architecture.lookback)} sesiones de
        {" "}{data.features.length} variables. Solo pasa a producción si supera los criterios de validación y no empeora al modelo anterior.
      </p>
      {data.production ? <VersionDetail v={data.production} titulo="Modelo en producción" /> : (
        <Card title="Modelo en producción"><div className="empty">{data.status_reason ?? "Todavía no hay un modelo promovido a producción."} Mientras tanto, el ranking funciona sin el componente neuronal.</div></Card>
      )}
      {data.latest_candidate && <VersionDetail v={data.latest_candidate} titulo="Último candidato (no promovido)" />}
      <Card title="Evaluación en vivo de las predicciones" extra="Snapshots diarios comprobados después del horizonte">
        {le.available ? (
          <>
            <div className="stats">
              <Stat label="Días evaluados" value={le.evaluated_days} sub={`${int(le.samples)} predicciones`} />
              <Stat label="Tasa base" value={pct(le.base_rate, 1, false)} />
              <Stat label="AUC score final" value={num(le.final_score_auc, 3)} />
              <Stat label="AUC score estadístico" value={num(le.statistical_score_auc, 3)} />
              <Stat label="AUC red neuronal" value={num(le.neural_auc, 3)} sub={`${int(le.neural_samples)} predicciones`} />
              <Stat label="Acierto top 10 diario" value={pct(le.top10_final_hit_rate, 1, false)} />
            </div>
            <SeriesChart data={(le.by_date ?? []).map((d) => ({ date: d.date, top10: d.top10_final_hit_rate, base: d.base_rate }))}
                         xKey="date" xFormat={date} height={220} yFormat={(v) => pct(v, 0, false)}
                         series={[{ key: "top10", label: "Acierto top 10 (score final)", slot: 0 }, { key: "base", label: "Tasa base", slot: 4 }]} />
            <p className="muted" style={{ fontSize: "0.8rem" }}>{le.note}</p>
          </>
        ) : <div className="empty">{le.note}</div>}
      </Card>
      <Card title="Historial de versiones">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Versión</th><th>Estado</th><th>Entrenado</th><th>Periodo</th><th>Features</th><th className="right">AUC OOS</th><th className="right">Lift</th><th className="right">Sharpe combinado</th><th>Criterios</th></tr></thead>
            <tbody>
              {[...data.registry].reverse().map((r) => (
                <tr key={r.model_version}>
                  <td><strong>{r.model_version}</strong></td>
                  <td>{r.status === "production" ? <StatusChip kind="good">producción</StatusChip> : r.status === "rejected" ? <StatusChip kind="critical">rechazado</StatusChip> : <StatusChip kind="neutral">{r.status}</StatusChip>}</td>
                  <td>{date(r.training_date)}</td>
                  <td>{r.training_period ? `${date(r.training_period.start)} → ${date(r.training_period.end)}` : "—"}</td>
                  <td>{r.features_version}</td>
                  <td className="right num">{num(r.oos_auc, 3)}</td>
                  <td className="right num">{num(r.top_decile_lift, 2)}</td>
                  <td className="right num">{num(r.combined_sharpe, 2)}</td>
                  <td>{r.checks.filter((c) => c.passed).length}/{r.checks.length}{r.promoted_at && <div className="muted" style={{ fontSize: "0.78rem" }}>promovido {dateTime(r.promoted_at)}</div>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!data.registry.length && <div className="empty">Aún no se ha entrenado ningún modelo (workflow semanal «Model training»).</div>}
      </Card>
      <Card title="Variables de entrada">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Grupo</th><th>Variables</th></tr></thead>
            <tbody>
              {[...new Set(data.features.map((f) => f.group))].map((g) => (
                <tr key={g}><td>{FEATURE_GROUP_LABELS[g] ?? g}</td><td>{data.features.filter((f) => f.group === g).map((f) => f.label).join(" · ")}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted" style={{ fontSize: "0.8rem" }}>Todas son relativas (retornos, ratios, osciladores) y usan solo datos hasta la fecha de decisión. El OHLCV entra normalizado.</p>
      </Card>
    </div>
  );
}

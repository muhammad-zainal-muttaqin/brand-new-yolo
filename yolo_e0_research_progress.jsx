import { useState, useMemo, useRef, useEffect } from "react";

const RAW_DATA = [
  {idx:0,run:"p0_yolo11n_640_s1_e3",label:"YOLO11n 640 s1 e3",phase:"phase0",precision:0.4379,recall:0.5519,map50:0.4450,map50_95:0.1982},
  {idx:1,run:"p0_yolo11n_1024_s1_e3",label:"YOLO11n 1024 s1 e3",phase:"phase0",precision:0.4123,recall:0.5759,map50:0.4544,map50_95:0.2119},
  {idx:2,run:"p0_yolo11n_640_s2_e3",label:"YOLO11n 640 s2 e3",phase:"phase0",precision:0.4241,recall:0.5474,map50:0.4487,map50_95:0.2074},
  {idx:3,run:"p0_yolo11n_1024_s2_e3",label:"YOLO11n 1024 s2 e3",phase:"phase0",precision:0.4189,recall:0.5530,map50:0.4472,map50_95:0.2126},
  {idx:4,run:"p0_yolo11n_640_s1_e30",label:"YOLO11n 640 e30",phase:"phase0",precision:0.4906,recall:0.5864,map50:0.5237,map50_95:0.2538},
  {idx:5,run:"p0_yolo11n_1024_s1_e30",label:"YOLO11n 1024 e30",phase:"phase0",precision:0.4888,recall:0.6016,map50:0.5363,map50_95:0.2571},
  {idx:6,run:"p0_yolo11n_640_s1_e30p10m30",label:"640 patience10 min30",phase:"phase0",precision:0.4906,recall:0.5864,map50:0.5237,map50_95:0.2538},
  {idx:7,run:"p0_yolo11n_1024_s1_e30p10m30",label:"1024 patience10 min30",phase:"phase0",precision:0.4888,recall:0.6016,map50:0.5363,map50_95:0.2571},
  {idx:8,run:"p0_yolo11n_640_s2_e30p10m30",label:"640 s2 p10m30",phase:"phase0",precision:0.4923,recall:0.5838,map50:0.5245,map50_95:0.2514},
  {idx:9,run:"p0_yolo11n_1024_s2_e30p10m30",label:"1024 s2 p10m30",phase:"phase0",precision:0.4952,recall:0.6004,map50:0.5276,map50_95:0.2589},
  {idx:10,run:"p0_lc25_yolo11n_640",label:"learning curve 25%",phase:"phase0",precision:0.4187,recall:0.5758,map50:0.4444,map50_95:0.1984},
  {idx:11,run:"p0_lc50_yolo11n_640",label:"learning curve 50%",phase:"phase0",precision:0.4410,recall:0.5791,map50:0.4637,map50_95:0.2202},
  {idx:12,run:"p0_lc75_yolo11n_640",label:"learning curve 75%",phase:"phase0",precision:0.4683,recall:0.5906,map50:0.5033,map50_95:0.2444},
  {idx:13,run:"p0_lc100_yolo11n_640",label:"learning curve 100%",phase:"phase0",precision:0.4906,recall:0.5864,map50:0.5237,map50_95:0.2538},
  {idx:14,run:"p1a_stage1_singlecls_s1",label:"single-class detect s1",phase:"phase1",precision:0.8092,recall:0.7426,map50:0.8252,map50_95:0.3848},
  {idx:15,run:"p1a_stage1_singlecls_s2",label:"single-class detect s2",phase:"phase1",precision:0.8133,recall:0.7308,map50:0.8213,map50_95:0.3852},
  {idx:16,run:"p1a_stage2_cls_gtcrop_s1",label:"cls GT-crop s1",phase:"phase1",precision:0.6378,recall:1.0,map50:0.6378,map50_95:1.0},
  {idx:17,run:"p1a_stage2_cls_gtcrop_s2",label:"cls GT-crop s2",phase:"phase1",precision:0.6382,recall:1.0,map50:0.6382,map50_95:1.0},
  {idx:18,run:"p1b_yolov8n_640",label:"YOLOv8n",phase:"phase1",precision:0.4916,recall:0.5686,map50:0.5134,map50_95:0.2466},
  {idx:19,run:"p1b_yolov8s_640",label:"YOLOv8s",phase:"phase1",precision:0.4941,recall:0.5861,map50:0.5232,map50_95:0.2490},
  {idx:20,run:"p1b_yolov8m_640",label:"YOLOv8m",phase:"phase1",precision:0.4906,recall:0.6020,map50:0.5246,map50_95:0.2505},
  {idx:21,run:"p1b_yolo11n_640",label:"YOLO11n (baseline)",phase:"phase1",precision:0.4906,recall:0.5864,map50:0.5237,map50_95:0.2538},
  {idx:22,run:"p1bfc_yolov8n_s1",label:"YOLOv8n fc s1",phase:"phase1",precision:0.4828,recall:0.5947,map50:0.5150,map50_95:0.2467},
  {idx:23,run:"p1bfc_yolov8n_s2",label:"YOLOv8n fc s2",phase:"phase1",precision:0.4899,recall:0.5979,map50:0.5198,map50_95:0.2471},
  {idx:24,run:"p1bfc_yolov8s_s1",label:"YOLOv8s fc s1",phase:"phase1",precision:0.5025,recall:0.5910,map50:0.5256,map50_95:0.2518},
  {idx:25,run:"p1bfc_yolov8s_s2",label:"YOLOv8s fc s2",phase:"phase1",precision:0.5030,recall:0.5864,map50:0.5256,map50_95:0.2524},
  {idx:26,run:"p1bfc_yolov8m_s1",label:"YOLOv8m fc s1",phase:"phase1",precision:0.5048,recall:0.5811,map50:0.5279,map50_95:0.2528},
  {idx:27,run:"p1bfc_yolov8m_s2",label:"YOLOv8m fc s2",phase:"phase1",precision:0.5036,recall:0.5967,map50:0.5229,map50_95:0.2533},
  {idx:28,run:"p1bfc_yolov9c_s1",label:"YOLOv9c fc s1",phase:"phase1",precision:0.5046,recall:0.5926,map50:0.5286,map50_95:0.2527},
  {idx:29,run:"p1bfc_yolov9c_s2",label:"YOLOv9c fc s2",phase:"phase1",precision:0.4982,recall:0.5793,map50:0.5298,map50_95:0.2510},
  {idx:30,run:"p1bfc_yolov10n_s1",label:"YOLOv10n fc s1",phase:"phase1",precision:0.4521,recall:0.5315,map50:0.4696,map50_95:0.2303},
  {idx:31,run:"p1bfc_yolov10n_s2",label:"YOLOv10n fc s2",phase:"phase1",precision:0.4759,recall:0.5382,map50:0.4884,map50_95:0.2383},
  {idx:32,run:"p1bfc_yolov10s_s1",label:"YOLOv10s fc s1",phase:"phase1",precision:0.4709,recall:0.5545,map50:0.4892,map50_95:0.2355},
  {idx:33,run:"p1bfc_yolov10s_s2",label:"YOLOv10s fc s2",phase:"phase1",precision:0.4515,recall:0.5706,map50:0.4955,map50_95:0.2415},
  {idx:34,run:"p1bfc_yolov10m_s1",label:"YOLOv10m fc s1",phase:"phase1",precision:0.5045,recall:0.5472,map50:0.4974,map50_95:0.2424},
  {idx:35,run:"p1bfc_yolov10m_s2",label:"YOLOv10m fc s2",phase:"phase1",precision:0.5035,recall:0.5684,map50:0.5125,map50_95:0.2429},
  {idx:36,run:"p1bfc_yolo26n_s1",label:"YOLO26n fc s1",phase:"phase1",precision:0.4690,recall:0.5441,map50:0.4812,map50_95:0.2332},
  {idx:37,run:"p1bfc_yolo26n_s2",label:"YOLO26n fc s2",phase:"phase1",precision:0.4662,recall:0.5646,map50:0.4975,map50_95:0.2357},
  {idx:38,run:"p1bfc_yolo26s_s1",label:"YOLO26s fc s1",phase:"phase1",precision:0.4522,recall:0.5560,map50:0.5089,map50_95:0.2453},
  {idx:39,run:"p1bfc_yolo26s_s2",label:"YOLO26s fc s2",phase:"phase1",precision:0.4473,recall:0.5742,map50:0.4980,map50_95:0.2341},
  {idx:40,run:"p1bfc_yolo26m_s1",label:"YOLO26m fc s1",phase:"phase1",precision:0.4898,recall:0.5917,map50:0.5211,map50_95:0.2516},
  {idx:41,run:"p1bfc_yolo26m_s2",label:"YOLO26m fc s2",phase:"phase1",precision:0.4627,recall:0.5755,map50:0.5099,map50_95:0.2470},
  {idx:42,run:"p1bfc_yolo11m_s1",label:"YOLO11m fc s1",phase:"phase1",precision:0.4913,recall:0.6023,map50:0.5315,map50_95:0.2576},
  {idx:43,run:"p1bfc_yolo11m_s2",label:"YOLO11m fc s2",phase:"phase1",precision:0.4932,recall:0.5933,map50:0.5282,map50_95:0.2565},
  {idx:44,run:"p2s0a_none_s1",label:"baseline (no change)",phase:"phase2",precision:0.4913,recall:0.6023,map50:0.5315,map50_95:0.2576},
  {idx:45,run:"p2s0a_none_s2",label:"baseline s2",phase:"phase2",precision:0.4932,recall:0.5933,map50:0.5282,map50_95:0.2565},
  {idx:46,run:"p2s0a_class_weighted_s1",label:"class-weighted loss s1",phase:"phase2",precision:0.4913,recall:0.6023,map50:0.5315,map50_95:0.2576},
  {idx:47,run:"p2s0a_class_weighted_s2",label:"class-weighted loss s2",phase:"phase2",precision:0.4932,recall:0.5933,map50:0.5282,map50_95:0.2565},
  {idx:48,run:"p2s0a_focal15_s1",label:"focal loss γ=1.5 s1",phase:"phase2",precision:0.4913,recall:0.6023,map50:0.5315,map50_95:0.2576},
  {idx:49,run:"p2s0a_focal15_s2",label:"focal loss γ=1.5 s2",phase:"phase2",precision:0.4932,recall:0.5933,map50:0.5282,map50_95:0.2565},
  {idx:50,run:"p2s0b_standard_s1",label:"standard aug s1",phase:"phase2",precision:0.4913,recall:0.6023,map50:0.5315,map50_95:0.2576},
  {idx:51,run:"p2s1_lr0005_s1",label:"lr=0.005 s1",phase:"phase2",precision:0.4930,recall:0.6126,map50:0.5366,map50_95:0.2606},
  {idx:52,run:"p2s1_lr0005_s2",label:"lr=0.005 s2",phase:"phase2",precision:0.5074,recall:0.5937,map50:0.5333,map50_95:0.2548},
  {idx:53,run:"p2s1_lr002_s1",label:"lr=0.02 s1",phase:"phase2",precision:0.5121,recall:0.5748,map50:0.5301,map50_95:0.2550},
  {idx:54,run:"p2s1_lr002_s2",label:"lr=0.02 s2",phase:"phase2",precision:0.5135,recall:0.5863,map50:0.5375,map50_95:0.2624},
  {idx:55,run:"p2s2_bs8_s1",label:"batch=8 s1",phase:"phase2",precision:0.4989,recall:0.5989,map50:0.5339,map50_95:0.2560},
  {idx:56,run:"p2s2_bs8_s2",label:"batch=8 s2",phase:"phase2",precision:0.5011,recall:0.5869,map50:0.5303,map50_95:0.2588},
  {idx:57,run:"p2s2_bs16_s1",label:"batch=16 s1",phase:"phase2",precision:0.4930,recall:0.6126,map50:0.5366,map50_95:0.2606},
  {idx:58,run:"p2s2_bs16_s2",label:"batch=16 s2",phase:"phase2",precision:0.5074,recall:0.5937,map50:0.5333,map50_95:0.2548},
  {idx:59,run:"p2s3_light_s1",label:"light aug s1",phase:"phase2",precision:0.4968,recall:0.5895,map50:0.5250,map50_95:0.2497},
  {idx:60,run:"p2s3_light_s2",label:"light aug s2",phase:"phase2",precision:0.4965,recall:0.5904,map50:0.5261,map50_95:0.2527},
  {idx:61,run:"p2s3_medium_s1",label:"medium aug s1",phase:"phase2",precision:0.4930,recall:0.6126,map50:0.5366,map50_95:0.2606},
  {idx:62,run:"p2s3_medium_s2",label:"medium aug s2",phase:"phase2",precision:0.5074,recall:0.5937,map50:0.5333,map50_95:0.2548},
  {idx:63,run:"p2confirm_s3",label:"confirm seed3",phase:"phase2",precision:0.5066,recall:0.6042,map50:0.5390,map50_95:0.2594},
  {idx:64,run:"p3_final_e60_test",label:"final 60ep (TEST set)",phase:"phase3",precision:0.4763,recall:0.5538,map50:0.4677,map50_95:0.2215},
];

const METRICS = [
  { key: "map50", name: "mAP50", higher: true, color: "#3b82f6", format: v => v.toFixed(4) },
  { key: "map50_95", name: "mAP50-95", higher: true, color: "#10b981", format: v => v.toFixed(4) },
  { key: "precision", name: "Precision", higher: true, color: "#f59e0b", format: v => v.toFixed(4) },
  { key: "recall", name: "Recall", higher: true, color: "#ef4444", format: v => v.toFixed(4) },
];

const PHASE_COLORS = {
  phase0: "#94a3b8",
  phase1: "#a78bfa",
  phase2: "#38bdf8",
  phase3: "#fb923c",
};

const PHASE_LABELS = {
  phase0: "Phase 0: Baseline",
  phase1: "Phase 1: Architecture",
  phase2: "Phase 2: Hyperparams",
  phase3: "Phase 3: Final",
};

// Exclude single-class detection (idx 14,15) and classification runs (idx 16,17) since they have different metric semantics
const DETECTION_DATA = RAW_DATA.filter(d => d.idx < 14 || d.idx > 17);

function computeRunningBest(data, metricKey, higherIsBetter) {
  let best = higherIsBetter ? -Infinity : Infinity;
  const results = [];
  const keptPoints = [];

  data.forEach((d) => {
    const val = d[metricKey];
    const isNew = higherIsBetter ? val > best : val < best;
    if (isNew) {
      best = val;
      keptPoints.push({ ...d, val });
    }
    results.push({ ...d, val, isKept: isNew, runningBest: best });
  });

  return { results, keptPoints };
}

function MetricChart({ metric, data, width, height, hoveredIdx, setHoveredIdx, showLabels }) {
  const { results, keptPoints } = useMemo(
    () => computeRunningBest(data, metric.key, metric.higher),
    [data, metric]
  );

  const pad = { top: 40, right: 30, bottom: 50, left: 65 };
  const cw = width - pad.left - pad.right;
  const ch = height - pad.top - pad.bottom;

  const vals = results.map(r => r.val);
  const yMin = Math.min(...vals);
  const yMax = Math.max(...vals);
  const yPad = (yMax - yMin) * 0.08 || 0.01;

  const xScale = (idx) => pad.left + (idx / (data.length - 1)) * cw;
  const yScale = (v) => pad.top + ch - ((v - (yMin - yPad)) / (yMax - yMin + yPad * 2)) * ch;

  // Running best line
  let bestLinePath = "";
  let prevX = null, prevY = null;
  results.forEach((r) => {
    const x = xScale(r.idx);
    const y = yScale(r.runningBest);
    if (prevX === null) {
      bestLinePath += `M${x},${y}`;
    } else {
      bestLinePath += `L${x},${prevY}L${x},${y}`;
    }
    prevX = x;
    prevY = y;
  });

  // Y ticks
  const tickCount = 6;
  const yTicks = Array.from({ length: tickCount }, (_, i) => {
    const v = (yMin - yPad) + ((yMax - yMin + yPad * 2) * i) / (tickCount - 1);
    return v;
  });

  // X ticks
  const xTicks = [0, 10, 20, 30, 40, 50, 60];

  const hovered = results.find(r => r.idx === hoveredIdx);

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      {/* Background */}
      <rect x={pad.left} y={pad.top} width={cw} height={ch} fill="var(--bg-chart)" rx="2" />

      {/* Grid lines */}
      {yTicks.map((v, i) => (
        <g key={i}>
          <line x1={pad.left} x2={pad.left + cw} y1={yScale(v)} y2={yScale(v)} stroke="var(--grid)" strokeWidth="0.5" />
          <text x={pad.left - 8} y={yScale(v) + 4} textAnchor="end" fill="var(--text-secondary)" fontSize="10" fontFamily="'JetBrains Mono', monospace">
            {v.toFixed(3)}
          </text>
        </g>
      ))}

      {xTicks.filter(t => t < data.length).map((t) => (
        <g key={t}>
          <line x1={xScale(t)} x2={xScale(t)} y1={pad.top} y2={pad.top + ch} stroke="var(--grid)" strokeWidth="0.5" />
          <text x={xScale(t)} y={pad.top + ch + 16} textAnchor="middle" fill="var(--text-secondary)" fontSize="10" fontFamily="'JetBrains Mono', monospace">
            {t}
          </text>
        </g>
      ))}

      {/* Axis labels */}
      <text x={pad.left + cw / 2} y={height - 6} textAnchor="middle" fill="var(--text-secondary)" fontSize="11" fontFamily="'JetBrains Mono', monospace">
        Experiment #
      </text>
      <text transform={`rotate(-90, 14, ${pad.top + ch / 2})`} x={14} y={pad.top + ch / 2 + 4} textAnchor="middle" fill="var(--text-secondary)" fontSize="11" fontFamily="'JetBrains Mono', monospace">
        {metric.name}
      </text>

      {/* Running best line */}
      <path d={bestLinePath} fill="none" stroke={metric.color} strokeWidth="2" opacity="0.7" />

      {/* Discarded dots */}
      {results.filter(r => !r.isKept).map((r) => (
        <circle
          key={r.idx}
          cx={xScale(r.idx)}
          cy={yScale(r.val)}
          r={hoveredIdx === r.idx ? 5 : 3}
          fill="var(--dot-discarded)"
          opacity={hoveredIdx === r.idx ? 1 : 0.5}
          style={{ cursor: "pointer", transition: "r 0.15s" }}
          onMouseEnter={() => setHoveredIdx(r.idx)}
          onMouseLeave={() => setHoveredIdx(null)}
        />
      ))}

      {/* Kept dots */}
      {results.filter(r => r.isKept).map((r) => (
        <g key={r.idx}>
          <circle cx={xScale(r.idx)} cy={yScale(r.val)} r={hoveredIdx === r.idx ? 7 : 5} fill={metric.color}
            style={{ cursor: "pointer", transition: "r 0.15s" }}
            onMouseEnter={() => setHoveredIdx(r.idx)}
            onMouseLeave={() => setHoveredIdx(null)}
          />
          {showLabels && (
            <text
              x={xScale(r.idx) + 6}
              y={yScale(r.val) - 8}
              fill={metric.color}
              fontSize="8"
              fontFamily="'JetBrains Mono', monospace"
              opacity="0.85"
              transform={`rotate(-25, ${xScale(r.idx) + 6}, ${yScale(r.val) - 8})`}
            >
              {r.label}
            </text>
          )}
        </g>
      ))}

      {/* Title */}
      <text x={pad.left + 6} y={pad.top - 12} fill="var(--text-primary)" fontSize="14" fontWeight="700" fontFamily="'JetBrains Mono', monospace">
        {metric.name}
      </text>
      <text x={pad.left + cw - 4} y={pad.top - 12} fill="var(--text-secondary)" fontSize="10" textAnchor="end" fontFamily="'JetBrains Mono', monospace">
        {keptPoints.length} improvements / {data.length} runs
      </text>

      {/* Hover tooltip */}
      {hovered && (
        <g>
          <rect
            x={Math.min(xScale(hovered.idx) + 10, width - 195)}
            y={Math.max(yScale(hovered.val) - 46, pad.top)}
            width="185" height="42" rx="4"
            fill="var(--tooltip-bg)" stroke="var(--tooltip-border)" strokeWidth="1"
          />
          <text
            x={Math.min(xScale(hovered.idx) + 16, width - 189)}
            y={Math.max(yScale(hovered.val) - 30, pad.top + 16)}
            fill="var(--text-primary)" fontSize="10" fontWeight="600" fontFamily="'JetBrains Mono', monospace"
          >
            #{hovered.idx} {hovered.label}
          </text>
          <text
            x={Math.min(xScale(hovered.idx) + 16, width - 189)}
            y={Math.max(yScale(hovered.val) - 14, pad.top + 32)}
            fill={hovered.isKept ? metric.color : "var(--text-secondary)"} fontSize="10" fontFamily="'JetBrains Mono', monospace"
          >
            {metric.name}: {metric.format(hovered.val)} {hovered.isKept ? "★ NEW BEST" : ""}
          </text>
        </g>
      )}
    </svg>
  );
}

export default function App() {
  const [hoveredIdx, setHoveredIdx] = useState(null);
  const [showLabels, setShowLabels] = useState(true);
  const [excludeCls, setExcludeCls] = useState(true);
  const containerRef = useRef(null);
  const [containerWidth, setContainerWidth] = useState(900);

  useEffect(() => {
    const measure = () => {
      if (containerRef.current) setContainerWidth(containerRef.current.offsetWidth);
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  const data = excludeCls ? DETECTION_DATA : RAW_DATA;

  const chartW = Math.min(containerWidth - 16, 1100);
  const chartH = 260;

  const totalRuns = data.length;
  const bestMap50 = Math.max(...data.map(d => d.map50)).toFixed(4);
  const bestMap5095 = Math.max(...data.map(d => d.map50_95)).toFixed(4);
  const bestPrec = Math.max(...data.map(d => d.precision)).toFixed(4);
  const bestRec = Math.max(...data.map(d => d.recall)).toFixed(4);

  return (
    <div ref={containerRef} style={{
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      padding: "20px 8px",
      minHeight: "100vh",
      background: "var(--bg-main)",
      color: "var(--text-primary)"
    }}>
      <style>{`
        :root {
          --bg-main: #0f1117;
          --bg-chart: #161822;
          --text-primary: #e2e8f0;
          --text-secondary: #64748b;
          --grid: #1e293b;
          --dot-discarded: #475569;
          --tooltip-bg: #1e293bee;
          --tooltip-border: #334155;
          --accent: #10b981;
          --card-bg: #161822;
          --card-border: #1e293b;
        }
        @media (prefers-color-scheme: light) {
          :root {
            --bg-main: #f8fafc;
            --bg-chart: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --grid: #e2e8f0;
            --dot-discarded: #94a3b8;
            --tooltip-bg: #ffffffee;
            --tooltip-border: #cbd5e1;
            --card-bg: #ffffff;
            --card-border: #e2e8f0;
          }
        }
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
      `}</style>

      {/* Header */}
      <div style={{ maxWidth: chartW, margin: "0 auto 20px", textAlign: "center" }}>
        <h1 style={{ fontSize: 20, fontWeight: 700, margin: "0 0 4px", letterSpacing: "-0.5px" }}>
          Oil Palm Detection: E0 Research Progress
        </h1>
        <p style={{ fontSize: 12, color: "var(--text-secondary)", margin: "0 0 12px" }}>
          {totalRuns} experiments across 4 phases · Best mAP50: {bestMap50} · mAP50-95: {bestMap5095} · Precision: {bestPrec} · Recall: {bestRec}
        </p>

        {/* Controls */}
        <div style={{ display: "flex", gap: 16, justifyContent: "center", flexWrap: "wrap" }}>
          <label style={{ fontSize: 11, display: "flex", alignItems: "center", gap: 5, cursor: "pointer", color: "var(--text-secondary)" }}>
            <input type="checkbox" checked={showLabels} onChange={e => setShowLabels(e.target.checked)} />
            Show labels
          </label>
          <label style={{ fontSize: 11, display: "flex", alignItems: "center", gap: 5, cursor: "pointer", color: "var(--text-secondary)" }}>
            <input type="checkbox" checked={excludeCls} onChange={e => setExcludeCls(e.target.checked)} />
            Exclude single-class &amp; classification runs
          </label>
        </div>

        {/* Legend */}
        <div style={{ display: "flex", gap: 20, justifyContent: "center", marginTop: 12, flexWrap: "wrap" }}>
          <span style={{ fontSize: 11, display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--dot-discarded)", display: "inline-block" }} />
            <span style={{ color: "var(--text-secondary)" }}>Discarded</span>
          </span>
          <span style={{ fontSize: 11, display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--accent)", display: "inline-block" }} />
            <span style={{ color: "var(--text-secondary)" }}>New Best (Kept)</span>
          </span>
          <span style={{ fontSize: 11, display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 20, height: 2, background: "var(--accent)", display: "inline-block" }} />
            <span style={{ color: "var(--text-secondary)" }}>Running Best</span>
          </span>
          {Object.entries(PHASE_LABELS).map(([k, v]) => (
            <span key={k} style={{ fontSize: 11, display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ width: 8, height: 8, borderRadius: 2, background: PHASE_COLORS[k], display: "inline-block" }} />
              <span style={{ color: "var(--text-secondary)" }}>{v}</span>
            </span>
          ))}
        </div>
      </div>

      {/* Charts */}
      <div style={{ maxWidth: chartW, margin: "0 auto" }}>
        {METRICS.map((m) => (
          <div key={m.key} style={{
            marginBottom: 12,
            border: "1px solid var(--card-border)",
            borderRadius: 6,
            overflow: "hidden",
            background: "var(--card-bg)"
          }}>
            <MetricChart
              metric={m}
              data={data}
              width={chartW}
              height={chartH}
              hoveredIdx={hoveredIdx}
              setHoveredIdx={setHoveredIdx}
              showLabels={showLabels}
            />
          </div>
        ))}
      </div>

      {/* Phase summary strip */}
      <div style={{
        maxWidth: chartW, margin: "16px auto 0",
        display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 8,
      }}>
        {Object.entries(PHASE_LABELS).map(([k, v]) => {
          const phaseData = data.filter(d => d.phase === k);
          if (!phaseData.length) return null;
          const bestInPhase = Math.max(...phaseData.map(d => d.map50_95));
          return (
            <div key={k} style={{
              background: "var(--card-bg)", border: "1px solid var(--card-border)",
              borderRadius: 6, padding: "10px 14px",
              borderLeft: `3px solid ${PHASE_COLORS[k]}`
            }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: PHASE_COLORS[k] }}>{v}</div>
              <div style={{ fontSize: 10, color: "var(--text-secondary)", marginTop: 2 }}>
                {phaseData.length} runs · best mAP50-95: {bestInPhase.toFixed(4)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

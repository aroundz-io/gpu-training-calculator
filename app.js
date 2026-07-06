/* ==========================================================================
   TrainLab — 온라인 AI 학습 서비스 (데모)
   - 데이터: GPU 스펙, 모델 프리셋
   - 엔진: FLOPs/시간/비용 계산
   - 서비스: 뷰 라우팅, 업로드 위저드, 모의 결제, 학습 시뮬레이션 대시보드
   - 계산기: 기존 상세 비교 계산기 뷰
   ========================================================================== */

// ==================== 데이터 ====================
// bf16/fp8: dense TFLOPS (sparsity 제외), price: $/GPU-hour (온디맨드 대략치)
// V100·T4는 BF16 미지원이라 FP16 텐서코어 수치를 사용
const GPUS = [
  { id:"gb300",  name:"GB300 NVL72",     cat:"데이터센터 · Blackwell", arch:"Blackwell Ultra", mem:288, bf16:2250, fp8:4500, price:9.0 },
  { id:"gb200",  name:"GB200 NVL72",     cat:"데이터센터 · Blackwell", arch:"Blackwell", mem:186, bf16:2500, fp8:5000, price:7.0 },
  { id:"b200",   name:"B200 SXM",        cat:"데이터센터 · Blackwell", arch:"Blackwell", mem:192, bf16:2250, fp8:4500, price:5.5 },
  { id:"b100",   name:"B100",            cat:"데이터센터 · Blackwell", arch:"Blackwell", mem:192, bf16:1750, fp8:3500, price:4.5 },
  { id:"h200",   name:"H200 SXM",        cat:"데이터센터 · Hopper", arch:"Hopper", mem:141, bf16:989,  fp8:1979, price:3.2 },
  { id:"h100",   name:"H100 SXM",        cat:"데이터센터 · Hopper", arch:"Hopper", mem:80,  bf16:989,  fp8:1979, price:2.5 },
  { id:"h100p",  name:"H100 PCIe",       cat:"데이터센터 · Hopper", arch:"Hopper", mem:80,  bf16:756,  fp8:1513, price:2.0 },
  { id:"h20",    name:"H20",             cat:"데이터센터 · Hopper", arch:"Hopper", mem:96,  bf16:148,  fp8:296,  price:1.2 },
  { id:"l40s",   name:"L40S",            cat:"데이터센터 · Ada (L-시리즈)", arch:"Ada", mem:48, bf16:181, fp8:362, price:1.0 },
  { id:"l40",    name:"L40",             cat:"데이터센터 · Ada (L-시리즈)", arch:"Ada", mem:48, bf16:91,  fp8:181, price:0.8 },
  { id:"l4",     name:"L4",              cat:"데이터센터 · Ada (L-시리즈)", arch:"Ada", mem:24, bf16:61,  fp8:121, price:0.4 },
  { id:"a100",   name:"A100 80GB",       cat:"데이터센터 · Ampere/Volta/Turing", arch:"Ampere", mem:80, bf16:312, fp8:null, price:1.4 },
  { id:"a10040", name:"A100 40GB",       cat:"데이터센터 · Ampere/Volta/Turing", arch:"Ampere", mem:40, bf16:312, fp8:null, price:1.1 },
  { id:"a40",    name:"A40",             cat:"데이터센터 · Ampere/Volta/Turing", arch:"Ampere", mem:48, bf16:150, fp8:null, price:0.6 },
  { id:"a30",    name:"A30",             cat:"데이터센터 · Ampere/Volta/Turing", arch:"Ampere", mem:24, bf16:165, fp8:null, price:0.7 },
  { id:"a10",    name:"A10",             cat:"데이터센터 · Ampere/Volta/Turing", arch:"Ampere", mem:24, bf16:125, fp8:null, price:0.4 },
  { id:"v100",   name:"V100 SXM2 32GB",  cat:"데이터센터 · Ampere/Volta/Turing", arch:"Volta · FP16", mem:32, bf16:125, fp8:null, price:0.5 },
  { id:"t4",     name:"T4",              cat:"데이터센터 · Ampere/Volta/Turing", arch:"Turing · FP16", mem:16, bf16:65, fp8:null, price:0.3 },
  { id:"dgxspark",   name:"DGX Spark (GB10)",       cat:"워크스테이션 · 개인용", arch:"Blackwell · 통합메모리", mem:128, bf16:100, fp8:208, price:0.2 },
  { id:"rtxpro6000", name:"RTX PRO 6000 Blackwell", cat:"워크스테이션 · 개인용", arch:"Blackwell", mem:96, bf16:250, fp8:500, price:1.8 },
  { id:"rtx6000ada", name:"RTX 6000 Ada",           cat:"워크스테이션 · 개인용", arch:"Ada",       mem:48, bf16:182, fp8:364, price:1.0 },
  { id:"rtxa6000",   name:"RTX A6000",              cat:"워크스테이션 · 개인용", arch:"Ampere",    mem:48, bf16:155, fp8:null, price:0.8 },
  { id:"rtx5090",  name:"RTX 5090",    cat:"소비자용 GeForce", arch:"Blackwell", mem:32, bf16:210, fp8:419, price:0.9 },
  { id:"rtx5080",  name:"RTX 5080",    cat:"소비자용 GeForce", arch:"Blackwell", mem:16, bf16:112, fp8:225, price:0.45 },
  { id:"rtx4090",  name:"RTX 4090",    cat:"소비자용 GeForce", arch:"Ada",       mem:24, bf16:165, fp8:330, price:0.5 },
  { id:"rtx4080",  name:"RTX 4080",    cat:"소비자용 GeForce", arch:"Ada",       mem:16, bf16:98,  fp8:195, price:0.35 },
  { id:"rtx3090ti",name:"RTX 3090 Ti", cat:"소비자용 GeForce", arch:"Ampere",    mem:24, bf16:80,  fp8:null, price:0.3 },
  { id:"rtx3090",  name:"RTX 3090",    cat:"소비자용 GeForce", arch:"Ampere",    mem:24, bf16:71,  fp8:null, price:0.25 },
];

// 비전/확산/기상 모델 프리셋: 샘플(스텝)당 순전파 GFLOPs
const VMODELS = {
  vision: [
    { id:"resnet50",  name:"ResNet-50 (224px)",   gflops:4.1 },
    { id:"convnextL", name:"ConvNeXt-L (224px)",  gflops:34.4 },
    { id:"vitb",      name:"ViT-B/16 (224px)",    gflops:17.6 },
    { id:"vitl",      name:"ViT-L/16 (224px)",    gflops:61.6 },
  ],
  diffusion: [
    { id:"sd15",  name:"SD 1.5 U-Net (512px, 스텝당)",   gflops:350 },
    { id:"sdxl",  name:"SDXL U-Net (1024px, 스텝당)",    gflops:1500 },
    { id:"flux",  name:"FLUX.1 급 DiT (1024px, 스텝당)", gflops:4000 },
  ],
  // NVIDIA Earth-2 — 공개된 학습 비용에 맞춰 보정한 추정치
  earth2: [
    { id:"fcn3",     name:"FourCastNet 3 (FCN3, 0.25° 전지구 예보)",  gflops:6.0e6,  bytes:3.0e8, defSamples:[5.4,"1e4"], defEpochs:60 },
    { id:"corrdiff", name:"CorrDiff (km-scale 다운스케일링, CONUS)",  gflops:1.33e6, bytes:2.5e7, defSamples:[3.5,"1e4"], defEpochs:10 },
  ],
};

const EARTH2_PRESETS = [ [1460,"ERA5 1년 (6h 간격)"], [14600,"ERA5 10년"], [54000,"ERA5 37년 (FCN3)"], [35000,"HRRR 4년 (CorrDiff)"] ];

// 유명 오픈소스 LLM 프리셋 — b: 전체 파라미터(B), active: MoE 활성 파라미터(B), tokens: 사전학습 토큰 수
const LLM_PRESETS = [
  { g:"NVIDIA",   id:"nemotron340", name:"Nemotron-4 340B",            b:340,  tokens:9e12 },
  { g:"NVIDIA",   id:"nemotron70",  name:"Llama-3.1-Nemotron 70B",     b:70,   tokens:15e12 },
  { g:"NVIDIA",   id:"nemonano9",   name:"Nemotron Nano 9B",           b:9,    tokens:20e12 },
  { g:"Meta",     id:"llama8",      name:"Llama 3.1 8B",               b:8,    tokens:15e12 },
  { g:"Meta",     id:"llama70",     name:"Llama 3.1 70B",              b:70,   tokens:15e12 },
  { g:"Meta",     id:"llama405",    name:"Llama 3.1 405B",             b:405,  tokens:15.6e12 },
  { g:"DeepSeek", id:"dsv3",        name:"DeepSeek-V3 / R1 671B (MoE)",b:671,  active:37,   tokens:14.8e12 },
  { g:"Alibaba",  id:"qwen7",       name:"Qwen2.5 7B",                 b:7.6,  tokens:18e12 },
  { g:"Alibaba",  id:"qwen72",      name:"Qwen2.5 72B",                b:72.7, tokens:18e12 },
  { g:"Alibaba",  id:"qwen235",     name:"Qwen3 235B (MoE)",           b:235,  active:22,   tokens:36e12 },
  { g:"Mistral",  id:"mistral7",    name:"Mistral 7B",                 b:7.2,  tokens:8e12 },
  { g:"Mistral",  id:"mixtral",     name:"Mixtral 8×7B (MoE)",         b:46.7, active:12.9, tokens:8e12 },
  { g:"Google",   id:"gemma27",     name:"Gemma 3 27B",                b:27,   tokens:14e12 },
  { g:"Microsoft",id:"phi4",        name:"Phi-4 14B",                  b:14,   tokens:9.8e12 },
  { g:"OpenAI",   id:"gptoss120",   name:"gpt-oss 120B (MoE)",         b:117,  active:5.1,  tokens:null },
];

const BYTES_PER_TOKEN = 4;      // 평문 텍스트 기준 1토큰 ≈ 4바이트
const BYTES_PER_IMAGE = 5e5;    // 이미지 1장 ≈ 500KB (JPEG 기준)
const FX = 1380;                // 서비스 뷰 기본 환율 (계산기 뷰는 별도 입력)

const TASK_LABELS = {
  pretrain:"LLM 사전학습", finetune:"LLM 전체 파인튜닝", lora:"LLM LoRA/QLoRA 파인튜닝",
  vision:"이미지 분류", diffusion:"확산 모델 학습", earth2:"기상 · 지구 시뮬레이션",
};

const $ = id => document.getElementById(id);

// ==================== SVG 심볼 아이콘 ====================
const ICONS = {
  upload:     '<path d="M12 15V4"/><path d="M8 8l4-4 4 4"/><path d="M4 16v2.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V16"/>',
  chip:       '<rect x="7" y="7" width="10" height="10" rx="2"/><path d="M9.5 3.5v3.5M14.5 3.5v3.5M9.5 17v3.5M14.5 17v3.5M3.5 9.5H7M3.5 14.5H7M17 9.5h3.5M17 14.5h3.5"/>',
  zap:        '<path d="M13 2.5 4.5 13.5H11L10 21.5l8.5-11H12l1-8z"/>',
  creditcard: '<rect x="3" y="5.5" width="18" height="13" rx="2.5"/><path d="M3 10h18M7 15h4"/>',
  calculator: '<rect x="5" y="3" width="14" height="18" rx="2.5"/><path d="M8.5 7.5h7M8.5 12h.01M12 12h.01M15.5 12h.01M8.5 15.5h.01M12 15.5h.01M15.5 15.5h.01"/>',
  activity:   '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
  shield:     '<path d="M12 3l7.5 3v5.5c0 4.5-3.2 7.8-7.5 9.5-4.3-1.7-7.5-5-7.5-9.5V6L12 3z"/><path d="M9 11.5l2.2 2.2L15.5 9.5"/>',
  lock:       '<rect x="5.5" y="10.5" width="13" height="9.5" rx="2"/><path d="M8.5 10.5V7.5a3.5 3.5 0 0 1 7 0v3"/><path d="M12 14.5v2"/>',
  trash:      '<path d="M4.5 6.5h15M9.5 6.5V4.8a1.3 1.3 0 0 1 1.3-1.3h2.4a1.3 1.3 0 0 1 1.3 1.3v1.7M6.5 6.5l.8 12.2a1.6 1.6 0 0 0 1.6 1.5h6.2a1.6 1.6 0 0 0 1.6-1.5l.8-12.2"/><path d="M10 10.5v6M14 10.5v6"/>',
  box:        '<path d="M12 2.8l8 4.4v9.6l-8 4.4-8-4.4V7.2l8-4.4z"/><path d="M4.3 7.4 12 11.7l7.7-4.3M12 11.7V21"/>',
  checkcircle:'<circle cx="12" cy="12" r="9"/><path d="M8.5 12.3l2.4 2.4 4.8-5"/>',
  file:       '<path d="M13.5 3H7a1.5 1.5 0 0 0-1.5 1.5v15A1.5 1.5 0 0 0 7 21h10a1.5 1.5 0 0 0 1.5-1.5V8L13.5 3z"/><path d="M13.5 3v5h5"/>',
  image:      '<rect x="3.5" y="4.5" width="17" height="15" rx="2"/><circle cx="9" cy="10" r="1.6"/><path d="M20.5 15.5l-4.5-4.5-9 8.5"/>',
  folder:     '<path d="M3.5 7A1.5 1.5 0 0 1 5 5.5h4.5l2 2.5H19A1.5 1.5 0 0 1 20.5 9.5v8A1.5 1.5 0 0 1 19 19H5a1.5 1.5 0 0 1-1.5-1.5V7z"/>',
  coin:       '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5v9M9.3 9.8c.5-.9 1.5-1.4 2.7-1.4 1.7 0 3 .9 3 2.1 0 2.7-5.8 1.3-5.8 4 0 1.2 1.3 2.1 3 2.1 1.2 0 2.2-.5 2.7-1.4"/>',
  database:   '<ellipse cx="12" cy="5.5" rx="7.5" ry="2.8"/><path d="M4.5 5.5v13c0 1.5 3.4 2.8 7.5 2.8s7.5-1.3 7.5-2.8v-13"/><path d="M4.5 12c0 1.5 3.4 2.8 7.5 2.8s7.5-1.3 7.5-2.8"/>',
};
function icon(name, cls = "") {
  return `<svg class="icon ${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ""}</svg>`;
}
function initIcons() {
  document.querySelectorAll("[data-ic]").forEach(el => {
    el.innerHTML = icon(el.dataset.ic, el.dataset.icCls || "");
  });
}

// ==================== 포맷 유틸 ====================
function fmtNum(x, d=1) { return x.toLocaleString("ko-KR", { maximumFractionDigits:d }); }
function fmtFlops(f) {
  if (f >= 1e24) return [fmtNum(f/1e24,2), "YFLOPs(×10²⁴)"];
  if (f >= 1e21) return [fmtNum(f/1e21,2), "ZFLOPs(×10²¹)"];
  if (f >= 1e18) return [fmtNum(f/1e18,2), "EFLOPs(×10¹⁸)"];
  return [fmtNum(f/1e15,2), "PFLOPs(×10¹⁵)"];
}
function fmtHours(h) {
  if (!isFinite(h)) return "–";
  if (h < 1) return fmtNum(h*60,0) + "분";
  if (h < 48) return fmtNum(h,1) + "시간";
  if (h < 24*365) {
    const d = Math.floor(h/24), rem = h - d*24;
    return d + "일 " + fmtNum(rem,0) + "시간";
  }
  return fmtNum(h/24/365,1) + "년";
}
function fmtUsd(x) {
  if (!isFinite(x)) return "–";
  if (x >= 1e6) return "$" + fmtNum(x/1e6,2) + "M";
  if (x >= 1000) return "$" + fmtNum(x,0);
  return "$" + fmtNum(x,2);
}
function fmtKrw(x) {
  if (x >= 1e8) return fmtNum(x/1e8,1) + "억 원";
  if (x >= 1e4) return fmtNum(x/1e4,0) + "만 원";
  return fmtNum(x,0) + "원";
}
function fmtBytes(b) {
  if (b >= 1e15) return fmtNum(b/1e15,1) + " PB";
  if (b >= 1e12) return fmtNum(b/1e12,1) + " TB";
  if (b >= 1e9)  return fmtNum(b/1e9,1) + " GB";
  if (b >= 1e6)  return fmtNum(b/1e6,1) + " MB";
  return fmtNum(b/1e3,0) + " KB";
}
function fmtTokens(t) {
  if (t >= 1e12) return fmtNum(t/1e12,1) + "T 토큰";
  if (t >= 1e9)  return fmtNum(t/1e9,1) + "B 토큰";
  return fmtNum(t/1e6,1) + "M 토큰";
}
function fmtGflops(g) {
  if (g >= 1e6) return fmtNum(g/1e6,1) + " PFLOPs";
  if (g >= 1e3) return fmtNum(g/1e3,1) + " TFLOPs";
  return fmtNum(g,1) + " GFLOPs";
}
function isLLMTask(t) { return ["pretrain","finetune","lora"].includes(t); }

// ==================== 공통 계산 엔진 ====================
// cfg: {task, paramsB, activeB, tokens, samples, vmodelId, epochs, precision, gpuCount, mfu}
function engineFlops(c) {
  const e = Math.max(0.1, c.epochs || 1);
  if (isLLMTask(c.task)) {
    const p = (c.activeB || c.paramsB) * 1e9;   // MoE는 활성 파라미터 기준
    return (c.task === "lora" ? 4 : 6) * p * (c.tokens || 0) * e;
  }
  const list = VMODELS[c.task] || VMODELS.vision;
  const vm = list.find(m => m.id === c.vmodelId) || list[0];
  return vm.gflops * 1e9 * 3 * (c.samples || 0) * e;
}
function engineRun(gpu, c) {
  const t = (c.precision === "fp8" && gpu.fp8) ? gpu.fp8 : gpu.bf16;
  const fellBack = c.precision === "fp8" && !gpu.fp8;
  // DGX Spark는 메모리 대역폭 한계로 실효 효율 상한을 낮게 적용
  const mfu = gpu.id === "dgxspark" ? Math.min(c.mfu, 0.2) : c.mfu;
  const flops = engineFlops(c);
  const hours = flops / (c.gpuCount * t * 1e12 * mfu) / 3600;
  const price = c.priceOf ? c.priceOf(gpu) : gpu.price;
  return { flops, hours, cost: hours * c.gpuCount * price, price, tflops: t, mfu, fellBack };
}
// LLM 필요 메모리(GB): full ≈ 18 bytes/param, LoRA ≈ 4 bytes/param (ZeRO-3 분산 가정)
function engineMem(c) {
  if (!isLLMTask(c.task)) return null;
  const bytesPerParam = c.task === "lora" ? 4 : 18;
  return c.paramsB * bytesPerParam;   // GB (B 파라미터 × bytes/param)
}

// ==================== 뷰 라우팅 ====================
function showView(v) {
  document.querySelectorAll(".view").forEach(el => el.classList.toggle("active", el.id === "view-" + v));
  document.querySelectorAll("[data-view]").forEach(a => {
    if (a.classList.contains("nav") || a.tagName === "A") a.classList.toggle("on", a.dataset.view === v);
  });
  location.hash = v;
  if (v === "dashboard") { renderJobs(); startTick(); } else { stopTick(); }
  window.scrollTo(0, 0);
}
document.querySelectorAll("[data-view]").forEach(el => {
  el.addEventListener("click", () => showView(el.dataset.view));
});

// ==================== 홈: 인기 GPU 시세 ====================
(function buildHomeTable() {
  const pick = ["b200","h200","h100","a100","l40s","rtx4090"];
  document.querySelector("#homePriceTable tbody").innerHTML = pick.map(id => {
    const g = GPUS.find(x => x.id === id);
    return `<tr>
      <td class="gname">${g.name}<small>${g.arch}</small></td>
      <td>${g.mem}GB</td>
      <td>${fmtNum(g.bf16,0)} TFLOPS</td>
      <td><b>$${g.price}</b> <small style="color:var(--muted)">(약 ${fmtKrw(g.price*FX)})</small></td>
    </tr>`;
  }).join("");
})();

/* ==========================================================================
   위저드 (새 학습)
   ========================================================================== */
const order = {
  step: 1,
  files: [], textBytes: 0, imgCount: 0, imgBytes: 0, otherBytes: 0,
  dataType: null,          // "text" | "image"
  tokens: 0, samples: 0,
  task: "finetune", modelId: "llama8", modelLabel: "Llama 3.1 8B",
  paramsB: 8, activeB: null, vmodelId: null,
  epochs: 1, precision: "bf16",
  gpuId: "h100", gpuCount: 8,
  storageUse: false, storageMonths: 3,
};

// ---------- 데이터 보관 / 스토리지 서비스 ----------
const STORAGE_KRW_PER_GB_MONTH = 40;   // 스토리지 보관료: GB당 월 40원
const STORAGE_MIN_KRW_MONTH = 500;     // 월 최소 보관료
const RETENTION_DAYS = 15;             // 임시 보관: 학습 완료 후 15일 뒤 영구 삭제

function orderDataBytes() { return order.dataType === "text" ? order.textBytes : order.imgBytes; }
function storageFeeKrw() {
  if (!order.storageUse) return 0;
  const gb = orderDataBytes() / 1e9;
  return Math.max(STORAGE_MIN_KRW_MONTH, Math.ceil(gb * STORAGE_KRW_PER_GB_MONTH)) * order.storageMonths;
}
function syncStorageUI() {
  order.storageUse = $("storageOpt").value === "paid";
  order.storageMonths = parseInt($("storageMonths").value);
  $("storageMonthsField").style.display = order.storageUse ? "block" : "none";
  if (order.storageUse) {
    const gb = orderDataBytes() / 1e9;
    const monthly = Math.max(STORAGE_MIN_KRW_MONTH, Math.ceil(gb * STORAGE_KRW_PER_GB_MONTH));
    $("storageFeeHint").textContent = "GB당 월 " + STORAGE_KRW_PER_GB_MONTH + "원 (최소 월 " + STORAGE_MIN_KRW_MONTH + "원) → 월 " + fmtNum(monthly,0) + "원";
  }
}
$("storageOpt").addEventListener("change", syncStorageUI);
$("storageMonths").addEventListener("change", syncStorageUI);

window.openTerms = function() { $("termsModal").classList.add("show"); };
window.closeTerms = function() { $("termsModal").classList.remove("show"); };

const TEXT_EXT = ["txt","json","jsonl","csv","tsv","md","xml","html","parquet"];
const IMG_EXT  = ["jpg","jpeg","png","webp","gif","bmp","tif","tiff"];

// ---------- STEP 1: 업로드 ----------
const dz = $("dropzone");
dz.addEventListener("click", () => $("fileInput").click());
dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("drag"); });
dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
dz.addEventListener("drop", e => {
  e.preventDefault(); dz.classList.remove("drag");
  addFiles([...e.dataTransfer.files]);
});
$("fileInput").addEventListener("change", e => addFiles([...e.target.files]));

function addFiles(list) {
  list.forEach(f => order.files.push({ name: f.name, size: f.size }));
  analyzeFiles();
}
function analyzeFiles() {
  order.textBytes = 0; order.imgCount = 0; order.imgBytes = 0; order.otherBytes = 0;
  order.files.forEach(f => {
    const ext = (f.name.split(".").pop() || "").toLowerCase();
    if (TEXT_EXT.includes(ext)) order.textBytes += f.size;
    else if (IMG_EXT.includes(ext)) { order.imgCount++; order.imgBytes += f.size; }
    else order.otherBytes += f.size;
  });
  // 유형 자동 판별: 더 큰 쪽 기준
  order.dataType = order.imgBytes > order.textBytes ? "image" : (order.textBytes > 0 ? "text" : null);
  order.tokens = order.textBytes / BYTES_PER_TOKEN;
  order.samples = order.imgCount;
  renderFileList();
  syncStep1();
}
function renderFileList() {
  const max = 8;
  const rows = order.files.slice(0, max).map(f =>
    `<div class="frow"><span class="fn">${icon("file","sm")} ${f.name}</span><small>${fmtBytes(f.size)}</small></div>`).join("");
  const more = order.files.length > max ? `<div class="frow"><span class="fn">… 외 ${order.files.length - max}개</span><small></small></div>` : "";
  const clear = order.files.length ? `<div style="text-align:right;margin-top:6px"><button class="btn ghost sm" onclick="clearFiles()">전체 삭제</button></div>` : "";
  $("fileList").innerHTML = rows + more + clear;
}
window.clearFiles = function() { order.files = []; analyzeFiles(); };

$("manualType").addEventListener("change", syncStep1);
$("manualAmount").addEventListener("input", syncStep1);

function syncStep1() {
  const manual = $("manualType").value;
  $("manualHint").textContent = manual === "text" ? "GB 단위" : manual === "image" ? "장 단위" : "";
  if (manual) {
    const amt = parseFloat($("manualAmount").value) || 0;
    order.dataType = manual;
    if (manual === "text") { order.tokens = amt * 1e9 / BYTES_PER_TOKEN; order.textBytes = amt * 1e9; }
    else { order.samples = amt; order.imgCount = amt; order.imgBytes = amt * BYTES_PER_IMAGE; }
  }
  const has = order.dataType && ((order.dataType === "text" && order.tokens > 0) || (order.dataType === "image" && order.samples > 0));
  $("dataSum").style.display = has ? "grid" : "none";
  syncStorageUI();
  if (has) {
    const isText = order.dataType === "text";
    $("dsType").innerHTML = isText
      ? icon("file","sm") + " 텍스트"
      : icon("image","sm") + " 이미지";
    $("dsTypeSub").textContent = manual ? "직접 입력 기준" : "업로드 파일 자동 판별";
    $("dsBytes").textContent = fmtBytes(isText ? order.textBytes : order.imgBytes);
    $("dsFiles").textContent = order.files.length ? order.files.length + "개 파일" : "";
    $("dsUnits").textContent = isText ? fmtTokens(order.tokens) : fmtNum(order.samples,0) + "장";
    $("dsUnitsSub").textContent = isText ? "1토큰 ≈ 4바이트 환산" : "학습 샘플 수";
  }
  $("toStep2").disabled = !has;
}
$("toStep2").addEventListener("click", () => goStep(2));

// ---------- STEP 2: 모델 설정 ----------
function buildStep2() {
  const isText = order.dataType === "text";
  const tasks = isText
    ? [["finetune","LLM 전체 파인튜닝 (Full FT)"],["lora","LLM LoRA / QLoRA 파인튜닝"],["pretrain","LLM 사전학습 (Pre-training)"]]
    : [["vision","이미지 분류 (CNN / ViT)"],["diffusion","확산 모델 학습 (SD 계열)"]];
  if (!tasks.some(t => t[0] === order.task)) order.task = tasks[0][0];
  $("wTask").innerHTML = tasks.map(([v,l]) => `<option value="${v}" ${v===order.task?"selected":""}>${l}</option>`).join("");
  buildWModel();
  echoData();
}
function buildWModel() {
  if (isLLMTask(order.task)) {
    const groups = [...new Set(LLM_PRESETS.map(m => m.g))];
    $("wModel").innerHTML = groups.map(g => `<optgroup label="${g}">` +
      LLM_PRESETS.filter(m => m.g === g).map(m => `<option value="${m.id}">${m.name}</option>`).join("") +
      "</optgroup>").join("") + '<option value="custom">직접 입력</option>';
    if (![...$("wModel").options].some(o => o.value === order.modelId)) order.modelId = "llama8";
    $("wModel").value = order.modelId;
    applyWModel();
    $("wParamsField").style.display = "block";
  } else {
    const list = VMODELS[order.task];
    $("wModel").innerHTML = list.map(m => `<option value="${m.id}">${m.name}</option>`).join("");
    order.vmodelId = list[0].id;
    order.modelLabel = list[0].name;
    $("wModel").value = order.vmodelId;
    $("wModelInfo").textContent = "순전파 ≈ " + fmtGflops(list[0].gflops) + "/샘플";
    $("wParamsField").style.display = "none";
  }
}
function applyWModel() {
  const m = LLM_PRESETS.find(x => x.id === $("wModel").value);
  if (m) {
    order.modelId = m.id; order.modelLabel = m.name;
    order.paramsB = m.b; order.activeB = m.active || null;
    $("wParams").value = m.b;
    $("wModelInfo").textContent = m.active ? "MoE — 연산은 활성 " + m.active + "B 기준" : "";
  } else {
    order.modelId = "custom"; order.modelLabel = "커스텀 " + order.paramsB + "B";
    order.activeB = null;
    $("wModelInfo").textContent = "";
  }
}
$("wTask").addEventListener("change", () => { order.task = $("wTask").value; buildWModel(); echoData(); });
$("wModel").addEventListener("change", () => {
  if (isLLMTask(order.task)) applyWModel();
  else {
    order.vmodelId = $("wModel").value;
    const vm = VMODELS[order.task].find(m => m.id === order.vmodelId);
    order.modelLabel = vm.name;
    $("wModelInfo").textContent = "순전파 ≈ " + fmtGflops(vm.gflops) + "/샘플";
  }
});
$("wParams").addEventListener("input", () => {
  order.paramsB = parseFloat($("wParams").value) || 0;
  order.modelId = "custom"; order.activeB = null;
  order.modelLabel = "커스텀 " + order.paramsB + "B";
  $("wModel").value = "custom"; $("wModelInfo").textContent = "";
});
$("wEpochs").addEventListener("input", () => { order.epochs = parseFloat($("wEpochs").value) || 1; echoData(); });
$("wPrecision").addEventListener("change", () => { order.precision = $("wPrecision").value; });

function echoData() {
  const isText = order.dataType === "text";
  $("wDataEcho").innerHTML = "<b>업로드 데이터</b> — " + (isText
    ? fmtTokens(order.tokens) + " (" + fmtBytes(order.textBytes) + " 텍스트)"
    : fmtNum(order.samples,0) + "장 (" + fmtBytes(order.imgBytes) + ")") +
    " × " + order.epochs + " 에포크로 학습합니다.";
}

// ---------- STEP 3: GPU·견적 ----------
function wizardCfg() {
  const n = order.gpuCount;
  const mfu = n >= 16 ? 0.32 : n >= 8 ? 0.35 : n > 1 ? 0.40 : 0.45;
  return {
    task: order.task, paramsB: order.paramsB, activeB: order.activeB,
    tokens: order.tokens, samples: order.samples, vmodelId: order.vmodelId,
    epochs: order.epochs, precision: order.precision, gpuCount: n, mfu,
  };
}
function buildStep3() {
  order.gpuCount = parseInt($("wGpuCount").value);
  const cfg = wizardCfg();
  const runs = GPUS.map(g => ({ g, r: engineRun(g, cfg) }));
  const cheapest = runs.reduce((a,b) => b.r.cost < a.r.cost ? b : a).g.id;
  const fastest  = runs.reduce((a,b) => b.r.hours < a.r.hours ? b : a).g.id;
  let html = "", lastCat = null;
  runs.forEach(({g, r}) => {
    if (g.cat !== lastCat) { html += `<div class="gpu-cat">${g.cat}</div>`; lastCat = g.cat; }
    const badge = g.id === cheapest ? `<span class="badge">${icon("coin","xs")} 최저 비용</span>`
                : g.id === fastest  ? `<span class="badge b2">${icon("zap","xs")} 최단 시간</span>` : "";
    html += `<div class="gpu-item ${g.id === order.gpuId ? "on" : ""}" data-wgpu="${g.id}">${badge}
      <div class="nm">${g.name}</div>
      <div class="sp">${g.mem}GB · $${g.price}/시간<br>${fmtHours(r.hours)} · ${fmtUsd(r.cost)}</div>
    </div>`;
  });
  $("wGpuGrid").innerHTML = html;
  document.querySelectorAll("[data-wgpu]").forEach(el => {
    el.addEventListener("click", () => { order.gpuId = el.dataset.wgpu; buildStep3(); });
  });
  renderQuote();
}
$("wGpuCount").addEventListener("change", buildStep3);

function quoteRows() {
  const cfg = wizardCfg();
  const g = GPUS.find(x => x.id === order.gpuId);
  const r = engineRun(g, cfg);
  const [fv, fu] = fmtFlops(r.flops);
  const isText = order.dataType === "text";
  const krw = Math.round(r.cost * FX);
  const storage = storageFeeKrw();
  const vat = Math.round((krw + storage) * 0.1);
  const storageRow = order.storageUse
    ? `<div class="qrow"><small>스토리지 보관료</small><span>${fmtBytes(orderDataBytes())} × ${order.storageMonths}개월 = ${fmtNum(storage,0)}원</span></div>
       <div class="qrow"><small>데이터 보관</small><span style="color:var(--accent)">${icon("lock","xs")} 이용 기간(${order.storageMonths}개월) 동안 안전 보관</span></div>`
    : `<div class="qrow"><small>데이터 보관</small><span style="color:var(--warn)">임시 보관 — 학습 완료 ${RETENTION_DAYS}일 후 영구 삭제</span></div>`;
  return { g, r, html: `
    <div class="qrow"><small>데이터</small><span>${isText ? fmtTokens(order.tokens) + " (" + fmtBytes(order.textBytes) + ")" : fmtNum(order.samples,0) + "장 (" + fmtBytes(order.imgBytes) + ")"}</span></div>
    <div class="qrow"><small>작업</small><span>${TASK_LABELS[order.task]} · ${order.modelLabel} · ${order.epochs}에포크</span></div>
    <div class="qrow"><small>총 연산량</small><span>${fv} ${fu}</span></div>
    <div class="qrow"><small>GPU 구성</small><span>${g.name} × ${order.gpuCount} (${order.precision.toUpperCase()}, MFU ${Math.round(r.mfu*100)}%)</span></div>
    <div class="qrow"><small>예상 학습 시간</small><span><b>${fmtHours(r.hours)}</b></span></div>
    <div class="qrow"><small>시간당 요금</small><span>$${fmtNum(r.price * order.gpuCount, 2)} (${g.name} $${r.price} × ${order.gpuCount})</span></div>
    <div class="qrow"><small>학습 비용</small><span>${fmtUsd(r.cost)} ≈ ${fmtNum(krw, 0)}원</span></div>
    ${storageRow}
    <div class="qrow"><small>부가세 (10%)</small><span>${fmtNum(vat, 0)}원</span></div>
    <div class="qrow total"><span>결제 예정 금액</span><span class="amt">${fmtNum(krw + storage + vat, 0)}원</span></div>`,
    krwTotal: krw + storage + vat, storageKrw: storage };
}
function renderQuote() {
  const q = quoteRows();
  $("wQuote").innerHTML = q.html;
  // 메모리 체크
  const needGB = engineMem(wizardCfg());
  const mw = $("wMemWarn");
  if (needGB) {
    const haveGB = q.g.mem * order.gpuCount;
    const ok = haveGB >= needGB * 1.2;
    mw.style.display = "block";
    mw.className = "memwarn " + (ok ? "ok" : "bad");
    mw.innerHTML = (ok ? "✅ " : "⚠️ ") + "예상 필요 메모리 약 <b>" + fmtNum(needGB,0) + " GB</b> / 보유 " +
      fmtNum(haveGB,0) + " GB (" + q.g.name + " " + q.g.mem + "GB × " + order.gpuCount + ")" +
      (ok ? "" : " — 메모리가 부족할 수 있습니다. GPU 수를 늘리거나 LoRA/QLoRA를 고려하세요.");
  } else mw.style.display = "none";
}

// ---------- STEP 4: 결제 ----------
function buildStep4() {
  $("paySummary").innerHTML = quoteRows().html;
  if (!$("jobName").value) {
    $("jobName").value = order.modelLabel.replace(/[ /()×]+/g, "-").toLowerCase() + "-" +
      (order.task === "lora" ? "lora" : order.task);
  }
  syncPayBtn();
}
$("payCard").addEventListener("input", e => {
  e.target.value = e.target.value.replace(/\D/g, "").slice(0,16).replace(/(\d{4})(?=\d)/g, "$1 ");
  syncPayBtn();
});
$("payExp").addEventListener("input", e => {
  let v = e.target.value.replace(/\D/g, "").slice(0,4);
  e.target.value = v.length > 2 ? v.slice(0,2) + "/" + v.slice(2) : v;
  syncPayBtn();
});
$("payCvc").addEventListener("input", e => { e.target.value = e.target.value.replace(/\D/g, "").slice(0,3); syncPayBtn(); });
$("payAgree").addEventListener("change", syncPayBtn);
$("payAgreeData").addEventListener("change", syncPayBtn);
function syncPayBtn() {
  const ok = $("payCard").value.length === 19 && $("payExp").value.length === 5 &&
             $("payCvc").value.length === 3 && $("payAgree").checked && $("payAgreeData").checked;
  $("payBtn").disabled = !ok;
}
$("payBtn").addEventListener("click", () => {
  $("payBtn").disabled = true;
  $("payBtn").textContent = "결제 처리 중…";
  setTimeout(() => {
    const q = quoteRows();
    const job = createJob(q);
    $("payModalMsg").innerHTML = `<b>${job.name}</b> 작업이 ${q.g.name} × ${order.gpuCount} 클러스터에 등록되었습니다.<br>
      결제 금액(데모): <b>${fmtNum(q.krwTotal,0)}원</b><br>
      실제 예상 학습 ${fmtHours(q.r.hours)}를 데모에서는 약 ${job.demoSec}초로 압축해 시뮬레이션합니다.`;
    $("payModal").classList.add("show");
    $("payBtn").textContent = "결제하고 학습 시작";
  }, 1400);
});
$("payModalGo").addEventListener("click", () => {
  $("payModal").classList.remove("show");
  resetWizard();
  showView("dashboard");
});

// ---------- 위저드 네비게이션 ----------
function goStep(n) {
  order.step = n;
  document.querySelectorAll(".wstep").forEach((el, i) => el.classList.toggle("active", i === n - 1));
  document.querySelectorAll(".stepper .stp").forEach(el => {
    const s = parseInt(el.dataset.step);
    el.classList.toggle("cur", s === n);
    el.classList.toggle("done", s < n);
  });
  if (n === 2) buildStep2();
  if (n === 3) buildStep3();
  if (n === 4) buildStep4();
}
document.querySelectorAll("[data-goto]").forEach(b => b.addEventListener("click", () => goStep(parseInt(b.dataset.goto))));
function resetWizard() {
  order.files = []; order.tokens = 0; order.samples = 0; order.dataType = null;
  order.textBytes = 0; order.imgBytes = 0; order.imgCount = 0;
  $("manualType").value = ""; $("fileInput").value = ""; $("jobName").value = "";
  $("storageOpt").value = "none"; order.storageUse = false;
  $("payCard").value = ""; $("payExp").value = ""; $("payCvc").value = ""; $("payAgree").checked = false;
  $("payAgreeData").checked = false;
  renderFileList(); syncStep1(); goStep(1);
}

/* ==========================================================================
   학습 작업 (대시보드 시뮬레이션)
   ========================================================================== */
const JOBS_KEY = "trainlab_jobs";
function loadJobs() { try { return JSON.parse(localStorage.getItem(JOBS_KEY)) || []; } catch { return []; } }
function saveJobs(jobs) { localStorage.setItem(JOBS_KEY, JSON.stringify(jobs)); }

function createJob(q) {
  const jobs = loadJobs();
  const isText = order.dataType === "text";
  const job = {
    id: "job_" + Date.now().toString(36),
    name: $("jobName").value || "학습 작업",
    created: Date.now(),
    taskLabel: TASK_LABELS[order.task],
    modelLabel: order.modelLabel,
    dataLabel: isText ? fmtTokens(order.tokens) + " (" + fmtBytes(order.textBytes) + ")" : fmtNum(order.samples,0) + "장",
    gpuName: q.g.name, gpuCount: order.gpuCount,
    precision: order.precision.toUpperCase(), mfu: Math.round(q.r.mfu * 100),
    hours: q.r.hours, costUsd: q.r.cost, krwTotal: q.krwTotal,
    dataBytes: orderDataBytes(),
    storage: order.storageUse ? { months: order.storageMonths, feeKrw: q.storageKrw } : null,
    flops: q.r.flops, effPflops: order.gpuCount * q.r.tflops * q.r.mfu / 1000,
    // 데모: 실제 예상시간을 40~150초로 압축
    demoSec: Math.max(40, Math.min(150, Math.round(40 + Math.log10(1 + q.r.hours) * 35))),
    startedAt: Date.now(), status: "running",
    seed: Math.floor(Math.random() * 1e9),
  };
  jobs.unshift(job);
  saveJobs(jobs);
  return job;
}
window.deleteJob = function(id) {
  saveJobs(loadJobs().filter(j => j.id !== id));
  renderJobs();
};
window.downloadModel = function(id) {
  const j = loadJobs().find(x => x.id === id);
  if (!j) return;
  const meta = {
    note: "TrainLab 데모 체크포인트 메타데이터 (실제 가중치 아님)",
    job: j.name, model: j.modelLabel, task: j.taskLabel, data: j.dataLabel,
    gpu: j.gpuName + " x " + j.gpuCount, estimated_hours: j.hours, total_flops: j.flops,
  };
  const blob = new Blob([JSON.stringify(meta, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = j.name + "-checkpoint.json";
  a.click();
  URL.revokeObjectURL(a.href);
};

// 시드 기반 의사난수 (loss 곡선·로그 재현용)
function mulberry32(seed) {
  return function() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function lossAt(x, rnd) { return 2.6 * Math.exp(-4 * x) + 0.42 + (rnd() - 0.5) * 0.12 * Math.exp(-2 * x); }

function lossChartSvg(job, p) {
  const rnd = mulberry32(job.seed);
  const N = 80, shown = Math.max(2, Math.floor(N * p));
  let pts = [];
  for (let i = 0; i < shown; i++) {
    const x = i / N;
    const y = lossAt(x, rnd);
    pts.push([12 + x * 280, 106 - (y / 3.2) * 92]);
  }
  const poly = pts.map(p2 => p2[0].toFixed(1) + "," + p2[1].toFixed(1)).join(" ");
  const last = pts[pts.length - 1];
  const area = `M${pts[0][0].toFixed(1)},106 L` + poly.replace(/ /g, " L") + ` L${last[0].toFixed(1)},106 Z`;
  const uid = job.id.replace(/[^a-z0-9]/gi, "");
  const lossVal = ((106 - last[1]) / 92 * 3.2).toFixed(3);
  const gridY = [30, 55, 80];
  return `<svg viewBox="0 0 300 122" style="width:100%;height:152px">
    <defs>
      <linearGradient id="lg${uid}" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0" stop-color="#22d3ee"/><stop offset=".55" stop-color="#8b5cf6"/><stop offset="1" stop-color="#86e01e"/>
      </linearGradient>
      <linearGradient id="la${uid}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="rgba(139,92,246,.30)"/><stop offset="1" stop-color="rgba(139,92,246,0)"/>
      </linearGradient>
      <filter id="gl${uid}" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="2.4" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>
    ${gridY.map(y => `<line x1="12" y1="${y}" x2="292" y2="${y}" stroke="rgba(126,145,255,.10)" stroke-width="1" stroke-dasharray="3 5"/>`).join("")}
    <line x1="12" y1="106" x2="292" y2="106" stroke="rgba(126,145,255,.22)" stroke-width="1"/>
    <path d="${area}" fill="url(#la${uid})"/>
    <polyline points="${poly}" fill="none" stroke="url(#lg${uid})" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" filter="url(#gl${uid})"/>
    <circle cx="${last[0]}" cy="${last[1]}" r="6" fill="rgba(134,224,30,.18)">
      <animate attributeName="r" values="4;8;4" dur="1.8s" repeatCount="indefinite"/>
    </circle>
    <circle cx="${last[0]}" cy="${last[1]}" r="3" fill="#86e01e" filter="url(#gl${uid})"/>
    <text x="290" y="${Math.max(16, last[1] - 10)}" fill="#86e01e" font-size="9.5" font-family="JetBrains Mono, monospace" text-anchor="end">loss ${lossVal}</text>
  </svg>`;
}
function logLines(job, p) {
  const rnd = mulberry32(job.seed + 7);
  const totalSteps = 10000, lines = [];
  const shown = Math.floor(40 * p);
  lines.push(`[클러스터] ${job.gpuName} × ${job.gpuCount} 할당 완료 · ${job.precision} · NCCL 초기화 OK`);
  lines.push(`[데이터] ${job.dataLabel} 로드 · 셔플 및 토크나이즈 완료`);
  for (let k = 1; k <= shown; k++) {
    const x = k / 40;
    const loss = lossAt(x, rnd).toFixed(3);
    const lr = (3e-5 * (1 - x * 0.9)).toExponential(1);
    lines.push(`step ${(k * totalSteps / 40).toLocaleString()}/${totalSteps.toLocaleString()}  loss=${loss}  lr=${lr}  MFU=${job.mfu}%  ${(job.effPflops).toFixed(1)} PFLOP/s`);
  }
  if (p >= 1) lines.push(`[완료] 체크포인트 저장 완료 · 총 ${fmtHours(job.hours)} 상당 학습 (데모 압축)`);
  return lines.join("\n");
}

// 작업별 데이터 보관 상태 안내
function retentionNotice(j) {
  const fmtDate = ts => new Date(ts).toLocaleDateString("ko-KR", { year:"numeric", month:"long", day:"numeric" });
  const size = j.dataBytes ? fmtBytes(j.dataBytes) : "";
  if (j.storage) {
    if (j.status !== "completed")
      return `<div class="retention stored">${icon("lock","sm")}<span>학습 데이터(${size}) 임시 보관 중 — 학습 완료 후 스토리지 서비스로 ${j.storage.months}개월간 안전하게 보관됩니다.</span></div>`;
    const expireAt = j.completedAt + j.storage.months * 30 * 86400e3;
    return `<div class="retention stored">${icon("lock","sm")}<span>스토리지 서비스 보관 중 — ${size}, ${j.storage.months}개월 (보관 만료: ${fmtDate(expireAt)}) · 이용 기간 동안 안전하게 보관되며, 만료 후 15일 유예를 거쳐 영구 삭제됩니다.</span></div>`;
  }
  if (j.status !== "completed")
    return `<div class="retention temp">${icon("trash","sm")}<span>학습 데이터(${size})는 임시 보관 중이며, 학습이 끝난 후 삭제 조치됩니다 — 완료 15일 후 영구 삭제. 계속 보관하려면 스토리지 서비스가 필요합니다.</span></div>`;
  const deleteAt = j.completedAt + RETENTION_DAYS * 86400e3;
  const dLeft = Math.max(0, Math.ceil((deleteAt - Date.now()) / 86400e3));
  return `<div class="retention temp">${icon("trash","sm")}<span>원본 학습 데이터(${size})는 <b>${fmtDate(deleteAt)}</b>에 영구 삭제됩니다 (D-${dLeft}). 삭제된 데이터는 복구할 수 없으며, 계속 보관하려면 스토리지 서비스를 이용하세요.</span></div>`;
}

function renderJobs() {
  const jobs = loadJobs();
  if (!jobs.length) {
    $("jobList").innerHTML = `<div class="empty"><div class="big">${icon("folder","xl")}</div>
      아직 학습 작업이 없습니다.<br><br>
      <button class="btn" data-view="new" onclick="showView('new')">첫 학습 시작하기 →</button></div>`;
    return;
  }
  let changed = false;
  const html = jobs.map(j => {
    let p = j.status === "completed" ? 1 : Math.min(1, (Date.now() - j.startedAt) / (j.demoSec * 1000));
    if (p >= 1 && j.status === "running") { j.status = "completed"; j.completedAt = Date.now(); changed = true; }
    const running = j.status === "running";
    return `<div class="job">
      <div class="jhead">
        <div>
          <h3>${j.name}</h3>
          <small>${j.taskLabel} · ${j.modelLabel} · ${j.dataLabel} · ${j.gpuName} × ${j.gpuCount}</small>
        </div>
        <span class="jstatus ${j.status}">${running ? '<span class="pulse"></span>학습 중' : icon("checkcircle","xs") + " 완료"}</span>
      </div>
      <div class="prog">
        <div class="bar-track"><div class="bar-fill" style="width:${(p*100).toFixed(1)}%"></div></div>
        <div class="meta">
          <span>${(p*100).toFixed(1)}%</span>
          <span>실제 예상 ${fmtHours(j.hours)} · 데모 ${j.demoSec}초 압축${running ? " · 남은 데모 " + Math.max(0, Math.ceil(j.demoSec * (1-p))) + "초" : ""}</span>
        </div>
      </div>
      <div class="jgrid">
        <div class="losschart"><h4>학습 손실 (loss)</h4>${lossChartSvg(j, p)}</div>
        <div class="logbox"><h4>학습 로그</h4><pre>${logLines(j, p)}</pre></div>
      </div>
      <div class="jstats">
        <div class="js">GPU 사용률<b>${running ? (88 + Math.floor((mulberry32(j.seed + Math.floor(Date.now()/1000))() * 10))) + "%" : "—"}</b></div>
        <div class="js">실효 처리량<b>${j.effPflops.toFixed(1)} PFLOP/s</b></div>
        <div class="js">누적 비용 (데모)<b>${fmtNum(Math.round(j.krwTotal * p), 0)}원</b></div>
        <div class="js">결제 금액<b>${fmtNum(j.krwTotal, 0)}원</b></div>
      </div>
      ${retentionNotice(j)}
      <div style="margin-top:14px;display:flex;gap:8px">
        ${j.status === "completed" ? `<button class="btn sm" onclick="downloadModel('${j.id}')">${icon("box","xs")} 모델 다운로드</button>` : ""}
        <button class="btn ghost sm" onclick="deleteJob('${j.id}')">삭제</button>
      </div>
    </div>`;
  }).join("");
  if (changed) saveJobs(jobs);
  $("jobList").innerHTML = html;
  // 로그를 항상 마지막 줄로 스크롤
  document.querySelectorAll(".logbox pre").forEach(el => { el.scrollTop = el.scrollHeight; });
}
window.showView = showView;

let tickTimer = null;
function startTick() {
  stopTick();
  tickTimer = setInterval(() => {
    if (loadJobs().some(j => j.status === "running")) renderJobs();
  }, 1000);
}
function stopTick() { if (tickTimer) { clearInterval(tickTimer); tickTimer = null; } }

/* ==========================================================================
   비용 계산기 뷰 (기존 상세 계산기)
   ========================================================================== */
const PARAM_PRESETS = [ [1,"1B"], [3,"3B"], [7,"7B"], [13,"13B"], [34,"34B"], [70,"70B"], [180,"180B"], [405,"405B"] ];
const TOKEN_PRESETS = [ [1e9,"1B (≈4GB)"], [1e10,"10B (≈40GB)"], [1e11,"100B (≈0.4TB)"], [1e12,"1T (≈4TB)"], [2e12,"2T (≈8TB)"], [15e12,"15T (≈60TB)"] ];
const IMAGE_PRESETS = [ [5e4,"5만"], [1.3e6,"130만 (ImageNet)"], [1e7,"1,000만"], [1e8,"1억"] ];

const calcState = { gpu:"h100", prices:{}, moeActive:null };
GPUS.forEach(g => calcState.prices[g.id] = g.price);

function currentVModel() {
  const list = VMODELS[$("task").value] || VMODELS.vision;
  return list.find(m => m.id === $("vmodel").value) || list[0];
}

function totalFlops() {
  const task = $("task").value;
  const epochs = Math.max(0.1, parseFloat($("epochs").value) || 1);
  if (isLLMTask(task)) {
    const p = calcState.moeActive ? calcState.moeActive * 1e9
            : (parseFloat($("params").value) || 0) * parseFloat($("paramUnit").value);
    const t = (parseFloat($("tokens").value) || 0) * parseFloat($("tokenUnit").value);
    const factor = task === "lora" ? 4 : 6;
    return factor * p * t * epochs;
  } else {
    const vm = currentVModel();
    const n = (parseFloat($("images").value) || 0) * parseFloat($("imageUnit").value);
    return vm.gflops * 1e9 * 3 * n * epochs;
  }
}

function effTflops(gpu, precision) {
  const t = (precision === "fp8" && gpu.fp8) ? gpu.fp8 : gpu.bf16;
  const fellBack = precision === "fp8" && !gpu.fp8;
  return { tflops: t, fellBack };
}

function calc(gpu) {
  const flops = totalFlops();
  const n = Math.max(1, parseInt($("gpuCount").value) || 1);
  const mfu = parseInt($("mfu").value) / 100;
  const { tflops, fellBack } = effTflops(gpu, $("precision").value);
  const effective = n * tflops * 1e12 * mfu;
  const hours = flops / effective / 3600;
  const price = calcState.prices[gpu.id];
  const cost = hours * n * price;
  return { flops, hours, cost, price, n, tflops, mfu, fellBack, effPflops: effective/1e15 };
}

function memCheck(gpu) {
  const task = $("task").value;
  if (!isLLMTask(task)) return null;
  const p = (parseFloat($("params").value) || 0) * parseFloat($("paramUnit").value);
  const bytesPerParam = task === "lora" ? 4 : 18;
  const needGB = p * bytesPerParam / 1e9;
  const n = Math.max(1, parseInt($("gpuCount").value) || 1);
  const haveGB = gpu.mem * n;
  return { needGB, haveGB, ok: haveGB >= needGB * 1.2 };
}

function updateDataLabels() {
  const p = (parseFloat($("params").value) || 0) * parseFloat($("paramUnit").value);
  $("paramLabel").textContent = fmtNum(p/1e9, p < 1e9 ? 2 : 1) + "B" +
    (calcState.moeActive ? " (MoE · 연산은 활성 " + calcState.moeActive + "B 기준)" : "");
  const t = (parseFloat($("tokens").value) || 0) * parseFloat($("tokenUnit").value);
  $("tokenLabel").textContent = fmtTokens(t) + " ≈ " + fmtBytes(t * BYTES_PER_TOKEN) + " 텍스트";
  const task = $("task").value;
  if (!isLLMTask(task)) {
    const vm = currentVModel();
    const bps = vm.bytes || BYTES_PER_IMAGE;
    const n = (parseFloat($("images").value) || 0) * parseFloat($("imageUnit").value);
    $("imageLabel").textContent = "≈ " + fmtBytes(n * bps) +
      (vm.bytes ? " (샘플당 " + fmtBytes(vm.bytes) + " 기준)" : " (장당 500KB 기준)");
  }
}

function render() {
  updateDataLabels();
  const gpu = GPUS.find(g => g.id === calcState.gpu);
  const r = calc(gpu);
  const fx = parseFloat($("fx").value) || 1380;

  $("resGpuName").textContent = "— " + gpu.name + " × " + r.n;
  const [fv, fu] = fmtFlops(r.flops);
  $("rFlops").innerHTML = fv + " <small>" + fu + "</small>";
  $("rFlopsSub").textContent = "MFU " + Math.round(r.mfu*100) + "% 적용, 실효 " + fmtNum(r.effPflops,1) + " PFLOP/s";
  $("rTime").textContent = fmtHours(r.hours);
  $("rTimeSub").textContent = "GPU·시간 " + fmtNum(r.hours * r.n, 0) + " (총합)" + (r.fellBack ? " · FP8 미지원→BF16" : "");
  $("rCost").textContent = fmtUsd(r.cost);
  $("rCostKrw").textContent = "약 " + fmtKrw(r.cost * fx);
  $("rHourly").innerHTML = fmtUsd(r.price * r.n) + " <small>/시간</small>";
  $("rHourlySub").textContent = gpu.name + " $" + r.price + "/시간 × " + r.n + "개";

  const mc = memCheck(gpu);
  const mw = $("memWarn");
  if (mc) {
    mw.style.display = "block";
    mw.className = "memwarn " + (mc.ok ? "ok" : "bad");
    mw.innerHTML = (mc.ok ? "✅ " : "⚠️ ") +
      "예상 필요 메모리 약 <b>" + fmtNum(mc.needGB,0) + " GB</b> (옵티마이저·그래디언트 포함, ZeRO-3 분산 가정) / 보유 " +
      fmtNum(mc.haveGB,0) + " GB (" + gpu.name + " " + gpu.mem + "GB × " + r.n + ")" +
      (mc.ok ? "" : " — 메모리가 부족할 수 있습니다. GPU 수를 늘리거나 QLoRA·오프로딩을 고려하세요.");
  } else {
    mw.style.display = "none";
  }
  renderTable();
}

function renderTable() {
  const tbody = document.querySelector("#cmpTable tbody");
  const rows = GPUS.map(g => ({ g, r: calc(g) }));
  const maxCost = Math.max(...rows.map(x => x.r.cost).filter(isFinite), 1e-9);
  const fx = parseFloat($("fx").value) || 1380;

  let html = "", lastCat = null;
  rows.forEach(({g, r}) => {
    if (g.cat !== lastCat) { html += `<tr class="catrow"><td colspan="6">${g.cat}</td></tr>`; lastCat = g.cat; }
    html += `
    <tr class="${g.id === calcState.gpu ? "sel" : ""}" data-gpu="${g.id}">
      <td class="gname">${g.name}<small>${g.arch} · ${g.mem}GB · BF16 ${fmtNum(g.bf16,0)} TFLOPS</small></td>
      <td>${fmtNum(r.tflops,0)} TFLOPS${r.fellBack ? " <span style='color:var(--warn)'>(BF16 대체)</span>" : ""}</td>
      <td>${fmtHours(r.hours)}</td>
      <td><input class="price" type="number" step="0.1" min="0" data-price="${g.id}" value="${calcState.prices[g.id]}"></td>
      <td><b>${fmtUsd(r.cost)}</b><br><small style="color:var(--muted)">${fmtKrw(r.cost*fx)}</small></td>
      <td class="bar-cell"><div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, r.cost/maxCost*100)}%"></div></div></td>
    </tr>`;
  });
  tbody.innerHTML = html;

  tbody.querySelectorAll("input.price").forEach(inp => {
    inp.addEventListener("input", e => {
      calcState.prices[e.target.dataset.price] = parseFloat(e.target.value) || 0;
      renderPartialCosts();
    });
    inp.addEventListener("click", e => e.stopPropagation());
  });
  tbody.querySelectorAll("tr[data-gpu]").forEach(tr => {
    tr.addEventListener("click", () => { calcState.gpu = tr.dataset.gpu; syncGpuGrid(); render(); });
  });
}

function renderPartialCosts() {
  const gpu = GPUS.find(g => g.id === calcState.gpu);
  const r = calc(gpu);
  const fx = parseFloat($("fx").value) || 1380;
  $("rCost").textContent = fmtUsd(r.cost);
  $("rCostKrw").textContent = "약 " + fmtKrw(r.cost * fx);
  $("rHourly").innerHTML = fmtUsd(r.price * r.n) + " <small>/시간</small>";
  $("rHourlySub").textContent = gpu.name + " $" + r.price + "/시간 × " + r.n + "개";

  const rows = GPUS.map(g => ({ g, r: calc(g) }));
  const maxCost = Math.max(...rows.map(x => x.r.cost).filter(isFinite), 1e-9);
  rows.forEach(({g, r}) => {
    const tr = document.querySelector(`tr[data-gpu="${g.id}"]`);
    if (!tr) return;
    const tds = tr.querySelectorAll("td");
    tds[4].innerHTML = `<b>${fmtUsd(r.cost)}</b><br><small style="color:var(--muted)">${fmtKrw(r.cost*fx)}</small>`;
    tr.querySelector(".bar-fill").style.width = Math.max(2, r.cost/maxCost*100) + "%";
  });
}

function buildChips(elId, presets, apply) {
  $(elId).innerHTML = presets.map(([v, l]) => `<span class="chip" data-v="${v}">${l}</span>`).join("");
  $(elId).querySelectorAll(".chip").forEach(c => {
    c.addEventListener("click", () => {
      apply(parseFloat(c.dataset.v));
      $(elId).querySelectorAll(".chip").forEach(x => x.classList.remove("on"));
      c.classList.add("on");
      render();
    });
  });
}

function syncGpuGrid() {
  document.querySelectorAll("#gpuGrid .gpu-item").forEach(el => el.classList.toggle("on", el.dataset.gpu === calcState.gpu));
}

function buildGpuGrid() {
  let html = "", lastCat = null;
  GPUS.forEach(g => {
    if (g.cat !== lastCat) { html += `<div class="gpu-cat">${g.cat}</div>`; lastCat = g.cat; }
    html += `
    <div class="gpu-item ${g.id === calcState.gpu ? "on" : ""}" data-gpu="${g.id}">
      <div class="nm">${g.name}</div>
      <div class="sp">${g.arch} · ${g.mem}GB<br>BF16 ${fmtNum(g.bf16,0)} TFLOPS</div>
    </div>`;
  });
  $("gpuGrid").innerHTML = html;
  document.querySelectorAll("#gpuGrid .gpu-item").forEach(el => {
    el.addEventListener("click", () => { calcState.gpu = el.dataset.gpu; syncGpuGrid(); render(); });
  });
}

function buildLLMSelect() {
  const groups = [...new Set(LLM_PRESETS.map(m => m.g))];
  $("llmModel").innerHTML = '<option value="custom">직접 입력</option>' +
    groups.map(g => `<optgroup label="${g}">` +
      LLM_PRESETS.filter(m => m.g === g).map(m => `<option value="${m.id}">${m.name}</option>`).join("") +
      "</optgroup>").join("");
}

function onLLMModelChange() {
  const m = LLM_PRESETS.find(x => x.id === $("llmModel").value);
  if (!m) {
    calcState.moeActive = null;
    $("llmModelInfo").textContent = "";
    render();
    return;
  }
  $("params").value = m.b;
  $("paramUnit").value = "1e9";
  calcState.moeActive = m.active || null;
  if ($("task").value === "pretrain" && m.tokens) {
    if (m.tokens >= 1e12) { $("tokens").value = Math.round(m.tokens/1e12*10)/10; $("tokenUnit").value = "1e12"; }
    else { $("tokens").value = m.tokens/1e9; $("tokenUnit").value = "1e9"; }
  }
  $("llmModelInfo").textContent = m.tokens ? "사전학습 " + fmtTokens(m.tokens) + " (공개 기준)" : "학습 토큰 비공개";
  render();
}

function buildVModelSelect(task) {
  const list = VMODELS[task] || VMODELS.vision;
  $("vmodel").innerHTML = list.map(m => `<option value="${m.id}">${m.name}</option>`).join("");
  updateVInfo(task);
}
function updateVInfo(task) {
  const list = VMODELS[task] || VMODELS.vision;
  const vm = list.find(m => m.id === $("vmodel").value) || list[0];
  $("vmodelInfo").textContent = "순전파 ≈ " + fmtGflops(vm.gflops) + "/샘플";
}

function applyEarthDefaults() {
  const vm = currentVModel();
  if (!vm.defSamples) return;
  $("images").value = vm.defSamples[0];
  $("imageUnit").value = vm.defSamples[1];
  $("epochs").value = vm.defEpochs;
}

function applyCount(v) {
  if (v >= 1e6) { $("images").value = v/1e6; $("imageUnit").value = "1e6"; }
  else if (v >= 1e4) { $("images").value = v/1e4; $("imageUnit").value = "1e4"; }
  else { $("images").value = v/1e3; $("imageUnit").value = "1e3"; }
}

function onTaskChange() {
  const task = $("task").value;
  const isLLM = isLLMTask(task);
  $("llmInputs").style.display = isLLM ? "block" : "none";
  $("visionInputs").style.display = isLLM ? "none" : "block";
  if (!isLLM) {
    buildVModelSelect(task);
    const isEarth = task === "earth2";
    $("vcountLbl").textContent = isEarth ? "학습 샘플 수 (시점)" : "학습 이미지 수";
    const noun = isEarth ? "개" : "장";
    const units = $("imageUnit").options;
    units[0].text = "천 " + noun; units[1].text = "만 " + noun; units[2].text = "백만 " + noun;
    buildChips("imageChips", isEarth ? EARTH2_PRESETS : IMAGE_PRESETS, applyCount);
    if (isEarth) applyEarthDefaults();
  } else {
    $("epochs").value = 1;
  }
  render();
}

// ---------- 계산기 초기화 ----------
buildGpuGrid();
buildLLMSelect();
$("llmModel").addEventListener("change", onLLMModelChange);
["params","paramUnit"].forEach(id => $(id).addEventListener("input", () => {
  $("llmModel").value = "custom"; calcState.moeActive = null; $("llmModelInfo").textContent = "";
}));
buildChips("paramChips", PARAM_PRESETS, v => {
  $("params").value = v; $("paramUnit").value = "1e9";
  $("llmModel").value = "custom"; calcState.moeActive = null; $("llmModelInfo").textContent = "";
});
buildChips("tokenChips", TOKEN_PRESETS, v => {
  if (v >= 1e12) { $("tokens").value = v/1e12; $("tokenUnit").value = "1e12"; }
  else if (v >= 1e9) { $("tokens").value = v/1e9; $("tokenUnit").value = "1e9"; }
  else { $("tokens").value = v/1e6; $("tokenUnit").value = "1e6"; }
});
buildChips("imageChips", IMAGE_PRESETS, applyCount);

$("task").addEventListener("change", onTaskChange);
$("vmodel").addEventListener("change", () => {
  updateVInfo($("task").value);
  if ($("task").value === "earth2") applyEarthDefaults();
  render();
});
$("mfu").addEventListener("input", () => { $("mfuLabel").textContent = $("mfu").value + "%"; render(); });
["params","paramUnit","tokens","tokenUnit","epochs","gpuCount","precision","fx","images","imageUnit"].forEach(id => {
  $(id).addEventListener("input", render);
  $(id).addEventListener("change", render);
});
render();

// ==================== 시작 뷰 ====================
initIcons();
goStep(1);
syncStep1();
const initial = (location.hash || "#home").slice(1);
showView(["home","new","dashboard","calc"].includes(initial) ? initial : "home");

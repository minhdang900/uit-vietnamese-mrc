// ════════════════════════════════════════════════════════════════════════════
// Deck báo cáo CS116 · Đề tài T11 — 21 slide theo cấu trúc sáu mục:
//
//   1 TỔNG QUAN               bối cảnh & động lực · phát biểu bài toán · đóng góp
//   2 CÁC CÔNG TRÌNH LIÊN QUAN hướng tiếp cận · bộ dữ liệu đã có · GAP
//   3 XÂY DỰNG DỮ LIỆU        nguồn & split · phân bố nhãn + độ dài · phân bố chủ đề
//   4 PHƯƠNG PHÁP & THỰC NGHIỆM bốn hệ thống · kiến trúc · bất biến · metric ·
//                             hai quyết định kỹ thuật · pipeline huấn luyện
//   5 KẾT QUẢ & PHÂN TÍCH     bảng chính · hai phát hiện
//   6 ỨNG DỤNG, KẾT LUẬN      sản phẩm chạy được · hạn chế & hướng phát triển · kết luận
//
// NGUỒN SỐ LIỆU — hai loại, tách rõ:
//
//   (a) Mọi EM / F1 / độ trễ / đường cong / siêu tham số được ĐỌC TỪ
//       ../results/*.json lúc dựng deck. Không con số nào trong nhóm này được
//       gõ tay, nên huấn luyện lại rồi dựng lại là slide tự đúng theo.
//       Bất biến này được test: tests/test_report_assets.py::
//       test_the_slide_deck_contains_no_hand_written_metrics soi cả tệp này.
//
//   (b) Thống kê mô tả dữ liệu (số câu hỏi, phân bố độ dài, phân bố chủ đề,
//       phân mảnh token) nằm trong hằng số DATA bên dưới, kèm script đã sinh ra
//       chúng. Sửa dữ liệu thì chạy lại script rồi cập nhật đúng một khối này.
//
// Dựng lại:  cd slides && node build_deck.js     (cần: npm install pptxgenjs)
// ════════════════════════════════════════════════════════════════════════════

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const RESULTS = path.join(__dirname, "..", "results");
const read = (f) => JSON.parse(fs.readFileSync(path.join(RESULTS, f), "utf8"));

// ── (a) Số đo — đọc từ results/ ──────────────────────────────────────────────
const EV = {
  tfidf:    read("eval_baseline_validation.json"),
  visobert: read("eval_visobert_validation.json"),
  xlmr:     read("eval_xlmr_validation.json"),
  mbert:    read("eval_mbert_validation.json"),
};
const CURVE = {
  mbert:    read("training_curve_mbert.json"),
  visobert: read("training_curve_visobert.json"),
};

// định dạng số kiểu Việt Nam: dấu phẩy thập phân, dấu chấm hàng nghìn
const fmt = (x, dp = 2) => {
  const [i, d] = Number(x).toFixed(dp).split(".");
  return i.replace(/\B(?=(\d{3})+(?!\d))/g, ".") + (d ? "," + d : "");
};
const em  = (k) => fmt(EV[k].overall.EM);
const f1  = (k) => fmt(EV[k].overall.F1);
const lat = (k) => fmt(EV[k].avg_latency_ms);
const ansEM = (k) => fmt(EV[k].answerable_only.EM);
const ansF1 = (k) => fmt(EV[k].answerable_only.F1);
const impEM = (k) => fmt(EV[k].impossible_only.EM);
const nOf   = (k) => EV[k].overall.count;
const nImp  = (k) => EV[k].impossible_only.count;
const nAns  = (k) => EV[k].answerable_only.count;
// tỉ lệ câu impossible của chính mẫu đánh giá — dùng để đối chiếu với EM tổng
const impShare = fmt(100 * nImp("mbert") / nOf("mbert"));
const CFG = (k) => CURVE[k].config;
const qtype = (k, t) => EV[k].by_question_type[t];
const bylen = (k, b) => EV[k].by_context_length[b];

// ── (b) Thống kê mô tả dữ liệu ───────────────────────────────────────────────
// Nguồn: 06_BaoCao_T11/05_BANG_CHUNG/stats.py (data_stats.json),
//        tokenizer_stats.py (tokenizer_stats.txt), ci.py (confidence_intervals.txt).
const DATA = {
  split: {           //            train        validation   test
    questions:  ["28.454",   "3.814",   "7.301"],
    contexts:   ["4.101",    "557",     "1.241"],
    articles:   ["138",      "19",      "48"],
    impossible: ["9.216",    "1.161",   "0"],
    gradeable:  ["28.454",   "3.814",   "0"],
    impPct:     ["32,39%",   "30,44%",  "—"],
    ctxWords:   ["179,0 / 160 / 1.537", "167,6 / 152 / 618", "175,8 / 159 / 823"],
    ansWords:   ["9,95 / 6 / 31",       "9,81 / 6 / 30",     "—"],
  },
  sharedCtx: { trainVal: "0", trainTest: "449", valTest: "133" },
  longCtx:   { n: "239", of: "4.101", pct: "5,8%" },   // đoạn văn train > 300 từ
  // phân bố chủ đề validation (19 bài viết, 3.814 câu) — đếm theo trường title
  topics: [
    ["Franklin D. Roosevelt", 503], ["Paris", 462], ["Chiến tranh Vùng Vịnh", 379],
    ["Canada", 356], ["Mahatma Gandhi", 324],
  ],
  topicTail: [["Gia cầm", 59], ["Uganda", 59], ["Kiến", 57], ["Nước biển", 53],
              ["Nội chiến Anh", 53], ["Montréal", 42]],
  topicTop5Pct: "53,1%",       // 5 bài viết lớn nhất / 3.814 câu
  topicSciN: "169", topicSciPct: "4,4%",  // Kiến + Gia cầm + Nước biển
  // bẫy lấy mẫu: 300 câu đầu tệp validation
  firstNArticles: 1, firstN: "300",
  biasEM: { head: "42,00", random: "52,00" },  // mBERT epoch 2, 300 câu
  subsetTop5Pct: "57,0%", subsetArticles: "19/19",
  wh: [["gì", "30,3%"], ["nào", "27,6%"], ["ai", "9,1%"], ["như thế nào", "8,3%"],
       ["bao nhiêu", "7,9%"], ["vì sao + tại sao", "5,1%"]],
  tok: {   //                        mBERT        ViSoBERT
    vocab:    ["119.547", "15.002"],
    params:   ["177,3 M", "97,0 M"],
    sentence: ["17 token", "23 token"],
    ctxMean:  ["204,6", "326,5"],
    perWord:  ["1,22", "1,95"],
    over357:  ["16 / 557", "154 / 557"],
  },
  features: ["30.540", "40.984"],   // số cửa sổ huấn luyện, từ nhật ký finetune
  // L_max suy luận của mBERT: mặc định của run_eval.py, không ghi trong training_curve_mbert.json
  lmaxMbert: "30",
  ci: { em: "±4,4", impGap: "+13,67 điểm ± 11,03", ansF1Gap: "1,60" },
  tests: { fast: "423", full: "474", files: "20", model: "51" },
  docker: { size: "2,49 GB", zipped: "501 MB", from: "10,2 GB" },
};

// ── Hệ thiết kế "Organic" — sao chép từ src/demo/theme.py TOKENS ─────────────
const C = {
  bg:        "F5EAD8",  surface: "EBDDC5",  card:    "F9F4ED",
  text:      "201E1D",  muted:   "645C50",  line:    "D3C9BA",
  accent:    "C67139",  sage:    "7A8A5E",
  accentSoft:"FFF2EB",  sageSoft:"F0FAE1",
  accentDeep:"8C491A",  sageDeep:"3D472B",
  dark:      "2E2B25",  dim:     "A19786",
};
const HEAD = "Baloo 2";
const BODY = "Nunito";

const W = 13.333, H = 7.5, M = 0.75, CW = W - 2 * M;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Nhóm 7 — CS116";
pres.title  = "Hệ thống đọc hiểu và trả lời câu hỏi tiếng Việt";

const shadow = () => ({ type: "outer", color: "2E2B25", blur: 8, offset: 2, angle: 90, opacity: 0.12 });

function slide(dark = false) {
  const s = pres.addSlide();
  s.background = { color: dark ? C.dark : C.bg };
  return s;
}

function kicker(s, txt, dark = false) {
  s.addText(txt.toUpperCase(), {
    x: M, y: 0.46, w: CW, h: 0.28, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 11, bold: true, charSpacing: 2.2,
    color: C.accent, valign: "middle",
  });
}

function title(s, txt, dark = false, size = 34) {
  s.addText(txt, {
    x: M, y: 0.76, w: CW, h: 0.78, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: size, bold: true,
    color: dark ? "F5EAD8" : C.text, valign: "top", lineSpacing: size * 1.12,
  });
}

function footer(s, label, n, dark = false) {
  s.addText(label, {
    x: M, y: 6.92, w: CW - 0.6, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 10, color: dark ? C.dim : C.muted, valign: "middle",
  });
  s.addText(String(n), {
    x: W - M - 0.6, y: 6.92, w: 0.6, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 10, color: dark ? C.dim : C.muted,
    align: "right", valign: "middle",
  });
}

function card(s, x, y, w, h, fill = C.card, withShadow = true) {
  const o = { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.16 };
  if (withShadow) o.shadow = shadow();
  s.addShape(pres.ShapeType.roundRect, o);
}

function label(s, txt, x, y, w, color = C.muted) {
  s.addText(txt.toUpperCase(), {
    x, y, w, h: 0.24, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 9.5, bold: true, charSpacing: 1.6, color, valign: "middle",
  });
}

function body(s, txt, x, y, w, h, opt = {}) {
  s.addText(txt, {
    x, y, w, h, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: opt.size || 14, color: opt.color || C.text,
    valign: opt.valign || "top", lineSpacing: (opt.size || 14) * 1.45,
    bold: opt.bold || false, align: opt.align || "left", italic: opt.italic || false,
  });
}

function mono(s, txt, x, y, w, h, opt = {}) {
  s.addText(txt, {
    x, y, w, h, isTextBox: true, margin: 0,
    fontFace: "Courier New", fontSize: opt.size || 12, bold: opt.bold !== false,
    color: opt.color || C.accentDeep, align: opt.align || "left",
    valign: opt.valign || "middle",
  });
}

function stat(s, value, unit, cap, x, y, w, color = C.accent, vsize = 40) {
  s.addText([
    { text: value, options: { fontFace: HEAD, fontSize: vsize, bold: true, color } },
    { text: unit ? " " + unit : "", options: { fontFace: BODY, fontSize: 14, color: C.muted } },
  ], { x, y, w, h: vsize / 60, isTextBox: true, margin: 0, valign: "bottom" });
  s.addText(cap, {
    x, y: y + vsize / 60 - 0.02, w, h: 0.5, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 11.5, color: C.muted, lineSpacing: 15, valign: "top",
  });
}

// bảng theo kiểu .om-table: chỉ có đường kẻ ngang mảnh
function table(s, head, rows, x, y, w, colW, opt = {}) {
  const fs = opt.size || 12.5;
  const rowsOut = [
    head.map((t, i) => ({
      text: t.toUpperCase(),
      options: {
        fontFace: BODY, fontSize: fs - 2.5, bold: true, charSpacing: 1.2, color: C.muted,
        align: i === 0 ? "left" : "right", valign: "bottom",
        border: [{ type: "none" }, { type: "none" }, { type: "solid", color: C.accent, pt: 1 }, { type: "none" }],
      },
    })),
    ...rows.map((r) =>
      r.map((cell, i) => {
        const isObj = typeof cell === "object" && cell !== null;
        const txt = isObj ? cell.t : cell;
        return {
          text: txt,
          options: {
            fontFace: BODY, fontSize: fs,
            bold: isObj ? !!cell.b : false,
            color: isObj && cell.c ? cell.c : C.text,
            fill: isObj && cell.fill ? { color: cell.fill } : undefined,
            align: i === 0 ? "left" : "right", valign: "middle",
            border: [{ type: "none" }, { type: "none" }, { type: "solid", color: C.line, pt: 0.5 }, { type: "none" }],
          },
        };
      })
    ),
  ];
  s.addTable(rowsOut, {
    x, y, w, colW, rowH: opt.rowH || 0.36,
    margin: [3, 6, 3, 0], autoPage: false,
  });
}

// dải kết luận cuối slide
function keybox(s, headline, detail, y = 5.98, h = 0.82, tone = "sage") {
  const fill = tone === "sage" ? C.sageSoft : C.accentSoft;
  const deep = tone === "sage" ? C.sageDeep : C.accentDeep;
  card(s, M, y, CW, h, fill, false);
  body(s, headline, M + 0.35, y + 0.13, CW - 0.7, 0.34, { size: 14.5, bold: true, color: deep });
  if (detail) body(s, detail, M + 0.35, y + 0.44, CW - 0.7, h - 0.5, { size: 11.5, color: deep });
}

const chartBase = {
  showLegend: true, legendPos: "t", legendFontSize: 11, legendFontFace: BODY, legendColor: C.muted,
  chartColors: [C.accent, C.sage],
  showValue: true, dataLabelFontFace: BODY, dataLabelFontSize: 10, dataLabelColor: C.text,
  dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
  catAxisLabelFontFace: BODY, catAxisLabelFontSize: 11, catAxisLabelColor: C.text,
  valAxisLabelFontFace: BODY, valAxisLabelFontSize: 10, valAxisLabelColor: C.muted,
  valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" },
  catAxisLineShow: false, valAxisLineShow: false,
  barGapWidthPct: 45, plotArea: { fill: { color: C.bg } }, chartArea: { fill: { color: C.bg } },
};

// ══════════════════════════════════════════════════════════════════════════
// 1 — Bìa + lộ trình sáu mục
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide(true);
  s.addShape(pres.ShapeType.roundRect, {
    x: 8.9, y: 1.15, w: 3.7, h: 4.1, fill: { color: "3B362E" },
    line: { color: "3B362E", width: 0 }, rectRadius: 0.2,
  });
  s.addText("CS116 · ĐỀ TÀI T11 · UIT, ĐHQG-HCM", {
    x: M, y: 1.25, w: 7.6, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 12, bold: true, charSpacing: 2.4, color: C.accent,
  });
  s.addText("Hệ thống đọc hiểu\nvà trả lời câu hỏi tiếng Việt", {
    x: M, y: 1.75, w: 7.7, h: 2.0, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 44, bold: true, color: "F5EAD8", lineSpacing: 50,
  });
  s.addText("Vietnamese Extractive Machine Reading Comprehension\ntrên UIT-ViQuAD 2.0", {
    x: M, y: 3.85, w: 7.7, h: 0.9, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 15, color: C.dim, lineSpacing: 24,
  });
  s.addText("Bốn mô hình · " + nOf("mbert") + " câu kiểm định · mọi số đo sinh từ results/", {
    x: M, y: 4.85, w: 7.7, h: 0.35, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 12.5, italic: true, color: C.accent,
  });

  label(s, "Nhóm 7", 9.25, 1.5, 3.0, C.accent);
  const team = [
    ["Lê Quang Thi", "25210337"], ["Trần Trọng Tấn", "25210334"],
    ["Nguyễn Quang Lâm", "25210289"], ["Vỏ Cẩm Thu", "25210342"],
  ];
  team.forEach(([n, id], i) => {
    s.addText(n, { x: 9.25, y: 1.92 + i * 0.52, w: 2.3, h: 0.26, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 12.5, bold: true, color: "F5EAD8" });
    s.addText(id, { x: 9.25, y: 2.14 + i * 0.52, w: 2.3, h: 0.24, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 10.5, color: C.dim });
  });
  s.addShape(pres.ShapeType.line, { x: 9.25, y: 4.15, w: 3.0, h: 0, line: { color: "554E44", width: 1 } });
  label(s, "GVHD", 9.25, 4.32, 3.0, C.accent);
  s.addText("ThS. Nguyễn Hữu Quyền", {
    x: 9.25, y: 4.6, w: 3.0, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 12.5, bold: true, color: "F5EAD8" });

  label(s, "Nội dung — sáu mục", M, 5.5, 4.0, C.accent);
  ["1 · Tổng quan", "2 · Liên quan", "3 · Dữ liệu", "4 · Phương pháp",
   "5 · Kết quả", "6 · Kết luận"].forEach((t, i) => {
    const x = M + i * 1.99;
    s.addShape(pres.ShapeType.roundRect, { x, y: 5.82, w: 1.85, h: 0.46,
      fill: { color: "3B362E" }, line: { color: "3B362E", width: 0 }, rectRadius: 0.22 });
    s.addText(t, { x, y: 5.82, w: 1.85, h: 0.46, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 10.5, bold: true, color: C.dim, align: "center", valign: "middle" });
  });

  footer(s, "TP. Hồ Chí Minh · tháng 9 năm 2026", 1, true);
  s.addNotes("[0:00–0:30] Chào thầy và các bạn. Nhóm em làm đề tài T11 — đọc hiểu và trả lời câu hỏi tiếng Việt. Bài trình bày đi theo sáu mục: tổng quan và phát biểu bài toán, các công trình liên quan và khoảng trống nhóm đứng vào, quá trình xây dựng và phân tích dữ liệu, phương pháp và thực nghiệm, kết quả và phân tích, cuối cùng là ứng dụng, hạn chế và hướng phát triển.");
}

// ══════════════════════════════════════════════════════════════════════════
// 2 — 1.1 Bối cảnh & động lực
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "1 · Tổng quan");
  title(s, "Bối cảnh & động lực — vì sao bài toán này quan trọng");

  card(s, M, 1.78, 6.3, 3.95);
  body(s, "Tìm kiếm trả về TÀI LIỆU.\nĐọc hiểu máy trả về CÂU TRẢ LỜI.",
       M + 0.34, 2.05, 5.6, 1.1, { size: 21, bold: true });
  body(s, "Khối lượng văn bản một người phải đọc để tìm một thông tin cụ thể tăng nhanh hơn thời gian họ có. Công cụ tìm kiếm giải quyết một nửa: trả về tài liệu liên quan. MRC giải quyết nửa còn lại — chỉ thẳng vào đoạn chữ trả lời câu hỏi.",
       M + 0.34, 3.2, 5.6, 1.0, { size: 12.5, color: C.muted });
  label(s, "Ba chỗ dùng trực tiếp", M + 0.34, 4.3, 5.6, C.accent);
  ["Trợ lý học tập: hỏi trên giáo trình, quy chế, tài liệu môn học",
   "Hỏi đáp trên tài liệu nội bộ của doanh nghiệp",
   "Tầng “đọc” trong pipeline truy hồi rồi đọc (retrieve-then-read)"].forEach((t, i) => {
    s.addText("·  " + t, { x: M + 0.34, y: 4.58 + i * 0.33, w: 5.6, h: 0.3, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 12, color: C.text, valign: "middle" });
  });

  card(s, 7.5, 1.78, 5.08, 3.95, C.accentSoft);
  label(s, "Bốn khó khăn riêng của tiếng Việt", 7.84, 2.05, 4.5, C.accentDeep);
  const hard = [
    ["Đơn vị chữ viết là âm tiết, không phải từ", "“Hà Nội”, “thủ đô” là từ gồm hai âm tiết cách nhau bởi khoảng trắng"],
    ["Dấu thanh mang nghĩa", "“hoà” ≠ “hoa”; một bước chuẩn hoá làm mất dấu là sai EM mà không báo lỗi"],
    ["Không có mạo từ", "quy tắc bỏ a/an/the của SQuAD không áp được: “các”, “những” có thể thuộc đáp án"],
    ["Khoảng một phần ba câu hỏi không có đáp án", "hệ thống phải biết TỪ CHỐI, không chỉ biết tìm"],
  ];
  hard.forEach(([k, v], i) => {
    const y = 2.38 + i * 0.84;
    s.addText(String(i + 1), { x: 7.84, y, w: 0.3, h: 0.3, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 14, bold: true, color: C.accent, valign: "middle" });
    body(s, k, 8.16, y, 4.2, 0.3, { size: 12.5, bold: true, color: C.accentDeep, valign: "middle" });
    body(s, v, 8.16, y + 0.3, 4.2, 0.5, { size: 10.5, color: C.accentDeep });
  });

  keybox(s, "Hướng dẫn môn học: “Kết quả cao trên benchmark không đồng nghĩa với khả năng triển khai thực tế.”",
         "Nên mục tiêu của đồ án không phải điểm cao nhất, mà: mọi con số được trình bày đều đúng, truy vết được, và được diễn giải đúng phạm vi thực nghiệm.",
         5.9, 0.9, "accent");

  footer(s, "1 · Tổng quan · Bối cảnh & động lực", 2);
  s.addNotes("[0:30–1:15] Vì sao bài toán này quan trọng: công cụ tìm kiếm trả về TÀI LIỆU, còn người dùng vẫn phải tự đọc để tìm CÂU TRẢ LỜI. Đọc hiểu máy làm nửa còn lại, và nó là tầng “đọc” trong mọi pipeline hỏi đáp trên tài liệu hiện nay. Với tiếng Việt có bốn khó khăn mà công cụ thiết kế cho tiếng Anh không tự xử lý đúng: đơn vị chữ viết là âm tiết chứ không phải từ; dấu thanh mang nghĩa; không có mạo từ nên quy tắc chuẩn hoá của SQuAD không áp được; và khoảng một phần ba câu hỏi không có đáp án. Điều cuối cùng định hình cả cách nhóm em đánh giá.");
}

// ══════════════════════════════════════════════════════════════════════════
// 3 — 1.2 Phát biểu bài toán: Input → Output
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "1 · Tổng quan");
  title(s, "Phát biểu bài toán — trích xuất, không phải sinh");

  card(s, M, 1.78, 6.55, 1.5);
  label(s, "Định nghĩa hình thức", M + 0.32, 1.98, 5.0, C.accent);
  body(s, "Cho đoạn văn C là chuỗi ký tự và câu hỏi Q, tìm cặp chỉ số ký tự (s, e):",
       M + 0.32, 2.24, 5.9, 0.3, { size: 12, color: C.muted });
  mono(s, "0 ≤ s < e ≤ |C|      sao cho      C[s:e]  trả lời  Q", M + 0.32, 2.56, 5.9, 0.32, { size: 13 });
  body(s, "hoặc trả về chuỗi rỗng khi C không chứa câu trả lời.",
       M + 0.32, 2.92, 5.9, 0.3, { size: 12, italic: true, color: C.muted });

  card(s, M, 3.4, 6.55, 2.35);
  label(s, "Ví dụ", M + 0.3, 3.6, 5.9);
  s.addText([
    { text: "Hà Nội là thủ đô của Việt Nam. Hà Nội được UNESCO công nhận là Thành phố vì hoà bình vào ", options: { color: C.text } },
    { text: "năm 1999", options: { color: C.accentDeep, bold: true, highlight: C.accentSoft } },
    { text: ".", options: { color: C.text } },
  ], { x: M + 0.3, y: 3.86, w: 5.9, h: 0.75, isTextBox: true, margin: 0,
       fontFace: BODY, fontSize: 13, lineSpacing: 19 });
  label(s, "Câu hỏi", M + 0.3, 4.66, 5.9);
  body(s, "Hà Nội được UNESCO công nhận vào năm nào?", M + 0.3, 4.9, 5.9, 0.3, { size: 13 });
  s.addShape(pres.ShapeType.roundRect, {
    x: M + 0.3, y: 5.22, w: 2.5, h: 0.45, fill: { color: C.sageSoft },
    line: { color: C.sageSoft, width: 0 }, rectRadius: 0.22 });
  s.addText([
    { text: "Đáp án:  ", options: { bold: true, color: C.sageDeep } },
    { text: "năm 1999", options: { color: C.sageDeep } },
  ], { x: M + 0.45, y: 5.22, w: 2.3, h: 0.45, isTextBox: true, margin: 0,
       fontFace: BODY, fontSize: 13, valign: "middle" });
  body(s, "Mô hình không sinh chữ mới — nó chỉ ra vị trí.", 3.6, 5.24, 3.6, 0.4,
       { size: 12.5, bold: true, color: C.accentDeep, valign: "middle" });

  card(s, 7.5, 1.78, 5.08, 3.97, C.accentSoft);
  label(s, "Đặc tả đầu vào / đầu ra", 7.82, 1.98, 4.5, C.accentDeep);
  const io = [
    ["Đầu vào", "context: đoạn văn Wikipedia tiếng Việt · question: câu hỏi tự nhiên"],
    ["Đầu ra", "answer: một chuỗi CON của context, hoặc chuỗi rỗng khi không có đáp án"],
    ["Loại bài toán", "trích xuất (extractive) — không sinh văn bản"],
    ["Bộ dữ liệu", "UIT-ViQuAD 2.0, định dạng SQuAD-2.0"],
    ["Thang đo", "Exact Match và F1 trên token (âm tiết)"],
  ];
  io.forEach(([k, v], i) => {
    const y = 2.3 + i * 0.6;
    label(s, k, 7.82, y, 4.5, C.accent);
    body(s, v, 7.82, y + 0.22, 4.5, 0.4, { size: 11, color: C.accentDeep });
  });
  mono(s, "predict(ctx, q)  ⊂  ctx", 7.82, 5.3, 4.5, 0.34, { size: 14 });

  keybox(s, "Tính trích xuất không phải chi tiết phụ — nó cho một bất biến kiểm được bằng máy.",
         "Với mọi mô hình và mọi đầu vào, đầu ra phải là chuỗi con của context. Bất biến này được test tự động và bắt được cả lỗi lệch offset của tokenizer lẫn trường hợp mô hình “bịa” chữ không có trong văn bản.",
         5.9, 0.9);

  footer(s, "1 · Tổng quan · Phát biểu bài toán", 3);
  s.addNotes("[1:15–2:05] Phát biểu hình thức: cho một đoạn văn và một câu hỏi, tìm cặp chỉ số ký tự sao cho đoạn con đó trả lời câu hỏi — hoặc trả về chuỗi rỗng nếu đoạn văn không chứa câu trả lời. Đầu ra là một CẶP VỊ TRÍ, không phải văn bản sinh mới. Điều đó cho nhóm em một bất biến kiểm được bằng máy: kết quả trả về luôn phải là chuỗi con của context. Bọn em test bất biến này cho mọi mô hình, và nó bắt được cả lỗi tokenizer lệch offset lẫn lỗi mô hình bịa đáp án.");
}

// ══════════════════════════════════════════════════════════════════════════
// 4 — 1.3 Đóng góp của nhóm
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "1 · Tổng quan");
  title(s, "Đóng góp của nhóm");

  const contrib = [
    ["Pipeline MRC tiếng Việt hoàn chỉnh",
     "Từ nạp dữ liệu đến ứng dụng web, tổ chức thành module có kiểm thử — " + DATA.tests.full + " test, " + DATA.tests.files + " tệp.", C.accent],
    ["So sánh có kiểm soát bốn mô hình",
     "Mỗi bước cô lập đúng một biến: so khớp từ khoá → chuyển giao đa ngữ zero-shot → fine-tune trong miền → pretraining chuyên tiếng Việt.", C.sage],
    ["Tách answerable / impossible",
     "Cho thấy điểm tổng che khuất hai kỹ năng tách biệt — tìm đúng biên đáp án, và biết khi nào không nên trả lời.", C.accent],
    ["Chặn bốn lỗi “sai âm thầm”",
     "Test split không có nhãn · giới hạn số cửa sổ của transformers · mẫu đánh giá thiên lệch chủ đề · ngưỡng độ dài đáp án phụ thuộc tokenizer.", C.sage],
  ];
  contrib.forEach(([name, desc, col], i) => {
    const x = M + (i % 2) * 6.05, y = 1.82 + Math.floor(i / 2) * 1.98;
    card(s, x, y, 5.78, 1.74);
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.32, y: y + 0.3, w: 0.36, h: 0.36,
      fill: { color: col }, line: { color: col, width: 0 } });
    s.addText(String(i + 1), { x: x + 0.32, y: y + 0.3, w: 0.36, h: 0.36, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 14, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    body(s, name, x + 0.82, y + 0.28, 4.6, 0.4, { size: 16.5, bold: true, valign: "middle" });
    body(s, desc, x + 0.32, y + 0.82, 5.15, 0.8, { size: 12, color: C.muted });
  });

  keybox(s, "Và một đóng góp về phương pháp: chẩn đoán một ca lỗi thay vì che nó đi.",
         "Thất bại của ViSoBERT được phân tích theo quy trình loại trừ, gắn với bằng chứng đo được về mức độ phân mảnh token — trình bày ở mục 5.",
         5.9, 0.9);

  footer(s, "1 · Tổng quan · Đóng góp", 4);
  s.addNotes("[2:05–2:45] Bốn đóng góp. Một: một pipeline hoàn chỉnh có kiểm thử, không phải một notebook. Hai: so sánh có kiểm soát bốn mô hình, mỗi bước cô lập đúng một biến. Ba: tách kết quả theo câu có đáp án và câu không có đáp án — đây là đóng góp phân tích quan trọng nhất, em sẽ chứng minh ở mục 5. Bốn: phát hiện và chặn bốn lỗi sai âm thầm, tức là loại lỗi làm kết quả sai mà không báo lỗi gì. Ngoài ra nhóm em giữ nguyên phần chẩn đoán một lần huấn luyện thất bại trong báo cáo, thay vì bỏ nó ra cho bảng số đẹp hơn.");
}

// ══════════════════════════════════════════════════════════════════════════
// 5 — 2.1/2.2 Hướng tiếp cận hiện tại + các bộ dữ liệu đã có
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "2 · Các công trình liên quan");
  title(s, "Bốn hướng tiếp cận và các bộ dữ liệu đã có");

  card(s, M, 1.78, 6.1, 4.05);
  label(s, "Khảo sát hướng tiếp cận", M + 0.32, 1.98, 5.4, C.accent);
  const approaches = [
    ["So khớp từ khoá / truy hồi", "TF-IDF + cosine trên câu — không huấn luyện, dùng làm sàn", C.muted],
    ["Chuyển giao đa ngữ zero-shot", "encoder đa ngữ đã fine-tune QA tiếng Anh, áp thẳng sang tiếng Việt (XLM-R squad2)", C.muted],
    ["Fine-tune trong miền, encoder đa ngữ", "mBERT + lớp QA, huấn luyện trên chính dữ liệu tiếng Việt", C.accent],
    ["PLM đơn ngữ tiếng Việt", "PhoBERT (2020) và ViSoBERT (2023) — hướng được kỳ vọng nhất cho tiếng Việt", C.sage],
  ];
  approaches.forEach(([name, desc, col], i) => {
    const y = 2.3 + i * 0.86;
    s.addShape(pres.ShapeType.ellipse, { x: M + 0.34, y: y + 0.1, w: 0.18, h: 0.18,
      fill: { color: col }, line: { color: col, width: 0 } });
    body(s, name, M + 0.66, y, 5.1, 0.3, { size: 13.5, bold: true, valign: "middle" });
    body(s, desc, M + 0.66, y + 0.3, 5.1, 0.5, { size: 11, color: C.muted });
  });

  card(s, 7.1, 1.78, 5.48, 4.05, C.surface, false);
  label(s, "Các bộ dữ liệu đã có", 7.42, 1.98, 4.8, C.accentDeep);
  table(s, ["Bộ dữ liệu", "Quy mô", "Không đáp án"], [
    ["SQuAD 1.1 · 2016", "> 100.000 câu (EN)", "không"],
    ["SQuAD 2.0 · 2018", "+ > 50.000 câu", { t: "có", b: true, c: C.sageDeep }],
    ["UIT-ViQuAD · 2020", "> 23.000 câu · 174 bài (VI)", "không"],
    [{ t: "UIT-ViQuAD 2.0 · 2021", b: true }, { t: DATA.split.questions.join(" / "), b: true }, { t: "có", b: true, c: C.sageDeep }],
  ], 7.42, 2.32, 4.8, [2.1, 1.85, 0.85], { size: 11, rowH: 0.42 });
  body(s, "Dòng cuối là số đo trực tiếp từ tệp tải về (train / validation / test) — không chép từ tài liệu; chi tiết ở mục 3.",
       7.42, 4.84, 4.8, 0.45, { size: 10.5, italic: true, color: C.muted });
  body(s, "UIT-ViQuAD 2.0 là bộ dữ liệu tiếng Việt duy nhất trong nhóm này có câu hỏi không có đáp án — nên nó là lựa chọn bắt buộc nếu muốn đo kỹ năng từ chối.",
       7.42, 5.3, 4.8, 0.5, { size: 10.5, color: C.text });

  keybox(s, "Bốn mô hình của đồ án chính là bốn hướng tiếp cận trên — đo trên cùng một tập, cùng một interface.",
         "Nhờ vậy chênh lệch giữa hai dòng bất kỳ trong bảng kết quả đọc được thành “biến nào đã thay đổi”, thay vì chỉ là hai con số cạnh nhau.",
         5.95, 0.85);

  footer(s, "2 · Các công trình liên quan · Khảo sát", 5);
  s.addNotes("[2:45–3:35] Về hướng tiếp cận, tài liệu hiện có chia thành bốn nhóm: so khớp từ khoá thuần, chuyển giao đa ngữ zero-shot, fine-tune trong miền trên encoder đa ngữ, và mô hình tiền huấn luyện đơn ngữ tiếng Việt. Về dữ liệu: SQuAD 1.1 mở đầu, SQuAD 2.0 thêm câu không có đáp án, UIT-ViQuAD của trường mình mang bài toán sang tiếng Việt, và UIT-ViQuAD 2.0 ở VLSP 2021 thêm câu không có đáp án. Điểm cần nhớ: trong nhóm này, ViQuAD 2.0 là bộ tiếng Việt duy nhất cho phép đo kỹ năng TỪ CHỐI. Bốn mô hình của nhóm em đúng là bốn hướng tiếp cận đó, đặt trên cùng một tập đánh giá và cùng một interface.");
}

// ══════════════════════════════════════════════════════════════════════════
// 6 — 2.3 GAP
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "2 · Các công trình liên quan");
  title(s, "GAP — chỗ nhóm đứng vào");

  const gaps = [
    ["Điểm tổng che hai kỹ năng khác nhau",
     "Công bố theo bảng xếp hạng thường báo EM/F1 TỔNG. Với khoảng 30% câu không có đáp án, số tổng trộn “tìm đúng span” với “biết từ chối”.",
     "Nhóm tách answerable_only / impossible_only cho cả bốn mô hình, kèm khoảng tin cậy."],
    ["Test split công khai đã bị lược nhãn",
     "Cả " + DATA.split.questions[2] + " câu test có gold rỗng nhưng is_impossible = False. Ai chấm trên đó sẽ thấy một mô hình luôn trả rỗng đạt EM 100%.",
     "Nhóm đo, nói rõ, và chặn bằng assert_gradeable() trước mọi lần chấm."],
    ["Ngân sách cửa sổ phụ thuộc tokenizer",
     "Khi đổi sang encoder tiếng Việt, max_length và độ dài đáp án tối đa thường được giữ nguyên theo tokenizer cũ — một biến gây nhiễu ít được nói tới.",
     "Nhóm đo mức phân mảnh token và coi đó là confound, không phải kết luận về encoder."],
  ];
  gaps.forEach(([k, why, ours], i) => {
    const y = 1.8 + i * 1.38;
    card(s, M, y, CW, 1.26);
    s.addShape(pres.ShapeType.roundRect, { x: M + 0.28, y: y + 0.42, w: 0.4, h: 0.4,
      fill: { color: C.accent }, line: { color: C.accent, width: 0 }, rectRadius: 0.1 });
    s.addText(String(i + 1), { x: M + 0.28, y: y + 0.42, w: 0.4, h: 0.4, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 15, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    body(s, k, M + 0.85, y + 0.16, 4.3, 0.6, { size: 14, bold: true });
    body(s, why, M + 0.85, y + 0.68, 4.3, 0.5, { size: 10.5, color: C.muted });
    label(s, "Nhóm làm gì", M + 5.55, y + 0.18, 2.4, C.accent);
    body(s, ours, M + 5.55, y + 0.45, 5.95, 0.7, { size: 12, color: C.text });
  });

  keybox(s, "GAP nhóm đứng vào là tính ĐÚNG ĐẮN của đánh giá, không phải điểm số.",
         "Đề tài không yêu cầu vượt bảng xếp hạng. Giá trị của đồ án nằm ở chỗ mỗi con số chịu được chất vấn: biết nó đo cái gì, trên bao nhiêu mẫu, với độ bất định bao nhiêu.",
         5.95, 0.85, "accent");

  footer(s, "2 · Các công trình liên quan · GAP", 6);
  s.addNotes("[3:35–4:20] Đây là chỗ nhóm em đứng vào. Ba khoảng trống. Một: các công bố theo bảng xếp hạng báo điểm TỔNG, mà điểm tổng trộn hai kỹ năng khác nhau — nhóm em tách ra. Hai: test split công khai đã bị lược nhãn, nên không ai bên ngoài cuộc thi chấm lại được trên đó; nhóm em đo, nói rõ và chặn bằng assertion. Ba: ngân sách cửa sổ và độ dài đáp án tối đa phụ thuộc tokenizer, nhưng khi đổi encoder người ta hay giữ nguyên — nhóm em đo mức phân mảnh và coi đó là biến gây nhiễu. Nói thẳng: GAP nhóm em chọn là GAP về tính đúng đắn của đánh giá, không phải GAP về điểm số.");
}

// ══════════════════════════════════════════════════════════════════════════
// 7 — 3.1 Nguồn dữ liệu, thống kê split, và tập test ẩn nhãn
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "3 · Xây dựng dữ liệu");
  title(s, "UIT-ViQuAD 2.0 — thống kê đo trực tiếp từ tệp tải về");

  card(s, M, 1.82, 6.9, 2.9);
  label(s, "taidng/UIT-ViQuAD2.0 → ba tệp JSON định dạng SQuAD-2.0", M + 0.3, 2.04, 6.3);
  table(s, ["Split", "Câu hỏi", "Context", "Article", "Impossible", "Chấm được"], [
    ["train", DATA.split.questions[0], DATA.split.contexts[0], DATA.split.articles[0], DATA.split.impossible[0], DATA.split.gradeable[0]],
    ["validation", DATA.split.questions[1], DATA.split.contexts[1], DATA.split.articles[1], DATA.split.impossible[1], DATA.split.gradeable[1]],
    ["test", DATA.split.questions[2], DATA.split.contexts[2], DATA.split.articles[2], DATA.split.impossible[2], { t: DATA.split.gradeable[2], b: true, c: C.accentDeep }],
  ], M + 0.3, 2.4, 6.3, [1.42, 0.92, 0.88, 0.83, 1.05, 1.2], { size: 12, rowH: 0.42 });
  body(s, "Đơn vị dedup và chống rò rỉ là CONTEXT, không phải ARTICLE: train có " + DATA.split.articles[0] +
          " bài viết nhưng " + DATA.split.contexts[0] + " đoạn văn — nhầm hai đơn vị này làm vô hiệu hoá chính cơ chế bảo vệ.",
       M + 0.3, 4.0, 6.3, 0.6, { size: 11, color: C.muted });

  card(s, 7.9, 1.82, 4.68, 2.9, C.accentSoft);
  label(s, "Phát hiện: test split là tập ẩn nhãn", 8.2, 2.04, 4.1, C.accentDeep);
  body(s, DATA.split.questions[2] + " câu test đều có gold rỗng, đồng thời is_impossible = False.",
       8.2, 2.34, 4.1, 0.6, { size: 13.5, bold: true, color: C.accentDeep });
  body(s, "Hai điều đó mâu thuẫn nhau nếu coi là nhãn thật ⇒ đáp án đã bị lược bỏ để dùng cho bảng xếp hạng.",
       8.2, 3.0, 4.1, 0.8, { size: 12, color: C.accentDeep });
  body(s, "Chấm nhầm trên tập này → EM 100%.", 8.2, 3.86, 4.1, 0.35, { size: 13, bold: true, color: C.accentDeep });
  body(s, "⇒ Mọi kết quả của đồ án đo trên validation.", 8.2, 4.24, 4.1, 0.35, { size: 12, italic: true, color: C.accentDeep });

  card(s, M, 4.9, CW, 1.0, C.surface, false);
  body(s, "Chặn bằng code, không bằng lời hứa", M + 0.35, 5.06, 4.6, 0.32, { size: 14.5, bold: true });
  mono(s, "assert_gradeable(examples)", M + 0.35, 5.4, 4.6, 0.32, { size: 13 });
  const notes3 = [
    ["0", "đoạn văn chung giữa train và validation — assert_no_leakage() xác nhận trước mỗi lần fine-tune"],
    [DATA.sharedCtx.trainTest + " / " + DATA.sharedCtx.valTest, "đoạn văn test chung với train / với validation — cảnh báo cho ai định tự gán nhãn lại test"],
  ];
  notes3.forEach(([k, v], i) => {
    const x = 5.6 + i * 3.55;
    s.addText(k, { x, y: 5.02, w: 1.2, h: 0.4, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 20, bold: true, color: C.accent, valign: "middle" });
    body(s, v, x + 1.25, 5.04, 2.25, 0.8, { size: 10, color: C.muted });
  });

  keybox(s, "Hệ quả thiết kế: validation là tập đánh giá cuối, và điều đó được nói rõ trong phần hạn chế.",
         "Validation cũng được dùng để chọn epoch (trên mẫu 300 câu), nên hai mô hình fine-tune có thể lạc quan nhẹ so với một tập kiểm tra độc lập hoàn toàn.",
         6.0, 0.8);

  footer(s, "3 · Xây dựng dữ liệu · Nguồn & split", 7);
  s.addNotes("[4:20–5:20] Phát hiện đầu tiên của nhóm em, và nó đổi cả thiết kế đánh giá. Toàn bộ 7.301 câu trong split test có đáp án rỗng nhưng cờ is_impossible lại là False — hai điều này mâu thuẫn nhau, nghĩa là đáp án đã bị lược bỏ để dùng cho bảng xếp hạng. Nếu chạy nhầm trên đó, một mô hình luôn trả rỗng sẽ đạt EM 100% và con số đó trông hoàn toàn hợp lý trong báo cáo. Bọn em chặn bằng một assertion chứ không bằng một đoạn văn hứa là đã cẩn thận. Một chi tiết nữa: đơn vị chống rò rỉ là đoạn văn, không phải bài viết — train có 138 bài viết nhưng 4.101 đoạn văn, nhầm hai đơn vị này là vô hiệu hoá chính cơ chế bảo vệ.");
}

// ══════════════════════════════════════════════════════════════════════════
// 8 — 3.2 Thống kê & phân tích: phân bố nhãn + phân bố độ dài
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "3 · Xây dựng dữ liệu · Thống kê & phân tích");
  title(s, "Phân bố nhãn và phân bố độ dài");

  card(s, M, 1.8, 5.6, 2.6);
  label(s, "Phân bố nhãn", M + 0.32, 2.0, 4.9, C.accent);
  table(s, ["Tập", "Không có đáp án", "Tỉ lệ"], [
    ["train", DATA.split.impossible[0] + " / " + DATA.split.questions[0], DATA.split.impPct[0]],
    ["validation", DATA.split.impossible[1] + " / " + DATA.split.questions[1], DATA.split.impPct[1]],
    ["test", "— (đã lược nhãn)", DATA.split.impPct[2]],
    [{ t: "mẫu đánh giá " + nOf("mbert") + " câu", b: true }, { t: nImp("mbert") + " / " + nOf("mbert"), b: true },
     { t: impShare + "%", b: true, c: C.accentDeep }],
  ], M + 0.32, 2.3, 4.9, [2.0, 1.9, 1.0], { size: 11, rowH: 0.3 });
  body(s, "Điểm tổng là trung bình có trọng số của hai kỹ năng theo đúng tỉ lệ này — mục 5 bóc tách ra.",
       M + 0.32, 4.08, 4.9, 0.35, { size: 10.5, italic: true, color: C.muted });

  card(s, 6.7, 1.8, 5.88, 2.6);
  label(s, "Phân bố độ dài (đơn vị: âm tiết, tách theo khoảng trắng)", 7.02, 2.0, 5.2, C.accent);
  table(s, ["Chỉ số", "train", "validation"], [
    ["Đoạn văn — TB / trung vị / max", DATA.split.ctxWords[0], DATA.split.ctxWords[1]],
    ["Đáp án vàng — TB / trung vị / p95", DATA.split.ansWords[0], DATA.split.ansWords[1]],
    ["Câu hỏi — TB", "14,6", "14,2"],
    ["Đoạn văn dài hơn 300 từ", DATA.longCtx.n + " / " + DATA.longCtx.of + " · " + DATA.longCtx.pct, "—"],
  ], 7.02, 2.3, 5.2, [2.3, 1.45, 1.45], { size: 10.5, rowH: 0.3 });
  body(s, "Đáp án có trung vị 6 âm tiết nhưng p95 = 31: đuôi dài, và đuôi này quyết định ngưỡng độ dài đáp án theo token.",
       7.02, 4.08, 5.2, 0.35, { size: 10.5, italic: true, color: C.muted });

  label(s, "Mẫu đánh giá " + nOf("mbert") + " câu theo nhóm độ dài đoạn văn — nguồn: results/eval_*.json", M, 4.52, 8.5, C.accent);
  [["<100", "< 100 từ"], ["100-200", "100 – 200 từ"], ["200-300", "200 – 300 từ"], ["300+", "300+ từ"]]
    .forEach(([bucket, cap], i) => {
      const g = bylen("mbert", bucket);
      stat(s, String(g.count), "câu", cap + (g.unreliable ? " · n < 30 ⇒ cờ unreliable" : ""),
           M + i * 3.0, 4.82, 2.8, g.unreliable ? C.muted : C.accent, 26);
    });

  keybox(s, "Hai hệ quả đọc thẳng ra từ phân bố, không phải suy diễn.",
         "Một: mẫu " + nOf("mbert") + " câu gần như không có đoạn dài (" + bylen("mbert", "300+").count +
         " câu ở nhóm 300+), nên câu hỏi “độ dài context ảnh hưởng thế nào” CHƯA được trả lời — nhóm ghi vào phần hạn chế. Hai: đuôi dài của đáp án là lý do L_max phải đặt theo tokenizer.",
         5.88, 0.92);

  footer(s, "3 · Xây dựng dữ liệu · Phân bố nhãn & độ dài", 8);
  s.addNotes("[5:20–6:10] Ba phân bố, bắt đầu bằng nhãn và độ dài. Về nhãn: khoảng 30 đến 32 phần trăm câu là không có đáp án, và mẫu đánh giá 500 câu của bọn em có 139 câu, tức 27,8 phần trăm — con số này sẽ quay lại ở mục 5 theo một cách rất đáng nói. Về độ dài: đoạn văn tập trung quanh 160 đến 180 âm tiết, đáp án có trung vị 6 âm tiết nhưng phân vị 95 là 31 — đuôi dài. Hai hệ quả. Một: mẫu 500 câu chỉ có 9 câu thuộc nhóm đoạn văn dài trên 300 từ, nên câu hỏi “độ dài context ảnh hưởng thế nào” bọn em KHÔNG kết luận, mà ghi vào phần hạn chế. Hai: vì đáp án có đuôi dài, ngưỡng độ dài đáp án tối đa phải đặt theo tokenizer của từng mô hình.");
}

// ══════════════════════════════════════════════════════════════════════════
// 9 — 3.3 Phân bố chủ đề (và cái bẫy lấy mẫu)
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "3 · Xây dựng dữ liệu · Thống kê & phân tích");
  title(s, "Phân bố chủ đề — và cái bẫy lấy mẫu nó gây ra");

  card(s, M, 1.78, 6.1, 3.05);
  label(s, "Validation: " + DATA.split.articles[1] + " bài viết · " + DATA.split.questions[1] + " câu hỏi",
        M + 0.32, 1.96, 5.4, C.accent);
  table(s, ["Bài viết (top 5)", "Số câu", "%"], [
    [DATA.topics[0][0], "503", "13,2"],
    [DATA.topics[1][0], "462", "12,1"],
    [DATA.topics[2][0], "379", "9,9"],
    [DATA.topics[3][0], "356", "9,3"],
    [DATA.topics[4][0], "324", "8,5"],
  ], M + 0.32, 2.26, 5.4, [3.2, 1.2, 1.0], { size: 11, rowH: 0.28 });
  body(s, "5 bài lớn nhất chiếm " + DATA.topicTop5Pct + " · đuôi rất mỏng: Montréal 42 · Kiến 57 · Gia cầm 59 câu. Chỉ 3 trong " +
          DATA.split.articles[1] + " bài viết thuộc khoa học tự nhiên (" + DATA.topicSciN + " câu, " +
          DATA.topicSciPct + ") — còn lại là lịch sử, chính trị, địa lý.",
       M + 0.32, 4.26, 5.4, 0.5, { size: 10.5, color: C.muted });

  card(s, 7.1, 1.78, 5.48, 3.05, C.accentSoft);
  label(s, "Cái bẫy: câu hỏi trong tệp được sắp theo bài viết", 7.42, 1.96, 4.8, C.accentDeep);
  body(s, DATA.firstN + " câu đầu tệp validation thuộc DUY NHẤT một bài viết: “Paris”.",
       7.42, 2.24, 4.8, 0.55, { size: 14, bold: true, color: C.accentDeep });
  s.addText([
    { text: DATA.biasEM.head, options: { fontFace: HEAD, fontSize: 30, bold: true, color: C.accentDeep } },
    { text: "   so với   ", options: { fontFace: BODY, fontSize: 12, color: C.accentDeep } },
    { text: fmt(CURVE.mbert.curve[1].val_em), options: { fontFace: HEAD, fontSize: 30, bold: true, color: C.sageDeep } },
  ], { x: 7.42, y: 2.86, w: 4.8, h: 0.55, isTextBox: true, margin: 0, valign: "middle" });
  body(s, "EM của cùng một checkpoint mBERT (epoch 2): " + DATA.firstN + " câu đầu tệp so với " +
          DATA.firstN + " câu ngẫu nhiên seed 42. Chênh 10 điểm chỉ do CÁCH LẤY MẪU, không do mô hình.",
       7.42, 3.48, 4.8, 0.7, { size: 11.5, color: C.accentDeep });
  mono(s, "reproducible_subset(examples, n, seed=42)", 7.42, 4.3, 4.8, 0.3, { size: 11.5 });

  card(s, M, 4.95, CW, 0.8, C.surface, false);
  label(s, "Phân bố từ để hỏi (phân nhóm từ khoá thô, chỉ để mô tả)", M + 0.35, 5.04, 7.5);
  DATA.wh.forEach(([k, v], i) => {
    const x = M + 0.35 + i * 1.86;
    s.addText(v, { x, y: 5.28, w: 1.8, h: 0.28, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 15, bold: true, color: C.accent, valign: "middle" });
    s.addText(k, { x, y: 5.52, w: 1.8, h: 0.22, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 10, color: C.muted, valign: "middle" });
  });

  keybox(s, "Vì phân bố chủ đề lệch, mẫu đánh giá phải ngẫu nhiên có seed — và mẫu đó được kiểm lại.",
         "Mẫu " + nOf("mbert") + " câu phủ " + DATA.subsetArticles + " bài viết, 5 bài lớn nhất chiếm " +
         DATA.subsetTop5Pct + " so với " + DATA.topicTop5Pct + " của toàn tập: mẫu đại diện được về chủ đề. Cùng hàm lấy mẫu dùng cho cả đường cong huấn luyện và bảng kết quả cuối, nên hai nguồn số liệu so sánh được với nhau.",
         5.85, 0.95);

  footer(s, "3 · Xây dựng dữ liệu · Phân bố chủ đề", 9);
  s.addNotes("[6:10–7:10] Phân bố thứ ba: chủ đề. Validation chỉ có 19 bài viết cho 3.814 câu hỏi, và năm bài lớn nhất chiếm hơn một nửa. Đuôi thì rất mỏng — Montréal 42 câu, Kiến 57 câu. Chỉ ba bài thuộc khoa học tự nhiên, còn lại là lịch sử, chính trị, địa lý; nên mọi kết luận của bọn em chỉ có giá trị trong miền đó. Quan trọng hơn, phân bố lệch này gây ra một cái bẫy thật: câu hỏi trong tệp được sắp theo bài viết, nên 300 câu đầu tệp đều thuộc DUY NHẤT bài Paris. Cùng một checkpoint mBERT cho EM 42 trên 300 câu đầu tệp nhưng EM 52 trên 300 câu ngẫu nhiên — chênh 10 điểm chỉ vì cách lấy mẫu. Nếu lấy mẫu bằng cách cắt đầu tệp thì toàn bộ so sánh bốn mô hình đã sai từ gốc. Bọn em dùng lấy mẫu ngẫu nhiên có seed, và kiểm lại rằng mẫu 500 câu phủ đủ 19 trên 19 bài viết.");
}

// ══════════════════════════════════════════════════════════════════════════
// 10 — 4.1 Bốn hệ thống, bốn câu hỏi
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "4 · Phương pháp & thực nghiệm");
  title(s, "Bốn hệ thống — mỗi cái trả lời một câu hỏi khác nhau");

  const items = [
    ["TF-IDF Baseline", "Bài toán này giải được bằng so khớp từ khoá thuần không?", "không huấn luyện", C.muted],
    ["XLM-R squad2", "Transfer từ SQuAD-2.0 tiếng Anh sang tiếng Việt được bao nhiêu?", "zero-shot", C.muted],
    ["mBERT + QA", "Fine-tune trong miền thêm được bao nhiêu, với encoder đa ngữ?", "fine-tuned · MPS", C.accent],
    ["ViSoBERT + QA", "Pretraining chuyên tiếng Việt đóng góp bao nhiêu?", "fine-tuned · MPS", C.sage],
  ];
  items.forEach(([name, q, tag, col], i) => {
    const x = M + (i % 2) * 6.05, y = 1.85 + Math.floor(i / 2) * 1.95;
    card(s, x, y, 5.78, 1.7);
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.32, y: y + 0.34, w: 0.34, h: 0.34, fill: { color: col }, line: { color: col, width: 0 } });
    s.addText(String(i + 1), { x: x + 0.32, y: y + 0.34, w: 0.34, h: 0.34, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 13, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    s.addText(name, { x: x + 0.82, y: y + 0.28, w: 3.0, h: 0.4, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 19, bold: true, color: C.text, valign: "middle" });
    s.addShape(pres.ShapeType.roundRect, { x: x + 3.95, y: y + 0.34, w: 1.55, h: 0.34,
      fill: { color: C.surface }, line: { color: C.surface, width: 0 }, rectRadius: 0.17 });
    s.addText(tag, { x: x + 3.95, y: y + 0.34, w: 1.55, h: 0.34, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 9.5, color: C.muted, align: "center", valign: "middle" });
    body(s, q, x + 0.32, y + 0.88, 5.15, 0.7, { size: 13, italic: true, color: C.muted });
  });

  keybox(s, "Giả thuyết được đăng ký TRƯỚC khi chạy (results/hypotheses.md) — và nó viết cho PhoBERT.",
         "Giả thuyết dự đoán encoder tiếng Việt sẽ vượt mBERT. PhoBERT không chạy được vì lý do kỹ thuật (slide sau), nên ViSoBERT là mô hình THAY THẾ, không phải phép kiểm cho giả thuyết đó: giả thuyết trung tâm vẫn CHƯA kiểm được, không phải bị bác bỏ.",
         5.85, 0.95, "accent");

  footer(s, "4 · Phương pháp · Bốn mô hình", 10);
  s.addNotes("[7:10–7:55] Bốn hệ thống không phải bốn lần thử cho vui — mỗi cái trả lời một câu hỏi khoa học khác nhau. TF-IDF là sàn: nếu một mô hình phức tạp không vượt được nó thì mô hình đó vô nghĩa. XLM-R đo khả năng chuyển giao từ tiếng Anh. mBERT đo phần fine-tune trong miền. ViSoBERT đo phần pretraining chuyên tiếng Việt. Một điểm em xin nói rõ ngay: giả thuyết ghi trước khi chạy được viết cho PhoBERT, mà PhoBERT thì không chạy được vì lý do kỹ thuật em trình bày ở slide sau. Nên ViSoBERT là mô hình thay thế, và giả thuyết trung tâm của đề tài vẫn chưa kiểm được — chứ không phải đã bị bác bỏ.");
}

// ══════════════════════════════════════════════════════════════════════════
// 11 — 4.2 Kiến trúc bốn tầng
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "4 · Phương pháp · Kiến trúc");
  title(s, "Bốn tầng, một chiều phụ thuộc");

  const layers = [
    ["Tầng dữ liệu", "data.py · tagging.py", "parse SQuAD-2.0 → Example · dedup & split theo CONTEXT · assert_no_leakage · assert_gradeable", C.sage, C.sageSoft, C.sageDeep],
    ["Tầng mô hình", "predictor.py · baseline_tfidf.py · transformer_qa.py · windowing.py · features.py · training.py", "một interface duy nhất: Predictor.predict(ctx, q) -> str", C.accent, C.accentSoft, C.accentDeep],
    ["Tầng đánh giá", "normalize.py · metrics.py · evaluate.py", "EM + token-F1 · tách answerable / impossible · provenance · cờ unreliable khi n < 30", C.sage, C.sageSoft, C.sageDeep],
    ["Tầng trình bày", "src/demo/ (hàm thuần) → app/ (chỉ gọi widget) · make_figures.py · make_report.py", "CHỈ ĐỌC results/ — không bao giờ sinh ra con số", C.accent, C.accentSoft, C.accentDeep],
  ];
  layers.forEach(([name, mods, desc, dot, fill, deep], i) => {
    const y = 1.78 + i * 1.14;
    card(s, M, y, CW, 0.98, fill, false);
    s.addShape(pres.ShapeType.ellipse, { x: M + 0.32, y: y + 0.4, w: 0.2, h: 0.2, fill: { color: dot }, line: { color: dot, width: 0 } });
    s.addText(name, { x: M + 0.66, y: y + 0.14, w: 2.3, h: 0.36, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 16, bold: true, color: deep, valign: "middle" });
    s.addText(mods, { x: M + 0.66, y: y + 0.46, w: 4.7, h: 0.42, isTextBox: true, margin: 0,
      fontFace: "Courier New", fontSize: 9, color: C.muted, lineSpacing: 11 });
    body(s, desc, M + 5.6, y + 0.2, 5.9, 0.7, { size: 12, color: deep });
    if (i < 3) s.addText("▼", { x: 6.5, y: y + 0.99, w: 0.4, h: 0.15, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 9, color: C.line, align: "center", valign: "middle" });
  });

  mono(s, "app/  →  src/demo/  →  src/mrc/  →  results/, data/raw/     ·     một mũi tên, một chiều, không ngoại lệ",
       M, 6.34, CW, 0.32, { size: 12, align: "center" });

  footer(s, "4 · Phương pháp · Kiến trúc (docs/ARCHITECTURE.md)", 11);
  s.addNotes("[7:55–8:45] Bốn tầng, và điều đáng nói là quy tắc phụ thuộc: một mũi tên, một chiều. src/mrc không biết demo tồn tại. src/demo không import Streamlit. app/ không chứa logic. Và tầng trình bày chỉ ĐỌC results — nó không bao giờ sinh ra con số nào. Hệ quả đo được của thiết kế này: chín trên mười hai module trong src/mrc không phụ thuộc torch, nên bộ test nhanh " + DATA.tests.fast + " test chạy trong khoảng một giây, không tải mô hình, không cần mạng.");
}

// ══════════════════════════════════════════════════════════════════════════
// 12 — 4.3 Bốn bất biến + cơ chế thực thi
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "4 · Phương pháp · Bất biến");
  title(s, "Bốn bất biến — và cơ chế thực thi bằng máy");

  const inv = [
    ["1", "Extractive", "Mọi predict() trả về chuỗi con của context", "Test cho mọi Predictor; decode_span() ném ValueError nếu khoảng không hợp lệ"],
    ["2", "Không leakage", "Một context chỉ thuộc một split", "assert_no_leakage() chạy trong mọi đường split và làm FAIL cả run"],
    ["3", "Truy vết được", "Mọi kết quả mang commit, timestamp, device, split, n", "run_evaluation() sinh các trường này — không đường nào tạo ra kết quả thiếu chúng"],
    ["4", "Có n kèm số", "Nhóm count < 30 tự gắn unreliable: true", "breakdown() gắn cờ; không bảng nào trong báo cáo có số thiếu n"],
  ];
  inv.forEach(([n, name, what, how], i) => {
    const y = 1.78 + i * 1.16;
    card(s, M, y, CW, 1.05);
    s.addShape(pres.ShapeType.roundRect, { x: M + 0.28, y: y + 0.3, w: 0.4, h: 0.4,
      fill: { color: C.accent }, line: { color: C.accent, width: 0 }, rectRadius: 0.1 });
    s.addText(n, { x: M + 0.28, y: y + 0.3, w: 0.4, h: 0.4, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 15, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    s.addText(name, { x: M + 0.85, y: y + 0.15, w: 2.2, h: 0.34, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 15, bold: true, color: C.text, valign: "middle" });
    body(s, what, M + 0.85, y + 0.53, 4.4, 0.4, { size: 11.5, color: C.muted });
    label(s, "Cơ chế thực thi", M + 5.6, y + 0.17, 2.4, C.accent);
    body(s, how, M + 5.6, y + 0.44, 5.95, 0.52, { size: 11.5, color: C.text });
  });

  body(s, "Bất biến thứ năm: demo không sinh ra con số nào — và điều đó do kernel bắt buộc, vì models/, data/, results/ được mount read-only.",
       M, 6.46, CW, 0.32, { size: 12, italic: true, color: C.accentDeep, align: "center" });

  footer(s, "4 · Phương pháp · Bất biến", 12);
  s.addNotes("[8:45–9:35] Điều làm đồ án này khác một bài tập thông thường không phải là bốn bất biến, mà là mỗi bất biến có một cơ chế thực thi BẰNG MÁY thay vì một đoạn văn trong báo cáo. Một quy trình có thể bị bỏ qua lúc gấp — đúng lúc nó cần nhất. Một raise thì không. Và bất biến thứ năm thì thậm chí do kernel bắt buộc: ba thư mục dữ liệu được mount read-only.");
}

// ══════════════════════════════════════════════════════════════════════════
// 13 — 4.4 Thang đo cho tiếng Việt   [slide cắt được nếu thiếu thời gian]
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "4 · Phương pháp · Thang đo");
  title(s, "Thước đo được cài trước vật được đo");

  card(s, M, 1.8, 5.6, 3.1);
  label(s, "Hai thang đo, hai câu hỏi khác nhau", M + 0.32, 2.0, 4.9, C.accent);
  body(s, "EM — đúng 1 điểm khi chuỗi dự đoán sau chuẩn hoá TRÙNG KHÍT một đáp án vàng; lệch một ký tự là 0.",
       M + 0.32, 2.3, 4.9, 0.55, { size: 12 });
  body(s, "F1 — trên TOKEN (âm tiết), dùng Counter (đa tập) thay vì set để token lặp được đếm đúng; cho điểm bán phần.",
       M + 0.32, 2.88, 4.9, 0.55, { size: 12 });
  body(s, "Câu không có đáp án — đúng khi và chỉ khi trả về chuỗi rỗng. Nhiều đáp án vàng: lấy điểm cao nhất.",
       M + 0.32, 3.46, 4.9, 0.55, { size: 12 });
  s.addShape(pres.ShapeType.roundRect, { x: M + 0.32, y: 4.02, w: 4.9, h: 0.8,
    fill: { color: C.accentSoft }, line: { color: C.accentSoft, width: 0 }, rectRadius: 0.14 });
  body(s, "Bằng chứng: TF-IDF có F1 " + f1("tfidf") + " nhưng EM chỉ " + em("tfidf") +
          " — nó trả về cả một câu: trùng token, gần như không bao giờ trùng khít.",
       M + 0.52, 4.12, 4.5, 0.6, { size: 10.5, color: C.accentDeep });

  card(s, 6.7, 1.8, 5.88, 3.1, C.surface, false);
  label(s, "Ba quyết định chuẩn hoá — mỗi cái được test GHIM", 7.02, 2.0, 5.2, C.accentDeep);
  const dec = [
    ["A", "Không loại bỏ mạo từ", "tiếng Việt không có mạo từ; “các”, “những”, “con” là loại từ và có thể thuộc đáp án", "“các tỉnh” giữ nguyên “các”"],
    ["B", "Tách token theo khoảng trắng", "khớp cách đánh giá thông dụng của ViQuAD; không nhiễm lỗi bộ tách từ", "“Hà Nội” là 2 token"],
    ["C", "Giữ dấu tiếng Việt", "“hoà” ≠ “hoa”; bỏ dấu làm sai EM hàng loạt mà không báo lỗi", "EM(“hoà”, “hoa”) = 0"],
  ];
  dec.forEach(([letter, what, why, test], i) => {
    const y = 2.3 + i * 0.86;
    s.addShape(pres.ShapeType.roundRect, { x: 7.02, y, w: 0.34, h: 0.34,
      fill: { color: C.accent }, line: { color: C.accent, width: 0 }, rectRadius: 0.09 });
    s.addText(letter, { x: 7.02, y, w: 0.34, h: 0.34, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 13, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    body(s, what, 7.48, y, 2.6, 0.3, { size: 12.5, bold: true, valign: "middle" });
    body(s, why, 7.48, y + 0.3, 2.6, 0.5, { size: 9.5, color: C.muted });
    mono(s, test, 10.2, y + 0.02, 2.3, 0.3, { size: 10, color: C.sageDeep, valign: "middle" });
  });

  card(s, M, 5.0, CW, 0.82, C.card);
  body(s, "Hai chi tiết cài đặt cũng được test:", M + 0.35, 5.16, 3.1, 0.3, { size: 12, bold: true, valign: "middle" });
  body(s, "dấu câu nhận diện theo nhóm Unicode P* thay vì string.punctuation (bắt được “–”, “”, “…” thường gặp trong Wikipedia)  ·  evaluate() raise KeyError nếu thiếu dự đoán của bất kỳ câu nào — bỏ qua âm thầm sẽ tính EM trên một tập con nhỏ hơn mà không ai biết.",
       M + 3.5, 5.1, 8.0, 0.6, { size: 11, color: C.muted });

  keybox(s, "Metric được cài trước mọi mô hình, và đạt độ phủ kiểm thử 100%.",
         "Thứ tự này có chủ ý: nếu viết mô hình trước rồi mới viết thước đo, mọi lần “sửa metric cho hợp lý” đều là một cơ hội để chỉnh số theo ý mình mà không nhận ra.",
         5.95, 0.85);

  footer(s, "4 · Phương pháp · Thang đo cho tiếng Việt", 13);
  s.addNotes("[9:35–10:20] Về thang đo. EM là trùng khít, F1 là trùng theo token và cho điểm bán phần. Bằng chứng trực quan rằng hai thang đo đo hai thứ khác nhau nằm ngay ở dòng baseline: TF-IDF có F1 hơn 23 nhưng EM chỉ 0,8 — vì nó trả về cả một câu trong khi đáp án vàng là cụm vài từ. Ba quyết định chuẩn hoá cho tiếng Việt: không loại mạo từ vì tiếng Việt không có mạo từ; tách token theo khoảng trắng để không nhiễm lỗi bộ tách từ; và giữ dấu, vì hoà khác hoa. Mỗi quyết định được GHIM bằng một test, để người sau không vô tình “sửa cho giống SQuAD”. Metric được cài trước mọi mô hình và có độ phủ test 100%.");
}

// ══════════════════════════════════════════════════════════════════════════
// 14 — 4.5 Hai quyết định kỹ thuật
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "4 · Phương pháp · Quyết định kỹ thuật");
  title(s, "Hai quyết định kỹ thuật đã đổi cả hệ thống");

  card(s, M, 1.85, 5.78, 4.5);
  label(s, "ADR-003", M + 0.32, 2.1, 2.0, C.accent);
  s.addText("Tự cài cửa sổ trượt", { x: M + 0.32, y: 2.36, w: 5.1, h: 0.4, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 19, bold: true, color: C.text });
  body(s, "return_overflowing_tokens của transformers 5.17.0 sinh tối đa 2 cửa sổ, bất kể context dài bao nhiêu:",
       M + 0.32, 2.82, 5.1, 0.6, { size: 12, color: C.muted });
  table(s, ["Độ dài context", "Thực tế", "Đúng ra"], [
    ["210 token", "2", "2"],
    ["420 token", { t: "2", b: true, c: C.accentDeep }, "5"],
    ["700 token", { t: "2", b: true, c: C.accentDeep }, "8"],
    ["1400 token", { t: "2", b: true, c: C.accentDeep }, "16"],
  ], M + 0.32, 3.5, 5.1, [2.5, 1.3, 1.3], { size: 12, rowH: 0.33 });
  body(s, "Đuôi context bị cắt ÂM THẦM: không exception, không cảnh báo — đáp án nằm cuối đoạn văn thì không bao giờ tìm được.",
       M + 0.32, 5.45, 5.1, 0.7, { size: 12, bold: true, color: C.accentDeep });

  card(s, 7.55, 1.85, 5.03, 4.5);
  label(s, "ADR-004 · ADR-005", 7.87, 2.1, 2.6, C.sage);
  s.addText("Dấu tiếng Việt là\nràng buộc cứng", { x: 7.87, y: 2.36, w: 4.4, h: 0.75, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 19, bold: true, color: C.text, lineSpacing: 22 });
  body(s, "PhoBERT không có fast tokenizer ⇒ không có offset_mapping ⇒ cách duy nhất lấy lại chuỗi là tokenizer.decode().",
       7.87, 3.2, 4.4, 0.8, { size: 12, color: C.muted });
  s.addShape(pres.ShapeType.roundRect, { x: 7.87, y: 4.05, w: 4.4, h: 0.6,
    fill: { color: C.sageSoft }, line: { color: C.sageSoft, width: 0 }, rectRadius: 0.14 });
  mono(s, "decode()  làm mất dấu:   “hoà” → “hoa”", 8.05, 4.05, 4.1, 0.6, { size: 12, color: C.sageDeep });
  body(s, "Sai một dấu là sai cả EM lẫn bất biến substring. Nên hệ thống cắt thẳng chuỗi gốc theo offset ký tự — và TransformerQA TỪ CHỐI khởi tạo mô hình không có fast tokenizer.",
       7.87, 4.8, 4.4, 1.0, { size: 12, color: C.text });
  body(s, "⇒ Thay PhoBERT bằng uitnlp/visobert — cũng là encoder tiếng Việt, của chính UIT NLP",
       7.87, 5.8, 4.4, 0.5, { size: 12, bold: true, color: C.sageDeep });

  footer(s, "4 · Phương pháp · Quyết định kỹ thuật", 14);
  s.addNotes("[10:20–11:15] Hai quyết định đáng nói. Thứ nhất: thư viện transformers phiên bản 5.17 chỉ sinh tối đa hai cửa sổ dù context dài bao nhiêu — bọn em đo được: 420, 700, 1400 token đều ra đúng 2 cửa sổ. Phần đuôi bị cắt âm thầm, nên đáp án nằm cuối đoạn văn thì không bao giờ tìm được; bọn em tự cài lại cơ chế cửa sổ trượt. Thứ hai: đề tài nêu PhoBERT, nhưng PhoBERT không có fast tokenizer, nên không có offset_mapping, nên cách duy nhất lấy lại chuỗi là decode — và decode làm mất dấu tiếng Việt. Đây là ràng buộc kỹ thuật, không liên quan gì đến GPU. Nên bọn em đổi sang ViSoBERT, cũng là encoder tiếng Việt, của chính UIT NLP.");
}

// ══════════════════════════════════════════════════════════════════════════
// 15 — 4.6 Pipeline huấn luyện: siêu tham số và chẩn đoán đường cong
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "4 · Phương pháp · Huấn luyện");
  title(s, "Huấn luyện: nhãn ký tự → token, chọn epoch bằng hàm thuần", false, 30);

  const lrFmt = (v) => Number(v).toExponential().replace("e-5", "·10⁻⁵").replace("e-6", "·10⁻⁶");
  const cm = CFG("mbert"), cv = CFG("visobert");

  card(s, M, 1.78, 6.3, 2.55);
  label(s, "Siêu tham số — đọc từ results/training_curve_*.json", M + 0.32, 1.96, 5.6, C.accent);
  table(s, ["Tham số", "mBERT", "ViSoBERT"], [
    ["Số epoch · seed", cm.epochs + " · " + cm.seed, cv.epochs + " · " + cv.seed],
    ["Batch × tích luỹ gradient", cm.batch_size + " × " + cm.grad_accum, cv.batch_size + " × " + cv.grad_accum],
    ["Tốc độ học", lrFmt(cm.lr), { t: lrFmt(cv.lr), b: true, c: C.accentDeep }],
    ["max_length / doc_stride", cm.max_length + " / " + cm.doc_stride, cv.max_length + " / " + cv.doc_stride],
    ["L_max (độ dài đáp án tối đa)", String(cm.max_answer_len || DATA.lmaxMbert),
      { t: String(cv.max_answer_len), b: true, c: C.accentDeep }],
    ["Số cửa sổ huấn luyện", DATA.features[0], { t: DATA.features[1], b: true }],
  ], M + 0.32, 2.24, 5.6, [2.9, 1.35, 1.35], { size: 10.5, rowH: 0.24 });

  card(s, 7.25, 1.78, 5.33, 2.55, C.accentSoft);
  label(s, "Chỗ sai âm thầm nguy hiểm nhất của cả pipeline", 7.55, 1.96, 4.7, C.accentDeep);
  body(s, "Chuyển vị trí đáp án từ KÝ TỰ sang TOKEN. Nếu ánh xạ lệch, mô hình học nhãn sai mà loss vẫn giảm bình thường.",
       7.55, 2.22, 4.7, 0.6, { size: 11.5, color: C.accentDeep });
  body(s, "⇒ Bất biến được test không phải “chạy không lỗi”, mà: giải mã ngược nhãn token phải ra ĐÚNG chuỗi đáp án vàng.",
       7.55, 2.84, 4.7, 0.6, { size: 11.5, bold: true, color: C.accentDeep });
  label(s, "Chọn epoch", 7.55, 3.44, 4.7, C.accentDeep);
  body(s, "Theo F1 validation, không theo loss (loss là đại lượng trên train). Logic nằm trong training.py dưới dạng hàm thuần — test trong vài mili-giây thay vì chờ một epoch 35 phút.",
       7.55, 3.68, 4.7, 0.6, { size: 11, color: C.accentDeep });

  label(s, "Đường cong huấn luyện — chẩn đoán tự động, ngưỡng COLLAPSE_TOLERANCE = 0,5", M, 4.48, 8.5, C.accent);
  const rows = [];
  ["mbert", "visobert"].forEach((k) => {
    const cur = CURVE[k].curve;
    const best = cur.reduce((a, b) => (b.val_f1 > a.val_f1 ? b : a), cur[0]);
    cur.forEach((e) => {
      const gap = e.val_f1 - e.val_em;
      const diag = gap <= 0.5
        ? { t: "EM ≈ F1 ⇒ suy sụp về trả rỗng", c: C.accentDeep }
        : (e.epoch === best.epoch
            ? { t: "epoch được chọn làm mô hình cuối", c: C.sageDeep }
            : { t: "còn cải thiện", c: C.muted });
      rows.push([
        (k === "mbert" ? "mBERT" : "ViSoBERT") + " · epoch " + e.epoch,
        fmt(e.train_loss, 4), fmt(e.val_em), fmt(e.val_f1),
        { t: fmt(gap), b: gap <= 0.5, c: gap <= 0.5 ? C.accentDeep : C.text }, diag,
      ]);
    });
  });
  table(s, ["Mô hình · epoch", "Loss train", "Val EM", "Val F1", "F1 − EM", "Chẩn đoán"],
        rows, M, 4.72, CW, [2.5, 1.35, 1.15, 1.15, 1.15, 4.53], { size: 11, rowH: 0.26 });
  body(s, "F1 cho điểm bán phần nên bình thường phải cao hơn EM. Hai giá trị trùng nhau ⇒ mỗi câu chỉ có thể đúng hoàn toàn hoặc sai hoàn toàn — dấu hiệu suy sụp về “luôn trả rỗng”, và training.py tự nhận ra điều đó.",
       M, 6.56, CW, 0.3, { size: 10, italic: true, color: C.muted });

  footer(s, "4 · Phương pháp · Pipeline huấn luyện", 15);
  s.addNotes("[11:15–12:10] Về huấn luyện, em nói hai chỗ. Thứ nhất là chỗ sai âm thầm nguy hiểm nhất của cả pipeline: chuyển vị trí đáp án từ ký tự sang token. Nếu ánh xạ đó lệch, mô hình học nhãn sai mà loss vẫn giảm bình thường — không có gì báo lỗi. Nên bất biến bọn em test không phải “chạy không lỗi”, mà là giải mã ngược nhãn token phải ra đúng chuỗi đáp án vàng. Thứ hai là chọn epoch: chọn theo F1 validation chứ không theo loss, đánh giá trên 300 câu ngẫu nhiên sau mỗi epoch, và toàn bộ logic ra quyết định nằm trong một hàm thuần — test trong vài mili-giây thay vì phải chạy một epoch 35 phút mới biết đúng sai. Bảng dưới là đường cong thật: mBERT giảm loss mạnh và F1 còn tăng ở epoch 2, tức là còn thiếu epoch. ViSoBERT thì EM bằng đúng F1 ở epoch 1 và 2, và hàm chẩn đoán tự gắn cờ suy sụp. Em bóc tách chỗ này ở mục 5. Lưu ý: nhóm em KHÔNG tìm kiếm siêu tham số có hệ thống, và điều đó được nói rõ trong phần hạn chế.");
}

// ══════════════════════════════════════════════════════════════════════════
// 16 — 5.1 Bảng kết quả chính
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "5 · Kết quả & phân tích");
  title(s, "UIT-ViQuAD 2.0 · validation · n = " + nOf("mbert") + " · thiết bị " + EV.mbert.device.toUpperCase());

  card(s, M, 1.8, 6.55, 3.5);
  table(s, ["Model", "EM", "F1", "Độ trễ"], [
    ["TF-IDF Baseline", em("tfidf"), f1("tfidf"), lat("tfidf") + " ms"],
    ["ViSoBERT + QA (fine-tuned)", em("visobert"), f1("visobert"), lat("visobert") + " ms"],
    ["XLM-R squad2 (zero-shot)", em("xlmr"), f1("xlmr"), lat("xlmr") + " ms"],
    [{ t: "mBERT + QA (fine-tuned)", b: true }, { t: em("mbert"), b: true, c: C.accentDeep },
     { t: f1("mbert"), b: true, c: C.accentDeep }, { t: lat("mbert") + " ms", b: true }],
  ], M + 0.32, 2.1, 5.9, [2.85, 1.0, 1.0, 1.05], { size: 12.5, rowH: 0.54 });

  s.addChart(pres.ChartType.bar, [
    { name: "EM", labels: ["TF-IDF", "ViSoBERT", "XLM-R", "mBERT"],
      values: [EV.tfidf.overall.EM, EV.visobert.overall.EM, EV.xlmr.overall.EM, EV.mbert.overall.EM] },
    { name: "F1", labels: ["TF-IDF", "ViSoBERT", "XLM-R", "mBERT"],
      values: [EV.tfidf.overall.F1, EV.visobert.overall.F1, EV.xlmr.overall.F1, EV.mbert.overall.F1] },
  ], { ...chartBase, x: 7.15, y: 1.7, w: 5.55, h: 3.7, barDir: "col", valAxisMaxVal: 70 });

  const commits = [...new Set(Object.values(EV).map((e) => e.commit))];
  const kpis = [
    [em("mbert"), "EM cao nhất — mBERT fine-tuned", C.accent],
    [DATA.ci.em, "khoảng tin cậy 95% của EM ở n = " + nOf("mbert"), C.sage],
    [impShare + "%", "câu không có đáp án trong mẫu", C.muted],
    [String(commits.length), "commit truy vết được cho 4 kết quả", C.muted],
  ];
  kpis.forEach(([v, c, col], i) => stat(s, v, "", c, M + i * 3.0, 5.65, 2.8, col, 28));

  footer(s, "5 · Kết quả · results/eval_*.json · commit " + commits.join(", "), 16);
  s.addNotes("[12:10–13:05] Đây là bảng kết quả. mBERT fine-tuned thắng. Baseline TF-IDF có F1 hơn 23 nhưng EM gần 0 — vì nó trả về CẢ MỘT CÂU trong khi gold là cụm vài từ. Khoảng cách EM–F1 đó chính là bằng chứng hai thang đo đo hai thứ khác nhau. Mọi con số trên slide này được deck ĐỌC TRỰC TIẾP từ results khi dựng, kèm commit và thiết bị — không con số nào gõ tay. Nhưng bảng tổng này che mất hai điều, và em sẽ bóc tách ở hai slide tiếp theo.");
}

// ══════════════════════════════════════════════════════════════════════════
// 17 — 5.2 Phát hiện 1: điểm tổng trộn hai kỹ năng
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "5 · Kết quả & phân tích · Phát hiện 1");
  title(s, "mBERT thắng KHÔNG phải vì tìm span giỏi hơn");

  s.addChart(pres.ChartType.bar, [
    { name: "answerable (tìm đúng span)", labels: ["TF-IDF", "ViSoBERT", "XLM-R", "mBERT"],
      values: [EV.tfidf.answerable_only.EM, EV.visobert.answerable_only.EM, EV.xlmr.answerable_only.EM, EV.mbert.answerable_only.EM] },
    { name: "impossible (biết trả lời rỗng)", labels: ["TF-IDF", "ViSoBERT", "XLM-R", "mBERT"],
      values: [EV.tfidf.impossible_only.EM, EV.visobert.impossible_only.EM, EV.xlmr.impossible_only.EM, EV.mbert.impossible_only.EM] },
  ], { ...chartBase, x: M - 0.15, y: 1.7, w: 6.9, h: 4.2, barDir: "col", valAxisMaxVal: 90, valAxisTitle: "EM (%)" });

  card(s, 7.55, 1.85, 5.03, 2.0, C.accentSoft);
  body(s, "Trên câu answerable, XLM-R zero-shot thực ra TỐT HƠN", 7.87, 2.12, 4.4, 0.6,
       { size: 14, bold: true, color: C.accentDeep });
  s.addText("F1  " + ansF1("xlmr") + "   so với   " + ansF1("mbert"), {
    x: 7.87, y: 2.78, w: 4.4, h: 0.4, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 22, bold: true, color: C.accentDeep });
  body(s, "— nhưng chênh lệch " + DATA.ci.ansF1Gap + " điểm này nằm gọn trong nhiễu, nên không phải kết luận vững.",
       7.87, 3.25, 4.4, 0.5, { size: 11.5, color: C.accentDeep });

  card(s, 7.55, 4.05, 5.03, 2.3, C.sageSoft);
  body(s, "mBERT thắng tổng thể vì biết khi nào KHÔNG nên trả lời", 7.87, 4.32, 4.4, 0.6,
       { size: 14, bold: true, color: C.sageDeep });
  s.addText("impossible EM  " + impEM("mbert") + "  so với  " + impEM("xlmr"), {
    x: 7.87, y: 4.95, w: 4.4, h: 0.4, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 19, bold: true, color: C.sageDeep });
  body(s, DATA.ci.impGap + " — có ý nghĩa thống kê. Với " + impShare + "% câu trong mẫu là không có đáp án, chính kỹ năng thứ hai quyết định bảng xếp hạng.",
       7.87, 5.42, 4.4, 0.75, { size: 11.5, color: C.sageDeep });

  body(s, "Đây là lý do báo cáo tách answerable_only và impossible_only — con số tổng trộn hai kỹ năng và che mất điều này.",
       M, 6.42, CW, 0.4, { size: 12.5, italic: true, color: C.muted, align: "center" });

  footer(s, "5 · Kết quả · Hai kỹ năng khác nhau", 17);
  s.addNotes("[13:05–14:05] Đây là phát hiện nhóm em thích nhất. Nhìn bảng tổng thì mBERT thắng XLM-R khoảng 10 điểm EM, và ta dễ kết luận mBERT tìm span giỏi hơn. Sai. Bóc tách ra: trên câu CÓ đáp án, XLM-R zero-shot thực ra tốt hơn một chút về F1, dù chênh lệch đó nằm trong nhiễu. mBERT thắng vì nó biết khi nào KHÔNG nên trả lời: hơn 41 so với hơn 27 trên câu impossible, chênh gần 14 điểm và có ý nghĩa thống kê. Vì gần 30% câu là không có đáp án, chính kỹ năng thứ hai quyết định bảng xếp hạng. Nếu chỉ báo cáo EM tổng thì cả kết luận này biến mất — đó chính là GAP em nói ở mục 2.");
}

// ══════════════════════════════════════════════════════════════════════════
// 18 — 5.3 Phát hiện 2: ViSoBERT suy sụp — và nguyên nhân cấu hình
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "5 · Kết quả & phân tích · Phát hiện 2");
  title(s, "Vì sao ViSoBERT suy sụp — một ca lỗi được chẩn đoán", false, 31);

  card(s, M, 1.72, 5.5, 2.38, C.accentSoft);
  s.addText("EM tổng  " + em("visobert"), { x: M + 0.32, y: 1.88, w: 5.0, h: 0.42, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 23, bold: true, color: C.accentDeep });
  s.addText("=   tỉ lệ câu không có đáp án của mẫu  " + impShare + "%", {
    x: M + 0.32, y: 2.3, w: 5.0, h: 0.36, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 17, bold: true, color: C.accentDeep });
  body(s, "Hai con số trùng nhau không phải trùng hợp — bóc tách ra:", M + 0.32, 2.7, 5.0, 0.3,
       { size: 11, italic: true, color: C.accentDeep });
  stat(s, impEM("visobert"), "", "impossible EM — gần như chỉ\năn điểm từ việc từ chối", M + 0.32, 3.04, 2.4, C.sage, 24);
  stat(s, ansEM("visobert"), "", "answerable EM — hầu như\nkhông tìm được span nào", M + 2.95, 3.04, 2.4, C.accent, 24);

  card(s, 6.55, 1.72, 6.03, 2.38, C.sageSoft);
  label(s, "Nguyên nhân: từ vựng nhỏ chia vụn văn bản Wikipedia", 6.87, 1.88, 5.4, C.sageDeep);
  table(s, ["", "mBERT", "ViSoBERT"], [
    ["vocab", DATA.tok.vocab[0], { t: DATA.tok.vocab[1], b: true, c: C.accentDeep }],
    ["“Hà Nội là thủ đô…” → số token", DATA.tok.sentence[0], { t: DATA.tok.sentence[1], b: true, c: C.accentDeep }],
    ["context validation, TB (token)", DATA.tok.ctxMean[0], { t: DATA.tok.ctxMean[1], b: true, c: C.accentDeep }],
    ["context > 357 token", DATA.tok.over357[0], { t: DATA.tok.over357[1], b: true, c: C.accentDeep }],
  ], 6.87, 2.18, 5.4, [2.7, 1.35, 1.35], { size: 10.5, rowH: 0.3 });
  body(s, "Mỗi cửa sổ thêm vào là một cơ hội nữa để điểm “rỗng” thắng.", 6.87, 3.72, 5.4, 0.3,
       { size: 11, italic: true, color: C.sageDeep });

  card(s, M, 4.2, CW, 1.35, C.card);
  label(s, "Bốn yếu tố đã đủ để giải thích sự suy sụp", M + 0.32, 4.32, 6.0, C.accent);
  const factors = [
    ["max_length = " + CFG("visobert").max_length + " chưa chỉnh", "CHƯA bù trừ — yếu tố mạnh nhất", C.accentDeep],
    ["loss chưa hội tụ", fmt(CURVE.visobert.curve[2].train_loss, 2) + " sau " + CFG("visobert").epochs +
      " epoch (mBERT " + fmt(CURVE.mbert.curve[1].train_loss, 2) + " sau " + CFG("mbert").epochs + ")", C.muted],
    ["sức chứa nhỏ hơn 45%", DATA.tok.params[1] + " so với " + DATA.tok.params[0] + " tham số", C.muted],
    ["mất cân bằng lớp", DATA.split.impPct[0] + " impossible ⇒ hố “luôn trả rỗng”", C.muted],
  ];
  factors.forEach(([k, v, col], i) => {
    const x = M + 0.32 + i * 2.87;
    body(s, k, x, 4.58, 2.7, 0.3, { size: 11.5, bold: true, color: i === 0 ? C.accentDeep : C.text });
    body(s, v, x, 4.86, 2.7, 0.5, { size: 10, color: col });
  });
  body(s, "Nói rõ cho công bằng: L_max (độ dài ĐÁP ÁN) ĐÃ được bù, " + DATA.lmaxMbert + " → " +
          CFG("visobert").max_answer_len + " — trục CHƯA bù là max_length. Hai tham số rất dễ nói lẫn.",
       M + 0.32, 5.24, 11.2, 0.28, { size: 10.5, bold: true, color: C.accentDeep });

  keybox(s, "Chưa kết luận được rằng tiền huấn luyện tiếng Việt không mang lại lợi ích.",
         "Từ vựng nhỏ chia vụn văn bản Wikipedia và ngân sách cửa sổ không được chỉnh theo — lời giải thích PHÙ HỢP với bằng chứng, chưa phải nhân quả đã chứng minh. ViSoBERT chưa huấn luyện lại sau chẩn đoán, nên " +
         em("visobert") + " EM phải đọc như MỘT LẦN HUẤN LUYỆN THẤT BẠI, không phải một phép đo năng lực encoder.",
         5.65, 1.15, "accent");

  footer(s, "5 · Kết quả · Một ca lỗi được chẩn đoán", 18);
  s.addNotes("[14:05–15:20] ViSoBERT — mô hình THAY THẾ cho PhoBERT — đạt EM tổng " + em("visobert") + ", và con số đó bằng ĐÚNG tỉ lệ câu không có đáp án của mẫu. Đó không phải trùng hợp: bóc tách ra thì impossible EM là 82 nhưng answerable EM chỉ gần 7 — nó gần như chỉ ăn điểm từ việc từ chối trả lời. Dấu hiệu đã xuất hiện ngay từ epoch 1 trên đường cong huấn luyện. Khi rà lại bằng chứng, bọn em thấy bốn yếu tố đã ĐỦ để giải thích sự suy sụp. Yếu tố mạnh nhất: max_length để nguyên 384, đặt theo tokenizer mBERT. Với tokenizer ViSoBERT thì 154 trên 557 đoạn văn vượt ngân sách, so với 16 của mBERT — gấp gần mười lần. Đây là chỗ dễ nói lẫn nhất nên em nói rõ: độ dài ĐÁP ÁN thì bọn em đã bù, 30 thành 64; trục chưa bù là max_length. Vì vậy báo cáo KHÔNG kết luận rằng tiền huấn luyện tiếng Việt không giúp ích. Con số " + em("visobert") + " phải đọc như một lần huấn luyện thất bại đã được chẩn đoán, và phép kiểm trực tiếp là huấn luyện lại với max_length lớn hơn.");
}

// ══════════════════════════════════════════════════════════════════════════
// 19 — 6.1 Ứng dụng: sản phẩm chạy được   [slide cắt được nếu thiếu thời gian]
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "6 · Ứng dụng, kết luận & hướng phát triển");
  title(s, "Ứng dụng: sáu màn hình, và một lệnh để chấm");

  const screens = [
    ["Hỏi đáp", "/", "nhập câu hỏi, kéo ngưỡng, xem span + bằng chứng", C.accent],
    ["Kết quả", "/ket-qua", "bảng EM/F1, breakdown, provenance", C.sage],
    ["Phân tích lỗi", "/phan-tich-loi", "pred (terracotta) so với gold (sage)", C.accent],
    ["Dữ liệu", "/du-lieu", "thống kê split, phân phối độ dài", C.sage],
    ["So sánh model", "/so-sanh", "bốn mô hình cạnh nhau, ô tốt nhất được TÍNH", C.accent],
    ["Huấn luyện", "/huan-luyen", "đường cong loss, EM, F1", C.sage],
  ];
  screens.forEach(([n, url, d, col], i) => {
    const x = M + (i % 3) * 4.02, y = 1.8 + Math.floor(i / 3) * 1.28;
    card(s, x, y, 3.78, 1.12);
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.3, y: y + 0.28, w: 0.16, h: 0.16, fill: { color: col }, line: { color: col, width: 0 } });
    s.addText(n, { x: x + 0.58, y: y + 0.16, w: 2.0, h: 0.34, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 15, bold: true, color: C.text, valign: "middle" });
    s.addText(url, { x: x + 2.4, y: y + 0.16, w: 1.2, h: 0.34, isTextBox: true, margin: 0,
      fontFace: "Courier New", fontSize: 9.5, color: C.muted, align: "right", valign: "middle" });
    body(s, d, x + 0.3, y + 0.55, 3.2, 0.5, { size: 10.5, color: C.muted });
  });

  card(s, M, 4.42, CW, 1.28, C.surface, false);
  label(s, "Kỹ thuật phần mềm — người chấm chỉ cần Docker, một ảnh cho cả test lẫn demo", M + 0.35, 4.54, 7.0, C.accentDeep);
  const cmds = [
    ["docker compose run --rm tests", DATA.tests.fast + " test, ~1 giây, HF_HUB_OFFLINE=1"],
    ["docker compose up app", "demo tại http://localhost:8501"],
    ["./run.sh", "nạp ảnh đã docker save rồi mở demo"],
  ];
  cmds.forEach(([cmd, what], i) => {
    const y = 4.84 + i * 0.28;
    mono(s, cmd, M + 0.35, y, 4.3, 0.26, { size: 10.5, color: C.text });
    body(s, what, M + 4.8, y, 3.6, 0.26, { size: 10.5, color: C.muted, valign: "middle" });
  });
  stat(s, DATA.tests.full, "test", "bộ đầy đủ, " + DATA.tests.files + " tệp (gồm " + DATA.tests.model + " test tải mô hình thật)",
       8.7, 4.86, 1.8, C.sage, 20);
  stat(s, DATA.docker.size, "", "ảnh Docker (nén " + DATA.docker.zipped + ") — từ " + DATA.docker.from,
       10.7, 4.86, 1.85, C.accent, 20);

  keybox(s, "Demo không sinh ra con số nào.",
         "Mọi số trên màn hình đọc từ results/*.json hoặc đo trực tiếp từ data/raw/ qua một điểm truy cập duy nhất — kể cả thứ hạng “tốt nhất” và ô in đậm trong bảng so sánh, chúng được TÍNH chứ không gõ tay. Huấn luyện lại là màn hình tự nói đúng.",
         5.85, 0.95);

  footer(s, "6 · Ứng dụng · Streamlit sáu màn hình + Docker/TDD", 19);
  s.addNotes("[15:20–16:05] Sản phẩm chạy được: sáu màn hình, mỗi màn hình một URL riêng để lúc thuyết trình mở thẳng được chỗ cần. Hai điều đáng nói về mặt kỹ thuật. Một: bộ test nhanh " + DATA.tests.fast + " test chạy trong khoảng một giây, không tải mô hình, không cần mạng — nên vòng lặp TDD thật sự dùng được; bộ đầy đủ " + DATA.tests.full + " test có cả test tải mô hình thật và test chạy BÊN TRONG container. Hai: một ảnh Docker duy nhất cho cả test lẫn demo, nên không bao giờ có chuyện test xanh trên một bộ thư viện còn demo chạy trên bộ khác. Và demo không sinh ra con số nào — mọi số đi qua đúng một module đọc file. [Nếu còn thời gian: chuyển sang demo thật, hỏi một câu answerable rồi một câu impossible, kéo thanh ngưỡng.]");
}

// ══════════════════════════════════════════════════════════════════════════
// 20 — 6.2 Hạn chế & hướng phát triển
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "6 · Ứng dụng, kết luận & hướng phát triển");
  title(s, "Hạn chế — nói thẳng, và chỗ nào thì đi tiếp");

  const lim = [
    ["PhoBERT chưa từng được chạy", "Giả thuyết trung tâm “encoder tiếng Việt vượt mBERT” CHƯA KIỂM ĐƯỢC — ViSoBERT là mô hình thay thế, không phải phép kiểm"],
    ["ViSoBERT chưa huấn luyện lại sau chẩn đoán", "Lời giải thích max_length vẫn là giả thuyết phù hợp bằng chứng, chưa phải nhân quả"],
    ["Đánh giá trên validation, không phải test", "Test là blind split — giới hạn của dữ liệu công khai, không phải của thiết kế. Validation cũng dùng để chọn epoch"],
    ["n = " + nOf("mbert") + ", một seed duy nhất", "CI 95% khoảng " + DATA.ci.em + " điểm — chênh lệch F1 answerable giữa mBERT và XLM-R KHÔNG có ý nghĩa"],
    ["Độ dài context và question_type", "Mẫu chỉ có " + bylen("mbert", "300+").count + " câu ở nhóm 300+ nên chưa trả lời được; question_type là heuristic tự gán, ngưỡng 0,6 chọn theo quan sát"],
  ];
  lim.forEach(([k, v], i) => {
    const y = 1.80 + i * 0.82;
    card(s, M, y, CW, 0.74, i % 2 === 0 ? C.card : C.surface, i % 2 === 0);
    s.addText("—", { x: M + 0.3, y: y + 0.18, w: 0.3, h: 0.38, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 16, bold: true, color: C.accent, valign: "middle" });
    body(s, k, M + 0.68, y + 0.19, 4.3, 0.38, { size: 12.5, bold: true, valign: "middle" });
    body(s, v, M + 5.2, y + 0.13, 6.35, 0.5, { size: 10.5, color: C.muted });
  });

  card(s, M, 5.98, CW, 0.82, C.sageSoft, false);
  label(s, "Hướng phát triển", M + 0.35, 6.12, 2.5, C.sageDeep);
  body(s, "Huấn luyện lại ViSoBERT với max_length 768 và lr thấp hơn — phép kiểm trực tiếp  ·  chạy --full trên " +
          DATA.split.questions[1] + " câu  ·  thêm epoch cho mBERT (loss vẫn đang giảm)  ·  tích hợp PhoBERT khi có đường lấy offset  ·  mở rộng dữ liệu ra ngoài miền Wikipedia",
       M + 2.6, 6.14, 9.0, 0.55, { size: 11, color: C.sageDeep });

  footer(s, "6 · Hạn chế & hướng phát triển", 20);
  s.addNotes("[16:05–17:00] Năm hạn chế, và em xếp hai cái quan trọng nhất lên đầu. Một: PhoBERT chưa từng được chạy, nên giả thuyết trung tâm của đề tài vẫn CHƯA kiểm được, không phải bị bác bỏ. Hai: ViSoBERT chưa được huấn luyện lại sau khi chẩn đoán, nên lời giải thích về max_length vẫn là giả thuyết phù hợp với bằng chứng chứ chưa phải nhân quả. Ba: đánh giá trên validation vì test là blind split, và validation cũng được dùng để chọn epoch nên hai mô hình fine-tune có thể lạc quan nhẹ. Bốn: n bằng 500 với một seed, khoảng tin cậy khoảng 4 điểm. Năm: mẫu chỉ có 9 câu thuộc nhóm đoạn văn dài, nên ảnh hưởng của độ dài context chưa trả lời được. Hướng đi tiếp thì rõ: huấn luyện lại ViSoBERT với max_length lớn hơn là phép kiểm trực tiếp cho chính chẩn đoán ở slide trước.");
}

// ══════════════════════════════════════════════════════════════════════════
// 21 — 6.3 Kết luận
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide(true);
  s.addText("KẾT LUẬN", {
    x: M, y: 1.15, w: CW, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 12, bold: true, charSpacing: 2.4, color: C.accent });
  s.addText("Điểm tổng trộn hai kỹ năng:\ntìm span, và biết khi nào im lặng.", {
    x: M, y: 1.65, w: 11.5, h: 1.7, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 36, bold: true, color: "F5EAD8", lineSpacing: 46 });
  s.addText("Còn câu hỏi “tiếng Việt chuyên biệt có giúp không?” thì vẫn để ngỏ — PhoBERT chưa chạy, lần huấn luyện ViSoBERT đã suy sụp vì một nguyên nhân cấu hình chưa được bù trừ. Nói rõ điều đó cũng là một kết quả.", {
    x: M, y: 3.4, w: 11.0, h: 0.6, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 14, italic: true, color: C.dim });

  const out = [
    [em("mbert"), "EM · mBERT fine-tuned", C.accent],
    ["4", "mô hình, một interface", C.dim],
    [DATA.tests.fast, "test trong ~1 giây", C.dim],
    ["0", "số đo được gõ tay — deck đọc results/", C.accent],
  ];
  out.forEach(([v, c, col], i) => {
    const x = M + i * 3.0;
    s.addShape(pres.ShapeType.roundRect, { x, y: 4.3, w: 2.8, h: 1.65,
      fill: { color: "3B362E" }, line: { color: "3B362E", width: 0 }, rectRadius: 0.16 });
    s.addText(v, { x: x + 0.25, y: 4.5, w: 2.3, h: 0.6, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 30, bold: true, color: col });
    s.addText(c, { x: x + 0.25, y: 5.16, w: 2.35, h: 0.7, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 11.5, color: C.dim, lineSpacing: 15 });
  });

  s.addText("Cảm ơn thầy và các bạn đã lắng nghe.", {
    x: M, y: 6.25, w: 6.0, h: 0.4, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 17, bold: true, color: "F5EAD8" });
  s.addText("docs/ARCHITECTURE.md  ·  docs/SYSTEM_DESIGN.md", {
    x: 6.9, y: 6.3, w: 5.7, h: 0.32, isTextBox: true, margin: 0,
    fontFace: "Courier New", fontSize: 10.5, color: C.dim, align: "right" });

  footer(s, "CS116 · Đề tài T11 · Nhóm 7", 21, true);
  s.addNotes("[17:00–17:45] Tóm lại: bốn mô hình so sánh được trên cùng một interface, mBERT fine-tuned đạt EM " + em("mbert") + "; bộ test nhanh chạy trong một giây; và không số đo nào trong deck này được gõ tay — chúng được đọc từ results khi dựng slide. Phát hiện chính: điểm tổng trộn hai kỹ năng, và bóc tách ra thì mBERT thắng nhờ biết từ chối chứ không nhờ tìm span giỏi hơn. Còn câu hỏi về encoder tiếng Việt thì nhóm em để ngỏ một cách có chủ ý — PhoBERT chưa chạy được vì lý do kỹ thuật, và lần huấn luyện ViSoBERT đã suy sụp vì một nguyên nhân cấu hình bọn em chưa bù trừ. Cảm ơn thầy và các bạn. Nhóm em sẵn sàng nhận câu hỏi.");
}

pres.writeFile({ fileName: __dirname + "/CS116_T11_Slide_BaoCao.pptx" })
  .then((f) => console.log("written:", f));

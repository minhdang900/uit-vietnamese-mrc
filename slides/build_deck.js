const pptxgen = require("pptxgenjs");

// ── Hệ thiết kế "Organic" — sao chép từ src/demo/theme.py TOKENS ──────────
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
    color: dark ? C.accent : C.accent, valign: "middle",
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

// nhãn nhỏ in hoa trong thẻ
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
// 1 — Bìa
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
  s.addText("Bốn mô hình · 500 câu kiểm định · mọi con số sinh từ một lần chạy thật", {
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

  label(s, "Lộ trình 15 phút", M, 5.5, 4.0, C.accent);
  ["Bài toán", "Dữ liệu", "Kiến trúc", "Kết quả", "Ba phát hiện", "Demo"].forEach((t, i) => {
    const x = M + i * 1.99;
    s.addShape(pres.ShapeType.roundRect, { x, y: 5.82, w: 1.85, h: 0.46,
      fill: { color: "3B362E" }, line: { color: "3B362E", width: 0 }, rectRadius: 0.22 });
    s.addText(t, { x, y: 5.82, w: 1.85, h: 0.46, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 12, bold: true, color: C.dim, align: "center", valign: "middle" });
  });

  footer(s, "TP. Hồ Chí Minh · tháng 9 năm 2026", 1, true);
  s.addNotes("[0:00–0:30] Chào thầy và các bạn. Nhóm em làm đề tài T11 — đọc hiểu và trả lời câu hỏi tiếng Việt. Trong 15 phút em sẽ đi qua: bài toán và dữ liệu, kiến trúc hệ thống, kết quả bốn mô hình, và ba điều mà bảng số nói ra còn con số tổng thì che mất.");
}

// ══════════════════════════════════════════════════════════════════════════
// 2 — Bài toán
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "Bài toán");
  title(s, "Trích xuất, không phải sinh");

  card(s, M, 1.85, 6.5, 3.2);
  label(s, "Đoạn văn (context)", M + 0.3, 2.1, 5.9);
  s.addText([
    { text: "Hà Nội là thủ đô của nước Cộng hoà Xã hội chủ nghĩa Việt Nam. Hà Nội được UNESCO công nhận là Thành phố vì hoà bình vào ", options: { color: C.text } },
    { text: "năm 1999", options: { color: C.accentDeep, bold: true, highlight: C.accentSoft } },
    { text: ".", options: { color: C.text } },
  ], { x: M + 0.3, y: 2.42, w: 5.9, h: 1.0, isTextBox: true, margin: 0,
       fontFace: BODY, fontSize: 13.5, lineSpacing: 20 });

  label(s, "Câu hỏi", M + 0.3, 3.55, 5.9);
  s.addText("Hà Nội được UNESCO công nhận vào năm nào?", {
    x: M + 0.3, y: 3.83, w: 5.9, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 13.5, color: C.text });

  s.addShape(pres.ShapeType.roundRect, {
    x: M + 0.3, y: 4.25, w: 2.5, h: 0.5, fill: { color: C.sageSoft },
    line: { color: C.sageSoft, width: 0 }, rectRadius: 0.24 });
  s.addText([
    { text: "Đáp án:  ", options: { bold: true, color: C.sageDeep } },
    { text: "năm 1999", options: { color: C.sageDeep } },
  ], { x: M + 0.45, y: 4.25, w: 2.3, h: 0.5, isTextBox: true, margin: 0,
       fontFace: BODY, fontSize: 13.5, valign: "middle" });

  card(s, 7.6, 1.85, 4.98, 3.2, C.accentSoft);
  body(s, "Mô hình không sinh chữ mới.\nNó chỉ ra vị trí.", 7.9, 2.15, 4.4, 1.1,
       { size: 22, bold: true, color: C.accentDeep });
  body(s, "Vì đáp án luôn nằm trong context, ta có một bất biến kiểm được bằng máy:", 7.9, 3.3, 4.4, 0.6, { size: 13, color: C.accentDeep });
  s.addText("predict(ctx, q)  ⊂  ctx", {
    x: 7.9, y: 4.05, w: 4.4, h: 0.45, isTextBox: true, margin: 0,
    fontFace: "Courier New", fontSize: 15, bold: true, color: C.accentDeep });
  body(s, "và nó được kiểm thử cho mọi mô hình.", 7.9, 4.5, 4.4, 0.3, { size: 12, italic: true, color: C.accentDeep });

  const facts = [["Dataset", "UIT-ViQuAD 2.0"], ["Định dạng", "SQuAD-2.0"], ["Metric", "Exact Match + token-F1"], ["Đầu ra", "(start_char, end_char)"]];
  facts.forEach(([k, v], i) => {
    const x = M + i * 3.02;
    label(s, k, x, 5.4, 2.9);
    s.addText(v, { x, y: 5.68, w: 2.9, h: 0.35, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 15, bold: true, color: C.text });
  });

  footer(s, "Bài toán · Extractive MRC", 2);
  s.addNotes("[0:30–1:30] Nhấn mạnh chữ TRÍCH XUẤT. Mô hình không sinh chữ mới — nó chỉ ra vị trí đáp án nằm ở đâu trong đoạn văn. Điều đó cho ta một bất biến kiểm được bằng máy: kết quả trả về luôn phải là chuỗi con của context. Bọn em test bất biến này cho MỌI mô hình, và nó bắt được cả lỗi tokenizer lệch offset lẫn lỗi mô hình bịa đáp án.");
}

// ══════════════════════════════════════════════════════════════════════════
// 3 — Dữ liệu + phát hiện blind set
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "Dữ liệu · Phát hiện 1");
  title(s, "Tập test là blind set — không chấm được");

  card(s, M, 1.85, 6.9, 2.75);
  label(s, "Đo trực tiếp từ file tải về", M + 0.3, 2.08, 6.3);
  table(s, ["Split", "Câu hỏi", "Context", "Article", "Impossible", "Chấm được"], [
    ["train", "28.454", "4.101", "138", "9.216", "28.454"],
    ["validation", "3.814", "557", "19", "1.161", "3.814"],
    ["test", "7.301", "1.241", "48", "0", { t: "0", b: true, c: C.accentDeep }],
  ], M + 0.3, 2.42, 6.3, [1.42, 0.92, 0.88, 0.83, 1.05, 1.2], { size: 12, rowH: 0.4 });

  card(s, 7.9, 1.85, 4.68, 2.75, C.accentSoft);
  body(s, "7.301 câu test đều có gold rỗng, đồng thời is_impossible = False.", 8.2, 2.1, 4.1, 0.8, { size: 14, bold: true, color: C.accentDeep });
  body(s, "Hai điều đó mâu thuẫn nhau nếu coi là nhãn thật ⇒ đáp án đã bị lược bỏ để dùng cho leaderboard.", 8.2, 2.95, 4.1, 0.9, { size: 12.5, color: C.accentDeep });
  body(s, "Chấm nhầm trên tập này → EM 100%.", 8.2, 3.95, 4.1, 0.35, { size: 13.5, bold: true, color: C.accentDeep });

  card(s, M, 4.85, 11.83, 1.7, C.surface, false);
  body(s, "Bọn em chặn bằng code, không bằng lời hứa", M + 0.35, 5.08, 5.0, 0.35, { size: 15, bold: true });
  s.addText("assert_gradeable(examples)", {
    x: M + 0.35, y: 5.5, w: 5.0, h: 0.35, isTextBox: true, margin: 0,
    fontFace: "Courier New", fontSize: 14, bold: true, color: C.accentDeep });
  body(s, "ném ValueError trước MỌI lần chấm điểm, thay vì để một mô hình luôn trả rỗng đạt EM 100%.", M + 0.35, 5.88, 5.0, 0.4, { size: 11.5, color: C.muted });

  const notes = [
    ["30,4%", "câu validation là impossible — metric tổng trộn hai kỹ năng"],
    ["4.101 ≠ 138", "context không phải article: nhầm hai đơn vị làm hỏng chống leakage"],
  ];
  notes.forEach(([k, v], i) => {
    const x = 6.6 + i * 3.0;
    s.addText(k, { x, y: 5.08, w: 2.8, h: 0.4, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 20, bold: true, color: C.accent });
    body(s, v, x, 5.52, 2.8, 0.8, { size: 11.5, color: C.muted });
  });

  footer(s, "Dữ liệu · UIT-ViQuAD 2.0", 3);
  s.addNotes("[1:30–2:45] Phát hiện đầu tiên, và nó đổi cả thiết kế đánh giá. Toàn bộ 7.301 câu trong split test có đáp án rỗng nhưng cờ is_impossible lại là False — hai điều này mâu thuẫn nhau. Nghĩa là đáp án bị lược bỏ để dùng cho leaderboard. Nếu chạy nhầm trên đó, một mô hình luôn trả rỗng sẽ đạt EM 100% và con số đó trông hoàn toàn hợp lý trong báo cáo. Bọn em chặn bằng một assertion chứ không bằng một đoạn văn hứa là đã cẩn thận. Hệ quả: mọi số cuối cùng đến từ validation.");
}

// ══════════════════════════════════════════════════════════════════════════
// 4 — Bốn hệ thống
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "Phương pháp");
  title(s, "Bốn hệ thống — mỗi cái trả lời một câu hỏi khác nhau");

  const items = [
    ["TF-IDF Baseline", "Bài toán này giải được bằng so khớp từ khoá thuần không?", "không huấn luyện", C.muted],
    ["XLM-R squad2", "Transfer từ SQuAD-2.0 tiếng Anh sang tiếng Việt được bao nhiêu?", "zero-shot", C.muted],
    ["mBERT + QA", "Fine-tune in-domain thêm được bao nhiêu, với encoder đa ngữ?", "fine-tuned · MPS", C.accent],
    ["ViSoBERT + QA", "Pretraining chuyên tiếng Việt đóng góp bao nhiêu?", "fine-tuned · MPS", C.sage],
  ];
  items.forEach(([name, q, tag, col], i) => {
    const x = M + (i % 2) * 6.05, y = 1.85 + Math.floor(i / 2) * 2.05;
    card(s, x, y, 5.78, 1.8);
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.32, y: y + 0.36, w: 0.34, h: 0.34, fill: { color: col }, line: { color: col, width: 0 } });
    s.addText(String(i + 1), { x: x + 0.32, y: y + 0.36, w: 0.34, h: 0.34, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 13, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    s.addText(name, { x: x + 0.82, y: y + 0.3, w: 3.0, h: 0.4, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 19, bold: true, color: C.text, valign: "middle" });
    s.addShape(pres.ShapeType.roundRect, { x: x + 3.95, y: y + 0.36, w: 1.55, h: 0.34,
      fill: { color: C.surface }, line: { color: C.surface, width: 0 }, rectRadius: 0.17 });
    s.addText(tag, { x: x + 3.95, y: y + 0.36, w: 1.55, h: 0.34, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 9.5, color: C.muted, align: "center", valign: "middle" });
    body(s, q, x + 0.32, y + 0.92, 5.15, 0.75, { size: 13, italic: true, color: C.muted });
  });

  body(s, "Giả thuyết đăng ký TRƯỚC khi chạy (results/hypotheses.md) dự đoán encoder tiếng Việt sẽ thắng. Kết quả ngược lại — và được báo cáo nguyên vẹn.",
       M, 6.15, 11.83, 0.5, { size: 13, italic: true, color: C.accentDeep });

  footer(s, "Phương pháp · Bốn mô hình", 4);
  s.addNotes("[2:45–3:45] Bốn hệ thống không phải bốn lần thử cho vui — mỗi cái trả lời một câu hỏi khoa học khác nhau. TF-IDF là sàn: nếu một mô hình phức tạp không vượt được nó thì mô hình đó vô nghĩa. XLM-R đo khả năng transfer từ tiếng Anh. mBERT đo phần fine-tune in-domain. ViSoBERT đo phần pretraining chuyên tiếng Việt. Quan trọng: bọn em ghi giả thuyết TRƯỚC khi chạy, và giả thuyết đó nói ViSoBERT phải thắng.");
}

// ══════════════════════════════════════════════════════════════════════════
// 5 — Kiến trúc bốn tầng
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "Kiến trúc");
  title(s, "Bốn tầng, một chiều phụ thuộc");

  const layers = [
    ["Tầng dữ liệu", "data.py · tagging.py", "parse SQuAD-2.0 → Example · dedup & split theo CONTEXT · assert_no_leakage · assert_gradeable", C.sage, C.sageSoft, C.sageDeep],
    ["Tầng mô hình", "predictor.py · baseline_tfidf.py · transformer_qa.py · windowing.py · features.py · training.py", "một interface duy nhất: Predictor.predict(ctx, q) -> str", C.accent, C.accentSoft, C.accentDeep],
    ["Tầng đánh giá", "normalize.py · metrics.py · evaluate.py", "EM + token-F1 · tách answerable / impossible · provenance · cờ unreliable khi n < 30", C.sage, C.sageSoft, C.sageDeep],
    ["Tầng trình bày", "src/demo/ (hàm thuần) → app/ (chỉ gọi widget) · make_figures.py · make_report.py", "CHỈ ĐỌC results/ — không bao giờ sinh ra con số", C.accent, C.accentSoft, C.accentDeep],
  ];
  layers.forEach(([name, mods, desc, dot, fill, deep], i) => {
    const y = 1.78 + i * 1.14;
    card(s, M, y, 11.83, 0.98, fill, false);
    s.addShape(pres.ShapeType.ellipse, { x: M + 0.32, y: y + 0.4, w: 0.2, h: 0.2, fill: { color: dot }, line: { color: dot, width: 0 } });
    s.addText(name, { x: M + 0.66, y: y + 0.14, w: 2.3, h: 0.36, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 16, bold: true, color: deep, valign: "middle" });
    s.addText(mods, { x: M + 0.66, y: y + 0.46, w: 4.7, h: 0.42, isTextBox: true, margin: 0,
      fontFace: "Courier New", fontSize: 9, color: C.muted, lineSpacing: 11 });
    body(s, desc, M + 5.6, y + 0.2, 5.9, 0.7, { size: 12, color: deep });
    if (i < 3) s.addText("▼", { x: 6.5, y: y + 0.99, w: 0.4, h: 0.15, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 9, color: C.line, align: "center", valign: "middle" });
  });

  s.addText("app/  →  src/demo/  →  src/mrc/  →  results/, data/raw/     ·     một mũi tên, một chiều, không ngoại lệ", {
    x: M, y: 6.34, w: 11.83, h: 0.32, isTextBox: true, margin: 0,
    fontFace: "Courier New", fontSize: 12, bold: true, color: C.accentDeep, align: "center" });

  footer(s, "Kiến trúc · docs/ARCHITECTURE.md", 5);
  s.addNotes("[3:45–5:15] Bốn tầng, và điều đáng nói là quy tắc phụ thuộc: một mũi tên, một chiều. src/mrc không biết demo tồn tại. src/demo không import Streamlit. app/ không chứa logic. Và tầng trình bày chỉ ĐỌC results — nó không bao giờ sinh ra con số nào. Hệ quả đo được của thiết kế này: chín trên mười hai module trong src/mrc không phụ thuộc torch, nên bộ test nhanh 423 test chạy trong khoảng một giây, không tải mô hình, không cần mạng.");
}

// ══════════════════════════════════════════════════════════════════════════
// 6 — Bốn bất biến
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "Kiến trúc · Bất biến");
  title(s, "Bốn bất biến — và cơ chế thực thi bằng máy");

  const inv = [
    ["1", "Extractive", "Mọi predict() trả về chuỗi con của context", "Test cho mọi Predictor; decode_span() ném ValueError nếu khoảng không hợp lệ"],
    ["2", "Không leakage", "Một context chỉ thuộc một split", "assert_no_leakage() chạy trong mọi đường split và làm FAIL cả run"],
    ["3", "Truy vết được", "Mọi kết quả mang commit, timestamp, device, split, n", "run_evaluation() sinh các trường này — không đường nào tạo ra kết quả thiếu chúng"],
    ["4", "Có n kèm số", "Nhóm count < 30 tự gắn unreliable: true", "breakdown() gắn cờ; không bảng nào trong báo cáo có số thiếu n"],
  ];
  inv.forEach(([n, name, what, how], i) => {
    const y = 1.78 + i * 1.16;
    card(s, M, y, 11.83, 1.05);
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
       M, 6.46, 11.83, 0.32, { size: 12, italic: true, color: C.accentDeep, align: "center" });

  footer(s, "Kiến trúc · Bất biến", 6);
  s.addNotes("[5:15–6:30] Điều làm đồ án này khác một bài tập thông thường không phải là bốn bất biến, mà là mỗi bất biến có một cơ chế thực thi BẰNG MÁY thay vì một đoạn văn trong báo cáo. Một quy trình có thể bị bỏ qua lúc gấp — đúng lúc nó cần nhất. Một raise thì không. Và bất biến thứ năm thì thậm chí do kernel bắt buộc: ba thư mục dữ liệu được mount read-only.");
}

// ══════════════════════════════════════════════════════════════════════════
// 7 — Hai quyết định kỹ thuật
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "Kiến trúc · Quyết định");
  title(s, "Hai quyết định kỹ thuật đã đổi cả hệ thống");

  card(s, M, 1.85, 5.78, 4.5);
  label(s, "ADR-003", M + 0.32, 2.1, 2.0, C.accent);
  s.addText("Tự cài cửa sổ trượt", { x: M + 0.32, y: 2.36, w: 5.1, h: 0.4, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 19, bold: true, color: C.text });
  body(s, "return_overflowing_tokens của transformers 5.17.0 sinh tối đa 2 cửa sổ, bất kể context dài bao nhiêu:", M + 0.32, 2.82, 5.1, 0.6, { size: 12, color: C.muted });
  table(s, ["Độ dài context", "Thực tế", "Đúng ra"], [
    ["210 token", "2", "2"],
    ["420 token", { t: "2", b: true, c: C.accentDeep }, "5"],
    ["700 token", { t: "2", b: true, c: C.accentDeep }, "8"],
    ["1400 token", { t: "2", b: true, c: C.accentDeep }, "16"],
  ], M + 0.32, 3.5, 5.1, [2.5, 1.3, 1.3], { size: 12, rowH: 0.33 });
  body(s, "Đuôi context bị cắt ÂM THẦM: không exception, không cảnh báo.", M + 0.32, 5.5, 5.1, 0.6, { size: 12, bold: true, color: C.accentDeep });

  card(s, 7.55, 1.85, 5.03, 4.5);
  label(s, "ADR-004 · ADR-005", 7.87, 2.1, 2.6, C.sage);
  s.addText("Dấu tiếng Việt là\nràng buộc cứng", { x: 7.87, y: 2.36, w: 4.4, h: 0.75, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 19, bold: true, color: C.text, lineSpacing: 22 });
  body(s, "PhoBERT không có fast tokenizer ⇒ không có offset_mapping ⇒ cách duy nhất lấy lại chuỗi là tokenizer.decode().", 7.87, 3.2, 4.4, 0.8, { size: 12, color: C.muted });
  s.addShape(pres.ShapeType.roundRect, { x: 7.87, y: 4.05, w: 4.4, h: 0.6,
    fill: { color: C.sageSoft }, line: { color: C.sageSoft, width: 0 }, rectRadius: 0.14 });
  s.addText('decode()  làm mất dấu:   "hoà" → "hoa"', {
    x: 8.05, y: 4.05, w: 4.1, h: 0.6, isTextBox: true, margin: 0,
    fontFace: "Courier New", fontSize: 12, bold: true, color: C.sageDeep, valign: "middle" });
  body(s, "Sai một dấu là sai cả EM lẫn bất biến substring. Nên hệ thống cắt thẳng chuỗi gốc theo offset ký tự — và TransformerQA TỪ CHỐI khởi tạo mô hình không có fast tokenizer.", 7.87, 4.8, 4.4, 1.0, { size: 12, color: C.text });
  body(s, "⇒ Thay PhoBERT bằng uitnlp/visobert", 7.87, 5.85, 4.4, 0.35, { size: 12.5, bold: true, color: C.sageDeep });

  footer(s, "Kiến trúc · Quyết định kỹ thuật", 7);
  s.addNotes("[6:30–7:45] Hai quyết định đáng nói. Thứ nhất: thư viện transformers phiên bản 5.17 chỉ sinh tối đa hai cửa sổ dù context dài bao nhiêu — bọn em đo được: 420, 700, 1400 token đều ra đúng 2 cửa sổ. Phần đuôi bị cắt âm thầm, nên đáp án nằm cuối đoạn văn thì không bao giờ tìm được. Bọn em tự cài lại. Thứ hai: đề tài nêu PhoBERT, nhưng PhoBERT không có fast tokenizer, nên không có offset_mapping, nên cách duy nhất lấy lại chuỗi là decode — và decode làm mất dấu tiếng Việt. Đây là ràng buộc kỹ thuật, không liên quan gì đến GPU. Nên bọn em đổi sang ViSoBERT, cũng là encoder tiếng Việt, của chính UIT NLP.");
}

// ══════════════════════════════════════════════════════════════════════════
// 8 — Kết quả
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "Kết quả");
  title(s, "UIT-ViQuAD 2.0 · validation · n = 500 · thiết bị MPS");

  card(s, M, 1.8, 6.55, 3.5);
  table(s, ["Model", "EM", "F1", "Latency"], [
    ["TF-IDF Baseline", "0,80", "23,09", "0,54 ms"],
    ["ViSoBERT + QA (fine-tuned)", "27,80", "31,31", "22,73 ms"],
    ["XLM-R squad2 (zero-shot)", "40,60", "56,84", "14,35 ms"],
    [{ t: "mBERT + QA (fine-tuned)", b: true }, { t: "50,80", b: true, c: C.accentDeep }, { t: "59,49", b: true, c: C.accentDeep }, { t: "12,26 ms", b: true }],
  ], M + 0.32, 2.1, 5.9, [2.85, 1.0, 1.0, 1.05], { size: 12.5, rowH: 0.54 });

  s.addChart(pres.ChartType.bar, [
    { name: "EM", labels: ["TF-IDF", "ViSoBERT", "XLM-R", "mBERT"], values: [0.8, 27.8, 40.6, 50.8] },
    { name: "F1", labels: ["TF-IDF", "ViSoBERT", "XLM-R", "mBERT"], values: [23.09, 31.31, 56.84, 59.49] },
  ], { ...chartBase, x: 7.15, y: 1.7, w: 5.55, h: 3.7, barDir: "col", valAxisMaxVal: 70 });

  const kpis = [
    ["50,80", "EM cao nhất — mBERT fine-tuned", C.accent],
    ["±4,3", "khoảng tin cậy 95% ở n = 500", C.sage],
    ["27,80", "% câu impossible trong mẫu", C.muted],
    ["4", "commit truy vết được cho 4 kết quả", C.muted],
  ];
  kpis.forEach(([v, c, col], i) => {
    stat(s, v, "", c, M + i * 3.0, 5.65, 2.8, col, 28);
  });

  footer(s, "Kết quả · results/eval_*.json · commit 590e775, c63b28c", 8);
  s.addNotes("[7:45–8:45] Đây là bảng kết quả. mBERT fine-tuned thắng với EM 50,80 và F1 59,49. Baseline TF-IDF có F1 23% nhưng EM chỉ 0,8% — vì nó trả về CẢ MỘT CÂU trong khi gold là cụm vài từ: có overlap token, nhưng gần như không bao giờ trùng khít. Khoảng cách EM–F1 đó là bằng chứng trực quan rằng hai metric đo hai thứ khác nhau. Mọi con số ở đây đọc từ results, kèm commit hash và thiết bị. Nhưng bảng tổng này che mất ba điều — em sẽ bóc tách ở ba slide tiếp theo.");
}

// ══════════════════════════════════════════════════════════════════════════
// 9 — Phát hiện 2
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "Phát hiện 2");
  title(s, "mBERT thắng KHÔNG phải vì tìm span giỏi hơn");

  s.addChart(pres.ChartType.bar, [
    { name: "answerable (tìm đúng span)", labels: ["TF-IDF", "ViSoBERT", "XLM-R", "mBERT"], values: [1.11, 6.93, 45.71, 54.57] },
    { name: "impossible (biết trả lời rỗng)", labels: ["TF-IDF", "ViSoBERT", "XLM-R", "mBERT"], values: [0.0, 82.01, 27.34, 41.01] },
  ], { ...chartBase, x: M - 0.15, y: 1.7, w: 6.9, h: 4.2, barDir: "col", valAxisMaxVal: 90, valAxisTitle: "EM (%)" });

  card(s, 7.55, 1.85, 5.03, 2.0, C.accentSoft);
  body(s, "Trên câu answerable, XLM-R zero-shot thực ra TỐT HƠN", 7.87, 2.12, 4.4, 0.6, { size: 14, bold: true, color: C.accentDeep });
  s.addText("F1  68,20   so với   66,60", { x: 7.87, y: 2.78, w: 4.4, h: 0.4, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 22, bold: true, color: C.accentDeep });
  body(s, "— nhưng chênh lệch 1,60 điểm này nằm gọn trong nhiễu, nên không phải kết luận vững.", 7.87, 3.25, 4.4, 0.5, { size: 11.5, color: C.accentDeep });

  card(s, 7.55, 4.05, 5.03, 2.3, C.sageSoft);
  body(s, "mBERT thắng tổng thể vì biết khi nào KHÔNG nên trả lời", 7.87, 4.32, 4.4, 0.6, { size: 14, bold: true, color: C.sageDeep });
  s.addText("impossible EM  41,01  so với  27,34", { x: 7.87, y: 4.95, w: 4.4, h: 0.4, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 19, bold: true, color: C.sageDeep });
  body(s, "+13,67 điểm ± 11,03 — có ý nghĩa thống kê. Với ~30% câu là unanswerable, chính kỹ năng thứ hai quyết định bảng xếp hạng.", 7.87, 5.42, 4.4, 0.75, { size: 11.5, color: C.sageDeep });

  body(s, "Đây là lý do báo cáo tách answerable_only và impossible_only — con số tổng trộn hai kỹ năng và che mất điều này.",
       M, 6.42, 11.83, 0.4, { size: 12.5, italic: true, color: C.muted, align: "center" });

  footer(s, "Phát hiện · Hai kỹ năng khác nhau", 9);
  s.addNotes("[8:45–10:00] Đây là phát hiện em thích nhất. Nhìn bảng tổng thì mBERT thắng XLM-R 10 điểm EM, và ta dễ kết luận mBERT tìm span giỏi hơn. Sai. Bóc tách ra: trên câu CÓ đáp án, XLM-R zero-shot thực ra tốt hơn một chút — 68,20 so với 66,60 F1, dù chênh lệch đó nằm trong nhiễu. mBERT thắng vì nó biết khi nào KHÔNG nên trả lời: 41,01 so với 27,34 trên câu impossible, chênh gần 14 điểm và có ý nghĩa thống kê. Vì khoảng 30% câu là unanswerable, chính kỹ năng thứ hai quyết định bảng xếp hạng. Nếu chỉ báo cáo EM tổng thì cả kết luận này biến mất.");
}

// ══════════════════════════════════════════════════════════════════════════
// 10 — Phát hiện 3
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "Phát hiện 3");
  title(s, "ViSoBERT suy sụp về “luôn trả lời rỗng”");

  card(s, M, 1.85, 5.6, 2.4, C.accentSoft);
  s.addText([
    { text: "EM tổng  27,80", options: { fontFace: HEAD, fontSize: 26, bold: true, color: C.accentDeep } },
  ], { x: M + 0.35, y: 2.15, w: 5.0, h: 0.5, isTextBox: true, margin: 0 });
  s.addText("=", { x: M + 0.35, y: 2.68, w: 5.0, h: 0.3, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 16, bold: true, color: C.accentDeep });
  s.addText([
    { text: "tỉ lệ impossible của mẫu  27,80%", options: { fontFace: HEAD, fontSize: 22, bold: true, color: C.accentDeep } },
  ], { x: M + 0.35, y: 3.0, w: 5.0, h: 0.5, isTextBox: true, margin: 0 });
  body(s, "Hai con số trùng nhau không phải trùng hợp.", M + 0.35, 3.62, 5.0, 0.35, { size: 12, italic: true, color: C.accentDeep });

  card(s, 6.7, 1.85, 5.88, 2.4);
  label(s, "Bóc tách ra", 7.02, 2.1, 3.0);
  stat(s, "82,01", "", "impossible EM — gần như chỉ ăn điểm\ntừ việc từ chối trả lời", 7.02, 2.45, 2.6, C.sage, 30);
  stat(s, "6,93", "", "answerable EM — nó hầu như\nkhông tìm được span nào", 9.85, 2.45, 2.6, C.accent, 30);

  card(s, M, 4.45, 11.83, 1.85, C.surface, false);
  label(s, "Dấu hiệu xuất hiện ngay trên đường cong huấn luyện", M + 0.35, 4.68, 6.0, C.accentDeep);
  table(s, ["ViSoBERT", "train loss", "val EM", "val F1", "chẩn đoán"], [
    ["epoch 1", "3,0512", "25,67", "25,67", { t: "EM = F1 → suy sụp", c: C.accentDeep }],
    ["epoch 2", "2,5678", "25,33", "25,61", { t: "EM ≈ F1 → suy sụp", c: C.accentDeep }],
    ["epoch 3", "2,1908", "27,00", "30,48", { t: "bắt đầu thoát ra", c: C.sageDeep }],
  ], M + 0.35, 5.0, 11.1, [1.8, 1.8, 1.6, 1.6, 4.3], { size: 11.5, rowH: 0.3 });

  body(s, "F1 cho điểm bán phần nên bình thường phải cao hơn EM — khi hai metric trùng nhau, training.py tự nhận ra bằng hằng số COLLAPSE_TOLERANCE = 0.5.",
       M, 6.46, 11.83, 0.32, { size: 11.5, italic: true, color: C.muted, align: "center" });

  footer(s, "Phát hiện · ViSoBERT collapse", 10);
  s.addNotes("[10:00–11:00] ViSoBERT — mô hình THAY THẾ cho PhoBERT, vốn là mô hình mà giả thuyết ghi trước nói tới — đạt EM tổng 27,80. Con số đó bằng ĐÚNG tỉ lệ câu impossible của mẫu, 27,80%. Đó không phải trùng hợp: bóc tách ra thì impossible EM là 82 nhưng answerable EM chỉ 6,9 — nó gần như chỉ ăn điểm từ việc từ chối trả lời. Và dấu hiệu đã xuất hiện ngay từ epoch 1: val EM bằng đúng val F1. F1 cho điểm bán phần nên bình thường phải cao hơn EM vài điểm; khi hai metric trùng nhau thì mỗi câu chỉ có thể đúng hoàn toàn hoặc sai hoàn toàn. Bọn em có một hằng số trong code để tự phát hiện điều đó.");
}

// ══════════════════════════════════════════════════════════════════════════
// 11 — Vì sao ViSoBERT thất bại
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "Phân tích nguyên nhân");
  title(s, "Vì sao ViSoBERT suy sụp — ca lỗi được chẩn đoán", false, 32);

  card(s, M, 1.78, 5.9, 2.95);
  label(s, "Bốn yếu tố đủ để giải thích sự suy sụp", M + 0.3, 2.0, 5.3, C.accent);
  table(s, ["Yếu tố", "Bằng chứng"], [
    ["max_length = 384 chưa chỉnh", { t: "CHƯA bù trừ — mạnh nhất", b: true, c: C.accentDeep }],
    ["loss chưa hội tụ", { t: "2,19 sau 3 epoch (mBERT 1,30 sau 2)", c: C.muted }],
    ["sức chứa nhỏ hơn 45%", { t: "97,0M so với 177,3M tham số", c: C.muted }],
    ["mất cân bằng lớp", { t: "32,39% impossible ⇒ hố “luôn trả rỗng”", c: C.muted }],
  ], M + 0.3, 2.32, 5.3, [2.4, 2.9], { size: 10, rowH: 0.36 });
  body(s, "max_answer_len ĐÃ bù (30→64). Trục chưa bù là max_length — hai tham số rất dễ nói lẫn.", M + 0.3, 4.18, 5.3, 0.45, { size: 11, bold: true, color: C.accentDeep });

  card(s, 6.85, 1.78, 5.73, 2.95, C.sageSoft);
  label(s, "Nguyên nhân: từ vựng nhỏ chia vụn tiếng Việt", 7.15, 2.0, 5.1, C.sageDeep);
  table(s, ["", "mBERT", "ViSoBERT"], [
    ["vocab", "119.547", { t: "15.002", b: true, c: C.accentDeep }],
    ['"Hà Nội là thủ đô…"', "17 token", { t: "23 token", b: true, c: C.accentDeep }],
    ["context val. trung bình", "204,6", { t: "326,5", b: true, c: C.accentDeep }],
    ["token / từ", "1,22", { t: "1,95", b: true, c: C.accentDeep }],
    ["context > 357 token", "16 / 557", { t: "154 / 557", b: true, c: C.accentDeep }],
  ], 7.15, 2.32, 5.1, [2.4, 1.35, 1.35], { size: 11, rowH: 0.36 });

  card(s, M, 4.9, 11.83, 1.65, C.surface, false);
  s.addText("Chưa kết luận được rằng tiền huấn luyện tiếng Việt không giúp ích.", {
    x: M + 0.4, y: 5.1, w: 11.0, h: 0.45, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 22, bold: true, color: C.accentDeep, align: "center" });
  body(s, "Từ vựng nhỏ chia vụn văn bản Wikipedia, và ngân sách cửa sổ không được chỉnh theo — đó là lời giải thích PHÙ HỢP với bằng chứng, chưa phải nhân quả đã chứng minh. ViSoBERT chưa được huấn luyện lại sau chẩn đoán, nên 27,80 EM phải đọc như một lần huấn luyện thất bại, không phải một phép đo năng lực encoder.",
       M + 0.4, 5.62, 11.0, 0.8, { size: 12.5, color: C.text, align: "center" });

  footer(s, "Phân tích · Một ca lỗi được chẩn đoán", 11);
  s.addNotes("[11:00–12:00] Khi rà lại toàn bộ bằng chứng để viết báo cáo, bọn em nhận ra bốn yếu tố dưới đây đã ĐỦ để giải thích sự suy sụp, mà không cần nói gì về chất lượng của ViSoBERT như một encoder. Yếu tố mạnh nhất: max_length để nguyên 384 và doc_stride 128 — đặt theo tokenizer của mBERT. Với tokenizer ViSoBERT thì 154 trên 557 đoạn văn vượt ngân sách, so với 16 của mBERT, gấp gần mười lần. Mỗi cửa sổ thêm vào là một cơ hội nữa để điểm rỗng thắng. Đây là chỗ dễ nói lẫn nhất, nên em nói rõ: max_answer_len thì bọn em ĐÃ bù, sửa 30 thành 64; trục chưa bù là max_length. Ba yếu tố còn lại: loss chưa hội tụ, sức chứa nhỏ hơn 45 phần trăm, và hố cực tiểu do 32 phần trăm câu impossible trong tập train. Vì vậy báo cáo KHÔNG kết luận rằng tiền huấn luyện tiếng Việt không giúp ích — câu hỏi đó vẫn để ngỏ, và phép kiểm trực tiếp là huấn luyện lại với max_length lớn hơn và learning rate thấp hơn.");
}

// ══════════════════════════════════════════════════════════════════════════
// 12 — Kỹ thuật phần mềm: TDD + Docker
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "Kỹ thuật phần mềm");
  title(s, "Kiến trúc trả về gì: test một giây, một lệnh để chấm");

  const cards = [
    ["423", "test chạy trong ~1 giây", "Không tải mô hình, không cần mạng. Có được là nhờ 9/12 module trong src/mrc và toàn bộ src/demo là hàm thuần.", C.accent],
    ["474", "test đầy đủ, 20 tệp", "Bộ đầy đủ có cả 51 test tải mô hình thật. tests/test_docker.py chạy BÊN TRONG container và tự kiểm chính môi trường đang chạy nó.", C.sage],
    ["2,48 GB", "ảnh Docker (nén 527 MB)", "Từ 10,2 GB. Ép PyTorch bản CPU trên MỌI kiến trúc — wheel aarch64 cũng kéo theo 4,1 GB thư viện CUDA.", C.accent],
  ];
  cards.forEach(([v, cap, desc, col], i) => {
    const x = M + i * 4.02;
    card(s, x, 1.85, 3.78, 2.5);
    s.addText(v, { x: x + 0.3, y: 2.12, w: 3.2, h: 0.6, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 34, bold: true, color: col });
    body(s, cap, x + 0.3, 2.76, 3.2, 0.3, { size: 12.5, bold: true, color: C.text });
    body(s, desc, x + 0.3, 3.12, 3.2, 1.1, { size: 11, color: C.muted });
  });

  card(s, M, 4.6, 11.83, 1.75, C.surface, false);
  label(s, "Người chấm chỉ cần Docker — một ảnh, bốn service", M + 0.4, 4.82, 6.0, C.accentDeep);
  const cmds = [
    ["docker compose run --rm tests", "423 test, ~1 giây, HF_HUB_OFFLINE=1"],
    ["docker compose up app", "demo tại http://localhost:8501"],
    ["./run.sh", "nạp ảnh đã docker save rồi mở demo"],
  ];
  cmds.forEach(([cmd, what], i) => {
    const y = 5.18 + i * 0.36;
    s.addText(cmd, { x: M + 0.4, y, w: 5.2, h: 0.3, isTextBox: true, margin: 0,
      fontFace: "Courier New", fontSize: 11.5, bold: true, color: C.text, valign: "middle" });
    s.addText(what, { x: M + 5.8, y, w: 5.6, h: 0.3, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 11.5, color: C.muted, valign: "middle" });
  });

  footer(s, "Kỹ thuật phần mềm · TDD + Docker", 12);
  s.addNotes("[12:00–12:45] Kiến trúc tách hàm thuần trả về hai thứ cụ thể. Một: bộ test nhanh 423 test chạy trong khoảng một giây, không tải mô hình, không cần mạng — nên vòng lặp TDD thật sự dùng được. Hai: một ảnh Docker duy nhất cho cả test lẫn demo, nên không bao giờ có chuyện test xanh trên một bộ thư viện còn demo chạy trên bộ khác. Ảnh từ 10,2 GB xuống 2,48 GB sau khi phát hiện wheel aarch64 của torch cũng kéo theo 4 GB thư viện CUDA mà container này không bao giờ chạm tới.");
}

// ══════════════════════════════════════════════════════════════════════════
// 13 — Demo
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "Demo");
  title(s, "Sáu màn hình — mỗi màn hình một URL");

  const screens = [
    ["Hỏi đáp", "/", "nhập câu hỏi, kéo ngưỡng, xem span + bằng chứng", C.accent],
    ["Kết quả", "/ket-qua", "bảng EM/F1, breakdown, provenance", C.sage],
    ["Phân tích lỗi", "/phan-tich-loi", "pred (terracotta) so với gold (sage)", C.accent],
    ["Dữ liệu", "/du-lieu", "thống kê split, phân phối độ dài", C.sage],
    ["So sánh model", "/so-sanh", "bốn mô hình cạnh nhau, ô tốt nhất được TÍNH", C.accent],
    ["Huấn luyện", "/huan-luyen", "đường cong loss, EM, F1", C.sage],
  ];
  screens.forEach(([n, url, d, col], i) => {
    const x = M + (i % 3) * 4.02, y = 1.85 + Math.floor(i / 3) * 1.5;
    card(s, x, y, 3.78, 1.3);
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.3, y: y + 0.33, w: 0.16, h: 0.16, fill: { color: col }, line: { color: col, width: 0 } });
    s.addText(n, { x: x + 0.58, y: y + 0.2, w: 2.0, h: 0.36, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 15, bold: true, color: C.text, valign: "middle" });
    s.addText(url, { x: x + 2.4, y: y + 0.2, w: 1.2, h: 0.36, isTextBox: true, margin: 0,
      fontFace: "Courier New", fontSize: 9.5, color: C.muted, align: "right", valign: "middle" });
    body(s, d, x + 0.3, y + 0.65, 3.2, 0.55, { size: 11, color: C.muted });
  });

  card(s, M, 4.95, 11.83, 1.4, C.sageSoft, false);
  body(s, "Demo không sinh ra con số nào.", M + 0.4, 5.18, 3.6, 0.35, { size: 15, bold: true, color: C.sageDeep });
  body(s, "Mọi số trên màn hình đọc từ results/*.json hoặc đo trực tiếp từ data/raw/ qua một điểm truy cập duy nhất — kể cả thứ hạng “tốt nhất” và ô in đậm trong bảng so sánh, chúng được TÍNH chứ không gõ tay. Huấn luyện lại là màn hình tự nói đúng.",
       M + 4.2, 5.15, 7.3, 0.9, { size: 12, color: C.sageDeep });

  footer(s, "Demo · Streamlit, sáu màn hình deep-link", 13);
  s.addNotes("[12:45–13:45] Demo có sáu màn hình, mỗi màn hình một URL riêng để lúc thuyết trình mở thẳng được chỗ cần. Điều đáng nói về mặt kỹ thuật: demo không sinh ra con số nào. Mọi số trên màn hình đi qua đúng một module đọc file, kể cả thứ hạng tốt nhất và ô in đậm trong bảng so sánh — chúng được tính từ dữ liệu chứ không gõ tay. Nên nếu bọn em huấn luyện lại thì màn hình tự nói đúng, không phải sửa gì. [Nếu còn thời gian: chuyển sang demo thật, hỏi một câu answerable rồi một câu impossible, kéo thanh ngưỡng.]");
}

// ══════════════════════════════════════════════════════════════════════════
// 14 — Giới hạn
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  kicker(s, "Giới hạn · Hướng phát triển");
  title(s, "Điều bọn em nói rõ trong báo cáo");

  const lim = [
    ["PhoBERT chưa từng được chạy", "Giả thuyết trung tâm “encoder tiếng Việt vượt mBERT” CHƯA KIỂM ĐƯỢC — ViSoBERT là mô hình thay thế, không phải phép kiểm"],
    ["ViSoBERT chưa huấn luyện lại sau chẩn đoán", "Lời giải thích max_length vẫn là giả thuyết phù hợp bằng chứng, chưa phải nhân quả"],
    ["Đánh giá trên validation, không phải test", "Test là blind split — giới hạn của dataset, không phải của thiết kế"],
    ["n = 500, một seed duy nhất", "CI 95% khoảng ±4,3 điểm — chênh lệch F1 answerable mBERT/XLM-R KHÔNG có ý nghĩa"],
    ["question_type là heuristic tự gán", "Ngưỡng 0.6 chọn theo quan sát — không phải nhãn có sẵn của ViQuAD"],
  ];
  lim.forEach(([k, v], i) => {
    const y = 1.80 + i * 0.82;
    card(s, M, y, 11.83, 0.74, i % 2 === 0 ? C.card : C.surface, i % 2 === 0);
    s.addText("—", { x: M + 0.3, y: y + 0.18, w: 0.3, h: 0.38, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 16, bold: true, color: C.accent, valign: "middle" });
    body(s, k, M + 0.68, y + 0.19, 4.3, 0.38, { size: 12.5, bold: true, valign: "middle" });
    body(s, v, M + 5.2, y + 0.15, 6.35, 0.46, { size: 10.5, color: C.muted });
  });

  card(s, M, 5.98, 11.83, 0.82, C.sageSoft, false);
  label(s, "Bước tiếp theo", M + 0.35, 6.12, 2.5, C.sageDeep);
  body(s, "Huấn luyện lại ViSoBERT với max_length 768 và lr 3e-5 — phép kiểm trực tiếp  ·  chạy --full trên 3.814 câu  ·  thêm epoch 3–4 cho mBERT  ·  tích hợp PhoBERT",
       M + 2.6, 6.14, 9.0, 0.5, { size: 11, color: C.sageDeep });

  footer(s, "Giới hạn · Hướng phát triển", 14);
  s.addNotes("[13:45–14:30] Năm giới hạn bọn em nói rõ, và em xếp hai cái quan trọng nhất lên đầu. Một: PhoBERT chưa từng được chạy, nên giả thuyết trung tâm của đề tài — encoder tiếng Việt có vượt mBERT không — vẫn CHƯA kiểm được, không phải bị bác bỏ. ViSoBERT là mô hình thay thế, không phải phép kiểm cho giả thuyết đó. Hai: ViSoBERT chưa được huấn luyện lại sau khi chẩn đoán, nên lời giải thích về max_length vẫn là giả thuyết phù hợp với bằng chứng chứ chưa phải nhân quả. Ba cái còn lại: đánh giá trên validation vì test là blind split; n=500 một seed, khoảng tin cậy khoảng ±4,3 điểm — chênh lệch F1 answerable giữa mBERT và XLM-R KHÔNG có ý nghĩa thống kê; và question_type là heuristic tự gán. Bốn mô hình đều chưa được huấn luyện đủ, mBERT rõ nhất: loss vẫn giảm ở epoch 2, nên 50,80 là cận dưới chứ không phải trần.");
}

// ══════════════════════════════════════════════════════════════════════════
// 15 — Kết luận
// ══════════════════════════════════════════════════════════════════════════
{
  const s = slide(true);
  s.addText("KẾT LUẬN", {
    x: M, y: 1.15, w: CW, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 12, bold: true, charSpacing: 2.4, color: C.accent });
  s.addText("Điểm tổng trộn hai kỹ năng:\ntìm span, và biết khi nào im lặng.", {
    x: M, y: 1.65, w: 11.5, h: 1.7, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 36, bold: true, color: "F5EAD8", lineSpacing: 46 });
  s.addText("Còn câu hỏi “tiếng Việt chuyên biệt có giúp không?” thì vẫn để ngỏ — PhoBERT chưa chạy, lần huấn luyện ViSoBERT đã suy sụp. Nói rõ điều đó cũng là một kết quả.", {
    x: M, y: 3.4, w: 11.0, h: 0.5, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 14, italic: true, color: C.dim });

  const out = [
    ["50,80", "EM · mBERT fine-tuned", C.accent],
    ["4", "mô hình, một interface", C.dim],
    ["423", "test trong ~1 giây", C.dim],
    ["0", "con số được gõ tay", C.accent],
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

  footer(s, "CS116 · Đề tài T11 · Nhóm 7", 15, true);
  s.addNotes("[14:30–15:00] Tóm lại: bốn mô hình so sánh được trên cùng một interface, mBERT fine-tuned đạt EM 50,80, bộ test 423 ca chạy trong một giây, và không con số nào trong báo cáo được gõ tay — tất cả đều truy vết được về một commit. Phát hiện chính: điểm tổng trộn hai kỹ năng, và bóc tách ra thì mBERT thắng nhờ biết từ chối chứ không nhờ tìm span giỏi hơn. Còn câu hỏi về encoder tiếng Việt thì bọn em để ngỏ một cách có chủ ý — PhoBERT chưa chạy, và lần huấn luyện ViSoBERT đã suy sụp vì một nguyên nhân cấu hình bọn em chưa bù trừ. Cảm ơn thầy và các bạn. Nhóm em sẵn sàng nhận câu hỏi.");
}

pres.writeFile({ fileName: __dirname + "/CS116_T11_Slide_BaoCao.pptx" })
  .then((f) => console.log("written:", f));

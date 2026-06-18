const ACCENT = "#2d5bd7";
const GRID = "#e3e6eb";
const MUTED = "#6b7280";

const MODE_LABEL = { bilingual: "Оба языка", ru: "Русский", en: "Английский" };
const TYPE_LABEL = {
  vocab: "Знание слов",
  sentence_gap: "Пропуск в предложении",
  fake_seen: "Выдуманные слова",
  translate: "Перевод",
};

function statCard(value, label) {
  return `<div class="stat"><div class="n">${value}</div><div class="l">${label}</div></div>`;
}

function fmtTime(iso) {
  if (!iso) return "—";
  const d = iso.replace("T", " ");
  return d.slice(0, 16);
}

function barChart(canvasId, labels, data) {
  new Chart(document.getElementById(canvasId), {
    type: "bar",
    data: { labels, datasets: [{ data, backgroundColor: ACCENT, borderRadius: 6, maxBarThickness: 48 }] },
    options: {
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${c.parsed.y}% верных` } } },
      scales: {
        y: { beginAtZero: true, max: 100, ticks: { color: MUTED, callback: (v) => v + "%" }, grid: { color: GRID } },
        x: { ticks: { color: MUTED }, grid: { display: false } },
      },
    },
  });
}

(async function () {
  const s = await (await fetch("/api/admin/stats")).json();

  document.getElementById("cards").innerHTML =
    statCard(s.total_sessions, "Завершённых тестов") +
    statCard(`${s.avg.ru}%`, "Средний русский") +
    statCard(`${s.avg.en}%`, "Средний английский") +
    statCard(`${s.avg.bilingualism}%`, "Средняя билингвальность") +
    statCard(`${s.avg.honesty}%`, "Средняя честность") +
    statCard(s.fake.total, "Показано ловушек") +
    statCard(`${s.fake.caught_rate}%`, "Попались на ловушку");

  barChart("levelChart", s.by_level.map((r) => r.level), s.by_level.map((r) => r.accuracy));
  barChart(
    "typeChart",
    s.by_type.map((r) => TYPE_LABEL[r.type] || r.type),
    s.by_type.map((r) => r.accuracy)
  );

  const tbody = document.querySelector("#recent tbody");
  if (!s.recent.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="muted">Пока нет завершённых прохождений</td></tr>';
    return;
  }
  tbody.innerHTML = s.recent
    .map(
      (r) => `<tr>
        <td>${fmtTime(r.finished_at)}</td>
        <td>${MODE_LABEL[r.mode] || r.mode}</td>
        <td class="num">${r.ru}%</td>
        <td class="num">${r.en}%</td>
        <td class="num">${r.bilingualism}%</td>
        <td class="num">${r.honesty}%</td>
      </tr>`
    )
    .join("");
})();

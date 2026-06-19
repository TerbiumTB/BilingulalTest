// Отрисовка результата теста из sessionStorage (с фолбэком на API).
// Набор метрик зависит от режима: один язык vs билингвальность.

// Короткие пояснения, всплывающие при наведении (data-tip -> CSS-тултип).
const TIP_FLEX =
  "Доля верных ответов в заданиях на стыке языков (вставка слова другого языка и сжатие текста). Среднее по двум упражнениям.";
const TIP_INTERFERENCE =
  "Насколько вы не попадались на выдуманные слова. Формула: (1 − пойманных ловушек ÷ всего ловушек) × 100%.";

function metric(label, valueHtml, tip) {
  const name = tip
    ? `<span class="name tip" data-tip="${tip}">${label}</span>`
    : `<span class="name">${label}</span>`;
  return `<div class="metric">${name}<span class="val">${valueHtml}</span></div>`;
}

// Балл языка как перцентиль: «лучше X% проходивших». Без статистики — сам балл.
function langValue(o) {
  if (o.percentile != null) return `лучше ${o.percentile}%<small>проходивших</small>`;
  return `${o.percent}%<small>мало данных</small>`;
}

// Баланс как соотношение сил: 50 / 50 — ровно, 60 / 40 — перекос.
function balanceValue(ru, en) {
  const total = ru + en;
  const r = total ? Math.round((ru / total) * 100) : 50;
  return `${r} / ${100 - r}`;
}

// Доминантность — только язык, без числа.
function dominanceValue(ru, en) {
  if (Math.abs(en - ru) < 5) return "Поровну";
  return en > ru ? "Английский" : "Русский";
}

function directionValue(da) {
  if (!da || !da.easier) return "симметрично";
  return `${da.easier}<small>легче</small>`;
}

function interferenceMetric(r) {
  if (r.interference_resistance == null) return "";
  return metric("Устойчивость к интерференции", `${r.interference_resistance}%`, TIP_INTERFERENCE);
}

function render(r) {
  const single = r.mode === "ru" || r.mode === "en";
  const heroLabel = document.getElementById("hero-label");
  const biling = document.getElementById("biling");
  let rows = "";

  if (single) {
    const o = r.mode === "ru" ? r.ru : r.en;
    const name = r.mode === "ru" ? "Русский язык" : "Английский язык";
    heroLabel.textContent = "Ваш уровень";
    biling.textContent = (o.percentile != null ? o.percentile : o.percent) + "%";
    rows += metric(name, langValue(o));
    rows += interferenceMetric(r);
  } else {
    heroLabel.textContent = "Степень билингвальности";
    biling.textContent = `${r.bilingualism}%`;
    rows += metric("Русский язык", langValue(r.ru));
    rows += metric("Английский язык", langValue(r.en));
    rows += metric("Баланс языков (RU / EN)", balanceValue(r.ru.percent, r.en.percent));
    rows += metric("Доминантность", dominanceValue(r.ru.percent, r.en.percent));
    if (r.cross_language_flexibility != null) {
      rows += metric("Межъязыковая гибкость", `${r.cross_language_flexibility}%`, TIP_FLEX);
    }
    rows += interferenceMetric(r);
    if (r.direction_asymmetry) {
      rows += metric("Направленность перевода", directionValue(r.direction_asymmetry));
    }
  }

  document.getElementById("metrics").innerHTML = rows;

  const honestyCard = document.getElementById("honesty");
  if (r.honesty_flag) {
    honestyCard.innerHTML =
      '<p class="flag">⚠️ Вы отмечали «знаю» на выдуманных словах — оценка словарного запаса могла быть завышена.</p>';
  } else {
    honestyCard.style.display = "none";
  }
}

(async function () {
  const cached = sessionStorage.getItem("result");
  if (cached) {
    render(JSON.parse(cached));
    return;
  }
  const sid = sessionStorage.getItem("session_id");
  if (sid) {
    const res = await fetch(`/api/session/${sid}/result`);
    if (res.ok) {
      render(await res.json());
      return;
    }
  }
  document.querySelector(".wrap").innerHTML =
    '<h1>Результат недоступен</h1><p class="muted"><a href="/">← Пройти тест</a></p>';
})();

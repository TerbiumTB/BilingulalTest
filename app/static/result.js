// Отрисовка результата теста из sessionStorage (с фолбэком на API).
function metric(label, valueHtml) {
  return `<div class="metric"><span class="name">${label}</span><span class="val">${valueHtml}</span></div>`;
}

function render(r) {
  document.getElementById("biling").textContent = `${r.bilingualism}%`;

  document.getElementById("metrics").innerHTML =
    metric("Русский язык", `${r.ru.percent}%<small>${r.ru.cefr}</small>`) +
    metric("Английский язык", `${r.en.percent}%<small>${r.en.cefr}</small>`) +
    metric("Сбалансированность языков", `${r.balance}%`);

  let honesty = metric(
    "Честность ответов",
    `${r.honesty}%<small>${r.fake_total - r.fake_caught} из ${r.fake_total} ловушек пройдено</small>`
  );
  if (r.honesty_flag) {
    honesty +=
      '<p class="flag">⚠️ Вы отмечали «знаю» на выдуманных словах — оценка словарного запаса могла быть завышена.</p>';
  }
  document.getElementById("honesty").innerHTML = honesty;
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

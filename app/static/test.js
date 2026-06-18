// Прохождение теста: старт сессии, рендер вопросов, отправка ответов.
const params = new URLSearchParams(location.search);
const mode = params.get("mode") || "bilingual";
const k = parseInt(params.get("k") || "10", 10);

let sessionId = null;
let current = null;
let locked = false;

const els = {
  bar: document.getElementById("bar"),
  counter: document.getElementById("counter"),
  lang: document.getElementById("lang"),
  instruction: document.getElementById("instruction"),
  body: document.getElementById("body"),
};

const LANG_LABEL = { RU: "Русский", EN: "English" };

async function post(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function renderQuestion(q) {
  current = q;
  locked = false;

  els.counter.textContent = `Вопрос ${q.index} из ${q.total}`;
  els.lang.textContent = `${LANG_LABEL[q.lang] || q.lang} · ${q.level}`;
  els.bar.style.width = `${((q.index - 1) / q.total) * 100}%`;
  els.instruction.textContent = q.instruction || "";
  els.body.innerHTML = "";

  if (q.answer_kind === "choice") {
    const sentence = document.createElement("div");
    sentence.className = "sentence";
    sentence.textContent = q.prompt;
    els.body.appendChild(sentence);

    const choices = document.createElement("div");
    choices.className = "choices";
    q.options.forEach((opt) => {
      const b = document.createElement("button");
      b.textContent = opt;
      b.addEventListener("click", () => submit(opt));
      choices.appendChild(b);
    });
    els.body.appendChild(choices);
  } else {
    const word = document.createElement("div");
    word.className = "word";
    word.textContent = q.prompt;
    els.body.appendChild(word);

    const row = document.createElement("div");
    row.className = "binary";
    const yes = document.createElement("button");
    yes.className = "yes";
    yes.textContent = q.yes_label || "Да";
    yes.addEventListener("click", () => submit("yes"));
    const no = document.createElement("button");
    no.className = "no";
    no.textContent = q.no_label || "Нет";
    no.addEventListener("click", () => submit("no"));
    row.appendChild(yes);
    row.appendChild(no);
    els.body.appendChild(row);
  }
}

async function submit(answer) {
  if (locked) return;
  locked = true;
  try {
    const data = await post("/api/session/answer", {
      session_id: sessionId,
      question_id: current.question_id,
      answer,
    });
    if (data.finished) {
      sessionStorage.setItem("result", JSON.stringify(data.result));
      sessionStorage.setItem("session_id", sessionId);
      location.href = "/result";
    } else {
      renderQuestion(data.question);
    }
  } catch (e) {
    locked = false;
    alert("Ошибка: " + e.message);
  }
}

async function start() {
  try {
    const data = await post("/api/session/start", { mode, k });
    sessionId = data.session_id;
    renderQuestion(data.question);
  } catch (e) {
    els.body.textContent = "Не удалось начать тест: " + e.message;
  }
}

start();

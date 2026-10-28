const $ = (selector) => document.querySelector(selector);
const messages = $("#messages");
const input = $("#input");
const sendButton = $("#send");

let sessionId = null;

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

async function startSession() {
  sessionId = sessionStorage.getItem("docchat-session");
  if (!sessionId) {
    ({ session_id: sessionId } = await api("/api/sessions", { method: "POST" }));
    sessionStorage.setItem("docchat-session", sessionId);
  }
  renderFiles((await api(`/api/sessions/${sessionId}/files`)).files);
}

function renderFiles(files) {
  const list = $("#files");
  const names = Object.keys(files);
  list.innerHTML = names.length
    ? names.map((name) => `<li><span>${escapeHtml(name)}</span><small>${files[name]} chunks</small></li>`).join("")
    : '<li class="empty">No documents yet</li>';
}

async function upload(fileList) {
  if (!fileList.length) return;
  const form = new FormData();
  [...fileList].forEach((file) => form.append("files", file));
  const drop = $("#drop");
  drop.classList.add("busy");
  drop.querySelector("strong").textContent = "Reading and indexing…";
  try {
    renderFiles((await api(`/api/sessions/${sessionId}/files`, { method: "POST", body: form })).files);
  } catch (err) {
    addMessage("error", err.message);
  } finally {
    drop.classList.remove("busy");
    drop.querySelector("strong").textContent = "Upload documents";
  }
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// Small, safe Markdown subset: paragraphs, bullet lists, **bold** and [n] citations.
function renderAnswer(text) {
  const inline = (line) =>
    escapeHtml(line)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\[(\d+)\]/g, '<sup class="cite">$1</sup>');
  const html = [];
  let list = null;
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    const item = line.match(/^(?:[-*•]|\d+\.)\s+(.*)$/);
    if (item) {
      list = list || [];
      list.push(`<li>${inline(item[1])}</li>`);
      continue;
    }
    if (list) {
      html.push(`<ul>${list.join("")}</ul>`);
      list = null;
    }
    if (line) html.push(`<p>${inline(line)}</p>`);
  }
  if (list) html.push(`<ul>${list.join("")}</ul>`);
  return html.join("");
}

function renderSources(sources) {
  if (!sources.length) return "";
  const items = sources
    .map(
      (s) => `<li><span class="cite">${s.n}</span><div><strong>${escapeHtml(s.source)}</strong> · page ${s.page}
        <p>${escapeHtml(s.text.slice(0, 260))}${s.text.length > 260 ? "…" : ""}</p></div></li>`
    )
    .join("");
  return `<details class="sources"><summary>Sources (${sources.length})</summary><ul>${items}</ul></details>`;
}

function addMessage(role, html) {
  $(".welcome")?.remove();
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = html;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}

async function ask(question) {
  addMessage("user", `<p>${escapeHtml(question)}</p>`);
  const pending = addMessage("bot pending", "<p>Thinking…</p>");
  sendButton.disabled = true;
  try {
    const { answer, sources } = await api(`/api/sessions/${sessionId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question }),
    });
    pending.className = "msg bot";
    pending.innerHTML = renderAnswer(answer) + renderSources(sources);
  } catch (err) {
    pending.className = "msg error";
    pending.innerHTML = `<p>${escapeHtml(err.message)}</p>`;
  } finally {
    sendButton.disabled = false;
    messages.scrollTop = messages.scrollHeight;
  }
}

$("#form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || sendButton.disabled) return;
  input.value = "";
  input.style.height = "";
  ask(question);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#form").requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
});

messages.addEventListener("click", (event) => {
  const example = event.target.closest("[data-q]");
  if (example) ask(example.dataset.q);
});

$("#file").addEventListener("change", (event) => {
  upload(event.target.files);
  event.target.value = "";
});

const drop = $("#drop");
["dragenter", "dragover"].forEach((type) =>
  drop.addEventListener(type, (event) => {
    event.preventDefault();
    drop.classList.add("over");
  })
);
["dragleave", "drop"].forEach((type) => drop.addEventListener(type, () => drop.classList.remove("over")));
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  upload(event.dataTransfer.files);
});

$("#reset").addEventListener("click", async () => {
  await api(`/api/sessions/${sessionId}`, { method: "DELETE" }).catch(() => {});
  sessionStorage.removeItem("docchat-session");
  location.reload();
});

startSession().catch((err) => addMessage("error", `<p>${escapeHtml(err.message)}</p>`));

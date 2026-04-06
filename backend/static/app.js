let currentTaskId = null;
let pollTimer = null;

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMarkdownBasic(md) {
  // Minimal markdown -> HTML (enough for this demo: headings, bold, code blocks).
  if (!md) return "";
  let html = escapeHtml(md);

  html = html.replaceAll(/```/g, "```"); // keep fences
  html = html.replaceAll(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  html = html.replaceAll(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replaceAll(/\n/g, "<br />");
  return html;
}

async function createTask() {
  const goal = document.getElementById("goal").value.trim();
  const enableCodeRun = document.getElementById("enableCodeRun").checked;
  const maxSteps = parseInt(document.getElementById("maxSteps").value || "6", 10);

  const submitStatus = document.getElementById("submitStatus");
  submitStatus.textContent = "Creating task...";

  const resp = await fetch("/v1/tasks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ goal, enable_code_run: enableCodeRun, max_steps: maxSteps }),
  });

  if (!resp.ok) {
    submitStatus.textContent = "Error: " + (await resp.text());
    return;
  }

  const data = await resp.json();
  currentTaskId = data.task_id;

  document.getElementById("taskId").textContent = currentTaskId;
  document.getElementById("taskStatus").textContent = data.status;
  document.getElementById("taskResult").innerHTML = "";
  document.getElementById("steps").innerHTML = "";

  submitStatus.textContent = "Task created. Running...";
  startPolling();
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);

  pollTimer = setInterval(async () => {
    if (!currentTaskId) return;
    const resp = await fetch(`/v1/tasks/${currentTaskId}`);
    if (!resp.ok) return;
    const data = await resp.json();

    document.getElementById("taskStatus").textContent = data.status;

    const stepsDiv = document.getElementById("steps");
    stepsDiv.innerHTML = "";
    if (data.steps && data.steps.length) {
      data.steps.forEach((s) => {
        const wrap = document.createElement("div");
        wrap.className = "muted";
        wrap.style.marginTop = "10px";
        wrap.innerHTML =
          `<b>${escapeHtml(s.agent)}: ${escapeHtml(s.step_type)}</b> ` +
          `(<span>${escapeHtml(s.status)}</span>)` +
          `<div style="margin-top:6px">${s.output ? renderMarkdownBasic(s.output) : ""}</div>` +
          `${s.error ? `<div style="margin-top:6px;color:#b00"><b>Error:</b> ${escapeHtml(s.error)}</div>` : ""}`;
        stepsDiv.appendChild(wrap);
      });
    }

    if (data.result) {
      const resultDiv = document.getElementById("taskResult");
      resultDiv.innerHTML = renderMarkdownBasic(data.result);
    }

    if (["succeeded", "failed"].includes(data.status)) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }, 1000);
}

document.getElementById("submitBtn").addEventListener("click", createTask);


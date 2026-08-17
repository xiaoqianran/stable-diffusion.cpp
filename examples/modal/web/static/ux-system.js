import { $, api, escapeHtml, queueLabel, state, systemDialog, toast } from "./ux-core.js";

export async function openSystem() {
  try {
    const doctor = state.doctor || await api("/api/doctor");
    state.doctor = doctor;
    const q = state.queue || {};
    const checks = doctor.checks || [];
    systemDialog.innerHTML = `<div class="system-dialog-content"><div class="system-dialog-head"><div><p class="eyebrow">SYSTEM</p><h2>运行状态</h2></div><button class="dialog-close" type="button" aria-label="关闭">×</button></div><div class="system-list"><div class="system-row"><span>GPU</span><b>${escapeHtml(queueLabel.textContent)}</b></div><div class="system-row"><span>等待任务</span><b>${q.queue_length || 0}</b></div><div class="system-row"><span>运行任务</span><b>${q.running_count || 0}</b></div><div class="system-row"><span>同模型亲和</span><b>${state.meta?.runtime?.same_model_affinity ? "已启用" : "关闭"}</b></div>${checks.map((check) => `<div class="system-row"><span>${escapeHtml(check.name)}</span><b class="${check.ok ? "" : "bad"}">${check.ok ? "正常" : "需要注意"} · ${escapeHtml(check.detail)}</b></div>`).join("")}</div></div>`;
    $(".dialog-close", systemDialog).addEventListener("click", () => systemDialog.close());
    systemDialog.showModal();
  } catch (error) {
    toast(error.message, "bad");
  }
}

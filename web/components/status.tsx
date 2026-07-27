import { STAGES, STATUS, type Job } from "@/lib/types";

export const STATUS_LABEL: Record<string, { icon: string; text: string; cls: string }> = {
  [STATUS.QUEUED]: { icon: "○", text: "queued", cls: "text-zinc-400 bg-zinc-800/60" },
  [STATUS.RUNNING]: { icon: "◐", text: "running", cls: "text-sky-300 bg-sky-500/10" },
  [STATUS.AWAITING]: { icon: "❚❚", text: "needs you", cls: "text-amber-300 bg-amber-500/10" },
  [STATUS.DONE]: { icon: "✓", text: "done", cls: "text-emerald-300 bg-emerald-500/10" },
  [STATUS.FAILED]: { icon: "✕", text: "failed", cls: "text-rose-300 bg-rose-500/10" },
  [STATUS.CANCELLED]: { icon: "⊘", text: "cancelled", cls: "text-zinc-500 bg-zinc-800/60" },
};

export function StatusBadge({ status }: { status: string }) {
  const s = STATUS_LABEL[status] ?? {
    icon: "·",
    text: status,
    cls: "text-zinc-400 bg-zinc-800/60",
  };
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ${s.cls}`}
    >
      <span aria-hidden>{s.icon}</span>
      {s.text}
    </span>
  );
}

/** scout → lens → quill, with each stage's state. */
export function StageBar({ job }: { job: Job }) {
  const done =
    job.stage === "done"
      ? [...STAGES]
      : STAGES.slice(0, Math.max(0, STAGES.indexOf(job.stage as (typeof STAGES)[number])));

  return (
    <div className="flex items-center gap-1.5 text-xs">
      {STAGES.map((stage, i) => {
        const isDone = done.includes(stage);
        const isCurrent = stage === job.stage;
        const failed = isCurrent && job.status === STATUS.FAILED;
        const cls = isDone
          ? "text-emerald-300 border-emerald-500/30 bg-emerald-500/10"
          : failed
            ? "text-rose-300 border-rose-500/30 bg-rose-500/10"
            : isCurrent
              ? "text-sky-300 border-sky-500/30 bg-sky-500/10"
              : "text-zinc-500 border-zinc-700/60";
        return (
          <span key={stage} className="flex items-center gap-1.5">
            {i > 0 && <span className="text-zinc-700">→</span>}
            <span className={`rounded border px-2 py-0.5 font-mono ${cls}`}>
              {isDone ? "✓ " : failed ? "✕ " : ""}
              {stage}
            </span>
          </span>
        );
      })}
    </div>
  );
}

export function relativeTime(epochSeconds: number): string {
  const d = new Date(epochSeconds * 1000);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

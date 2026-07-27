"use client";

import Link from "next/link";
import { useState } from "react";
import useSWR from "swr";

import { StageBar, StatusBadge, relativeTime } from "@/components/status";
import { STATUS, type Job } from "@/lib/types";

const MODELS = [
  "gpt2",
  "gpt2-medium",
  "gpt2-large",
  "gpt2-xl",
  "pythia-70m",
  "pythia-160m",
  "pythia-410m",
  "gpt-neo-125m",
];

const fetcher = (url: string) => fetch(url).then((r) => r.json());

export default function JobsPage() {
  const { data, mutate } = useSWR<{ jobs: Job[] }>("/api/jobs", fetcher, {
    refreshInterval: 3000,
  });
  const [open, setOpen] = useState(false);
  const [question, setQuestion] = useState("");
  const [model, setModel] = useState("gpt2");
  const [autoApprove, setAutoApprove] = useState(false);
  const [busy, setBusy] = useState(false);

  const jobs = data?.jobs ?? [];
  const count = (s: string) => jobs.filter((j) => j.status === s).length;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    setBusy(true);
    await fetch("/api/jobs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question, model, autoApprove }),
    });
    setBusy(false);
    setQuestion("");
    setOpen(false);
    mutate();
  }

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <header className="mb-8 flex items-baseline justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Research Jobs</h1>
          <p className="mt-1 text-sm text-zinc-400">Scout → Lens → Quill</p>
        </div>
        <button
          onClick={() => setOpen((v) => !v)}
          className="rounded-md bg-zinc-100 px-3 py-1.5 text-sm font-medium text-zinc-900 hover:bg-white"
        >
          {open ? "Cancel" : "New job"}
        </button>
      </header>

      {open && (
        <form
          onSubmit={submit}
          className="mb-8 space-y-3 rounded-lg border border-zinc-800 bg-zinc-900/40 p-4"
        >
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="What attention heads mediate indirect object identification in GPT-2 Small?"
            rows={3}
            className="w-full rounded-md border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm outline-none focus:border-zinc-600"
          />
          <div className="flex flex-wrap items-center gap-4">
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="rounded-md border border-zinc-800 bg-zinc-950 px-2 py-1.5 text-sm"
            >
              {MODELS.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
            <label className="flex items-center gap-2 text-sm text-zinc-400">
              <input
                type="checkbox"
                checked={autoApprove}
                onChange={(e) => setAutoApprove(e.target.checked)}
              />
              Run all stages without stopping
            </label>
            <button
              disabled={busy || !question.trim()}
              className="ml-auto rounded-md bg-sky-500 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-40"
            >
              {busy ? "Starting…" : "Start job"}
            </button>
          </div>
          <p className="text-xs text-zinc-500">
            This only queues the job — a worker has to be polling for it to run:{" "}
            <code className="text-zinc-400">
              python -m orchestrator.src.worker --poll
            </code>
          </p>
        </form>
      )}

      <div className="mb-6 flex gap-6 text-sm">
        <Stat label="Jobs" value={jobs.length} />
        <Stat label="Running" value={count(STATUS.RUNNING) + count(STATUS.QUEUED)} />
        <Stat label="Needs you" value={count(STATUS.AWAITING)} />
        <Stat label="Done" value={count(STATUS.DONE)} />
        <Stat label="Failed" value={count(STATUS.FAILED)} />
      </div>

      {jobs.length === 0 ? (
        <p className="rounded-lg border border-dashed border-zinc-800 p-8 text-center text-sm text-zinc-500">
          No jobs yet.
        </p>
      ) : (
        <ul className="space-y-2">
          {jobs.map((job) => (
            <li key={job.id}>
              <Link
                href={`/jobs/${job.id}`}
                className="block rounded-lg border border-zinc-800 bg-zinc-900/40 p-4 transition hover:border-zinc-700 hover:bg-zinc-900"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <code className="text-xs text-zinc-500">{job.id}</code>
                      <StatusBadge status={job.status} />
                    </div>
                    <p className="mt-1.5 truncate text-sm">{job.question}</p>
                    <div className="mt-2">
                      <StageBar job={job} />
                    </div>
                  </div>
                  <div className="shrink-0 text-right text-xs text-zinc-500">
                    <div className="font-mono">{job.model}</div>
                    <div className="mt-1">{relativeTime(job.created_at)}</div>
                  </div>
                </div>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="text-xl font-semibold">{value}</div>
      <div className="text-xs text-zinc-500">{label}</div>
    </div>
  );
}

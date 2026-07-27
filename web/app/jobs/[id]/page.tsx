"use client";

import Link from "next/link";
import { use, useState } from "react";
import ReactMarkdown from "react-markdown";
import useSWR from "swr";

import { StageBar, StatusBadge, relativeTime } from "@/components/status";
import { STATUS, type Job } from "@/lib/types";

const fetcher = (url: string) => fetch(url).then((r) => r.json());
const textFetcher = (url: string) =>
  fetch(url).then((r) => (r.ok ? r.text() : ""));
const jsonFetcher = (url: string) =>
  fetch(url).then((r) => (r.ok ? r.json() : null));

const GATE_PROMPT: Record<string, string> = {
  lens: "Approve Scout's research plan and run Lens experiments?",
  quill: "Send these experiment results to Quill for critique?",
};

type Tab = "plan" | "results" | "critique" | "log";

type ExperimentResult = {
  name: string;
  tool: string;
  status: string;
  summary?: string;
  error?: string;
  plot_paths?: string[];
};

export default function JobDetail({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const { data, mutate } = useSWR<{ job: Job }>(`/api/jobs/${id}`, fetcher, {
    refreshInterval: 3000,
  });
  const [tab, setTab] = useState<Tab>("plan");
  const job = data?.job;

  async function act(action: string) {
    await fetch(`/api/jobs/${id}/action`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ action }),
    });
    mutate();
  }

  if (!job) {
    return (
      <main className="mx-auto max-w-4xl px-6 py-10 text-sm text-zinc-500">
        Loading…
      </main>
    );
  }

  const active = job.status === STATUS.RUNNING || job.status === STATUS.QUEUED;

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <Link href="/" className="text-sm text-zinc-500 hover:text-zinc-300">
        ← All jobs
      </Link>

      <header className="mt-4 mb-6">
        <div className="flex items-center gap-2">
          <code className="text-xs text-zinc-500">{job.id}</code>
          <StatusBadge status={job.status} />
        </div>
        <h1 className="mt-2 text-xl font-semibold">{job.question}</h1>
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <StageBar job={job} />
          <span className="font-mono text-xs text-zinc-500">{job.model}</span>
          {job.auto_approve ? (
            <span className="text-xs text-zinc-500">auto-approve on</span>
          ) : null}
          <span className="text-xs text-zinc-500">
            {relativeTime(job.created_at)}
          </span>
        </div>
      </header>

      {job.error && (
        <pre className="mb-6 overflow-x-auto rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-200">
          {job.error}
        </pre>
      )}

      {job.status === STATUS.AWAITING && (
        <div className="mb-6 rounded-lg border border-amber-500/30 bg-amber-500/10 p-4">
          <p className="text-sm text-amber-100">
            {GATE_PROMPT[job.stage] ?? `Approve and run ${job.stage}?`}
          </p>
          <div className="mt-3 flex gap-2">
            <button
              onClick={() => act("approve")}
              className="rounded-md bg-emerald-500 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-400"
            >
              Approve
            </button>
            <button
              onClick={() => act("cancel")}
              className="rounded-md border border-zinc-700 px-3 py-1.5 text-sm hover:bg-zinc-800"
            >
              Stop here
            </button>
          </div>
        </div>
      )}

      {job.status === STATUS.FAILED && (
        <button
          onClick={() => act("retry")}
          className="mb-6 rounded-md bg-sky-500 px-3 py-1.5 text-sm font-medium text-white hover:bg-sky-400"
        >
          Retry stage
        </button>
      )}

      {active && (
        <button
          onClick={() => act("cancel")}
          className="mb-6 rounded-md border border-zinc-700 px-3 py-1.5 text-sm hover:bg-zinc-800"
        >
          Cancel
        </button>
      )}

      <nav className="mb-4 flex gap-1 border-b border-zinc-800">
        {(
          [
            ["plan", "Research plan"],
            ["results", "Experiment results"],
            ["critique", "Critique"],
            ["log", "Worker log"],
          ] as [Tab, string][]
        ).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`-mb-px border-b-2 px-3 py-2 text-sm ${
              tab === key
                ? "border-zinc-200 text-zinc-100"
                : "border-transparent text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {label}
          </button>
        ))}
      </nav>

      {tab === "plan" && (
        <MarkdownArtifact
          url={job.plan_path ? `/api/artifact?job=${job.id}&kind=plan` : null}
          empty="No plan yet — Scout hasn't finished."
        />
      )}
      {tab === "results" && <Results job={job} />}
      {tab === "critique" && (
        <MarkdownArtifact
          url={job.report_path ? `/api/artifact?job=${job.id}&kind=report` : null}
          empty="No critique yet — Quill hasn't finished."
        />
      )}
      {tab === "log" && (
        <pre className="max-h-[32rem] overflow-auto rounded-lg border border-zinc-800 bg-zinc-950 p-3 text-xs leading-relaxed text-zinc-400">
          {job.log_tail || "(no output mirrored yet)"}
        </pre>
      )}
    </main>
  );
}

function MarkdownArtifact({
  url,
  empty,
}: {
  url: string | null;
  empty: string;
}) {
  const { data } = useSWR(url, textFetcher);
  if (!url) return <p className="text-sm text-zinc-500">{empty}</p>;
  if (data === undefined)
    return <p className="text-sm text-zinc-500">Loading…</p>;
  return (
    <article className="prose-invert max-w-none space-y-3 text-sm leading-relaxed text-zinc-300 [&_a]:text-sky-400 [&_code]:text-zinc-200 [&_h1]:mt-6 [&_h1]:text-lg [&_h1]:font-semibold [&_h2]:mt-5 [&_h2]:text-base [&_h2]:font-semibold [&_h3]:mt-4 [&_h3]:font-medium [&_li]:ml-5 [&_li]:list-disc [&_strong]:text-zinc-100">
      <ReactMarkdown>{data}</ReactMarkdown>
    </article>
  );
}

function Results({ job }: { job: Job }) {
  const { data } = useSWR(
    job.bundle_path ? `/api/artifact?job=${job.id}&kind=bundle` : null,
    jsonFetcher,
  );

  if (!job.bundle_path)
    return (
      <p className="text-sm text-zinc-500">
        No results yet — Lens hasn&apos;t finished.
      </p>
    );
  if (!data) return <p className="text-sm text-zinc-500">Loading…</p>;

  const results: ExperimentResult[] = data.results ?? [];

  return (
    <div className="space-y-3">
      <p className="text-sm text-zinc-400">
        <span className="text-lg font-semibold text-zinc-100">
          {data.n_success}/{data.n_total}
        </span>{" "}
        experiments succeeded
      </p>
      {results.map((r, i) => {
        const ok = r.status === "success";
        return (
          <details
            key={i}
            className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3"
          >
            <summary className="cursor-pointer text-sm">
              <span className={ok ? "text-emerald-400" : "text-rose-400"}>
                {ok ? "✓" : "✕"}
              </span>{" "}
              {r.name}{" "}
              <code className="text-xs text-zinc-500">({r.tool})</code>
            </summary>
            <div className="mt-3 space-y-3">
              {ok ? (
                <>
                  <p className="text-sm text-zinc-300">
                    {r.summary || "(no summary)"}
                  </p>
                  {(r.plot_paths ?? []).map((p) => (
                    <img
                      key={p}
                      src={`/api/artifact?file=${encodeURIComponent(p)}`}
                      alt={r.name}
                      className="w-full rounded border border-zinc-800"
                    />
                  ))}
                </>
              ) : (
                <pre className="overflow-x-auto text-xs text-rose-300">
                  {r.error || "unknown error"}
                </pre>
              )}
            </div>
          </details>
        );
      })}
    </div>
  );
}

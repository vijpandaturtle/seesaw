import { NextResponse } from "next/server";

import { createJob, listJobs } from "@/lib/db";

export const runtime = "nodejs";

export async function GET() {
  return NextResponse.json({ jobs: listJobs() });
}

/**
 * Creating a job only inserts a queued row — a worker claims it. Nothing is
 * spawned here, which is what lets the UI be deployed away from the agents.
 * Run `python -m orchestrator.src.worker --poll` for anything to pick it up.
 */
export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  const question = typeof body?.question === "string" ? body.question.trim() : "";
  if (!question) {
    return NextResponse.json({ error: "question is required" }, { status: 400 });
  }
  const job = createJob(
    question,
    typeof body?.model === "string" && body.model ? body.model : "gpt2",
    Boolean(body?.autoApprove),
  );
  return NextResponse.json({ job }, { status: 201 });
}

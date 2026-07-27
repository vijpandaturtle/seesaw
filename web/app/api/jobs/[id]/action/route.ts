import { NextResponse } from "next/server";

import { approveJob, cancelJob, getJob, retryJob } from "@/lib/db";

export const runtime = "nodejs";

type Ctx = { params: Promise<{ id: string }> };

/**
 * approve / retry / cancel.
 *
 * approve and retry are guarded on the job's current status inside the UPDATE,
 * so a click on a stale page can't re-queue something a worker already has.
 */
export async function POST(request: Request, { params }: Ctx) {
  const { id } = await params;
  const { action } = (await request.json().catch(() => ({}))) as {
    action?: string;
  };

  if (!getJob(id)) {
    return NextResponse.json({ error: "not found" }, { status: 404 });
  }

  switch (action) {
    case "approve":
      if (!approveJob(id)) {
        return NextResponse.json(
          { error: "job is no longer awaiting approval" },
          { status: 409 },
        );
      }
      break;
    case "retry":
      if (!retryJob(id)) {
        return NextResponse.json(
          { error: "job is not in a failed state" },
          { status: 409 },
        );
      }
      break;
    case "cancel":
      cancelJob(id);
      break;
    default:
      return NextResponse.json(
        { error: "action must be approve, retry, or cancel" },
        { status: 400 },
      );
  }

  return NextResponse.json({ job: getJob(id) });
}

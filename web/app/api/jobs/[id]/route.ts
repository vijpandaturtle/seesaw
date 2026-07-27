import { NextResponse } from "next/server";

import { deleteJob, getJob } from "@/lib/db";

export const runtime = "nodejs";

type Ctx = { params: Promise<{ id: string }> };

export async function GET(_request: Request, { params }: Ctx) {
  const { id } = await params;
  const job = getJob(id);
  if (!job) return NextResponse.json({ error: "not found" }, { status: 404 });
  return NextResponse.json({ job });
}

export async function DELETE(_request: Request, { params }: Ctx) {
  const { id } = await params;
  deleteJob(id);
  return NextResponse.json({ ok: true });
}

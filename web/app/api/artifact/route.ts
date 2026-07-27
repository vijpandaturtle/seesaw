import { readFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

import { OUTPUTS_DIR, getJob } from "@/lib/db";

export const runtime = "nodejs";

const KINDS = {
  plan: "plan_path",
  bundle: "bundle_path",
  report: "report_path",
} as const;

/**
 * Serve a run artifact.
 *
 * Two addressing modes, both of which resolve to a path under outputs/ and
 * refuse anything that escapes it:
 *   ?job=<id>&kind=plan|bundle|report   — path comes from the job row
 *   ?file=<path>                        — for plots referenced inside a bundle
 */
export async function GET(request: Request) {
  const url = new URL(request.url);
  const kind = url.searchParams.get("kind");
  const jobId = url.searchParams.get("job");
  const file = url.searchParams.get("file");

  let target: string | null = null;

  if (jobId && kind && kind in KINDS) {
    const job = getJob(jobId);
    target = job?.[KINDS[kind as keyof typeof KINDS]] ?? null;
  } else if (file) {
    target = file;
  }

  if (!target) {
    return NextResponse.json({ error: "artifact not found" }, { status: 404 });
  }

  // Anything outside outputs/ is out of bounds, however it was addressed.
  const resolved = path.resolve(target);
  const root = path.resolve(OUTPUTS_DIR);
  if (resolved !== root && !resolved.startsWith(root + path.sep)) {
    return NextResponse.json({ error: "forbidden" }, { status: 403 });
  }

  let data: Buffer;
  try {
    data = await readFile(resolved);
  } catch {
    return NextResponse.json({ error: "artifact missing on disk" }, { status: 404 });
  }

  const ext = path.extname(resolved).toLowerCase();
  const type =
    ext === ".png"
      ? "image/png"
      : ext === ".json"
        ? "application/json"
        : ext === ".md"
          ? "text/markdown; charset=utf-8"
          : "application/octet-stream";

  return new NextResponse(new Uint8Array(data), {
    headers: { "content-type": type, "cache-control": "no-store" },
  });
}

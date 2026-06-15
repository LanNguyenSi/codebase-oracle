// Worker instantiation and job handler dispatch for queue-toy.
//
// Creates a BullMQ Worker that listens on the main queue and dispatches
// each job to a typed handler based on job.name. Unknown job types are
// logged and skipped so the queue never stalls on unrecognised entries.

import { Worker, type Job } from "bullmq";
import IORedis from "ioredis";
import type { EmailJobData, ReportJobData } from "./queue.js";

const connection = new IORedis(process.env.REDIS_URL ?? "redis://localhost:6379", {
  maxRetriesPerRequest: null,
});

async function handleSendEmail(job: Job<EmailJobData>): Promise<void> {
  const { to, subject, body } = job.data;
  // In production this would call an email provider SDK.
  console.log(`[worker] sending email to=${to} subject="${subject}" body="${body}"`);
}

async function handleGenerateReport(job: Job<ReportJobData>): Promise<void> {
  const { reportId, format } = job.data;
  console.log(`[worker] generating report reportId=${reportId} format=${format}`);
  // Simulate work.
  await new Promise((r) => setTimeout(r, 100));
}

export const worker = new Worker(
  "main",
  async (job) => {
    switch (job.name) {
      case "send-email":
        return handleSendEmail(job as Job<EmailJobData>);
      case "generate-report":
        return handleGenerateReport(job as Job<ReportJobData>);
      default:
        console.warn(`[worker] unknown job type: ${job.name}`);
    }
  },
  { connection, concurrency: 4 },
);

worker.on("completed", (job) => {
  console.log(`[worker] job ${job.id} (${job.name}) completed`);
});

worker.on("failed", (job, err) => {
  console.error(`[worker] job ${job?.id} (${job?.name}) failed: ${err.message}`);
});

// Queue and scheduler setup for queue-toy.
//
// Exports a shared Queue instance connected to Redis via REDIS_URL.
// Default job options (attempts, backoff) are set at queue level so
// individual job additions do not have to repeat them.

import { Queue, QueueScheduler } from "bullmq";
import IORedis from "ioredis";

const QUEUE_NAME = "main";

function createConnection(): IORedis {
  const url = process.env.REDIS_URL ?? "redis://localhost:6379";
  return new IORedis(url, { maxRetriesPerRequest: null });
}

const connection = createConnection();

export const mainQueue = new Queue(QUEUE_NAME, {
  connection,
  defaultJobOptions: {
    attempts: 3,
    backoff: { type: "exponential", delay: 1000 },
    removeOnComplete: 100,
    removeOnFail: 200,
  },
});

// QueueScheduler is required for delayed jobs and auto-retries.
export const scheduler = new QueueScheduler(QUEUE_NAME, { connection });

export interface EmailJobData {
  to: string;
  subject: string;
  body: string;
}

export interface ReportJobData {
  reportId: string;
  format: "pdf" | "csv";
}

export async function enqueueEmail(data: EmailJobData): Promise<void> {
  await mainQueue.add("send-email", data);
}

export async function enqueueReport(data: ReportJobData): Promise<void> {
  await mainQueue.add("generate-report", data, { priority: 5 });
}

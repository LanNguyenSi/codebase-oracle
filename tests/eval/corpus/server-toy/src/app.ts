// Fastify app factory for server-toy.
//
// Builds and returns a configured Fastify instance. Registers the cors and
// sensible plugins then mounts the /items route group.  Call .listen() on
// the returned instance to start the server.

import Fastify from "fastify";
import cors from "@fastify/cors";
import sensible from "@fastify/sensible";
import { itemRoutes } from "./routes.js";

export interface AppOptions {
  logger?: boolean;
  port?: number;
}

export async function buildApp(options: AppOptions = {}): Promise<ReturnType<typeof Fastify>> {
  const app = Fastify({ logger: options.logger ?? false });

  await app.register(cors, { origin: "*" });
  await app.register(sensible);

  // Mount route groups under versioned prefix.
  await app.register(itemRoutes, { prefix: "/api/v1" });

  app.get("/health", async (_req, reply) => {
    return reply.send({ status: "ok" });
  });

  return app;
}

export async function startServer(options: AppOptions = {}): Promise<void> {
  const app = await buildApp(options);
  const port = options.port ?? Number(process.env.PORT) ?? 3000;
  await app.listen({ port, host: "0.0.0.0" });
  app.log.info(`server listening on port ${port}`);
}

// Route definitions for the /items resource in server-toy.
//
// Each route handler is a typed async function.  Validation is done via
// JSON Schema on the request body so Fastify can coerce and reject early
// before the handler runs.

import type { FastifyInstance } from "fastify";

interface Item {
  id: string;
  name: string;
  createdAt: string;
}

const items: Item[] = [];

interface CreateItemBody {
  name: string;
}

export async function itemRoutes(app: FastifyInstance): Promise<void> {
  // GET /items - return all items
  app.get("/items", async (_req, reply) => {
    return reply.send(items);
  });

  // GET /items/:id - return a single item or 404
  app.get<{ Params: { id: string } }>("/items/:id", async (req, reply) => {
    const item = items.find((i) => i.id === req.params.id);
    if (!item) return reply.notFound(`item ${req.params.id} not found`);
    return reply.send(item);
  });

  // POST /items - create a new item
  app.post<{ Body: CreateItemBody }>(
    "/items",
    {
      schema: {
        body: {
          type: "object",
          required: ["name"],
          properties: { name: { type: "string", minLength: 1 } },
        },
      },
    },
    async (req, reply) => {
      const newItem: Item = {
        id: String(Date.now()),
        name: req.body.name,
        createdAt: new Date().toISOString(),
      };
      items.push(newItem);
      return reply.code(201).send(newItem);
    },
  );
}

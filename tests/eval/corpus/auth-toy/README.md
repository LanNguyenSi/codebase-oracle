# auth-toy

Tiny JWT auth surface used as eval fixture for codebase-oracle.

The flow is the boring textbook one:

1. A client posts credentials to `/login`. The server signs a JWT with the
   `JWT_SECRET` environment variable and returns it.
2. On subsequent requests, the client sends `Authorization: Bearer <token>`.
3. The `requireAuth` middleware extracts the token, verifies it against
   `JWT_SECRET`, and attaches the decoded subject to `req.user`.

Two real source files: `src/token.ts` (signing + verifying) and
`src/middleware.ts` (Express-style auth gate). Nothing here is meant
to be used in production; it exists to give the eval corpus a
recognisable auth shape.

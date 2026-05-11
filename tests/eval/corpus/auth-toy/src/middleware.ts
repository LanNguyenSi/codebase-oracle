// Express-style auth middleware for auth-toy.
//
// Extracts a Bearer token from the Authorization header, verifies it with
// verifyAccessToken, and attaches the decoded subject to req.user. Returns
// 401 on missing or invalid tokens.

import type { NextFunction, Request, Response } from "express";
import { verifyAccessToken } from "./token.js";

export interface AuthenticatedRequest extends Request {
  user?: { id: string };
}

export function requireAuth(
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction,
): void {
  const header = req.headers.authorization;
  if (!header || !header.startsWith("Bearer ")) {
    res.status(401).json({ error: "missing_bearer_token" });
    return;
  }
  const token = header.slice("Bearer ".length);
  try {
    const decoded = verifyAccessToken(token);
    req.user = { id: decoded.sub };
    next();
  } catch {
    res.status(401).json({ error: "invalid_token" });
  }
}

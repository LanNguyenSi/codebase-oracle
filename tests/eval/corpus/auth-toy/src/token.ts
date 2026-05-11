// JWT signing and verification for auth-toy.
//
// The signing key is the JWT_SECRET environment variable. In production
// this would come from a secrets manager; here it is read once at module
// load and cached so tests can stub process.env before importing.

import jwt from "jsonwebtoken";

const TOKEN_TTL_SECONDS = 60 * 60; // 1 hour

let cachedSecret: string | null = null;

function getJwtSecret(): string {
  if (cachedSecret) return cachedSecret;
  const secret = process.env.JWT_SECRET;
  if (!secret) {
    throw new Error("JWT_SECRET environment variable is required to sign tokens.");
  }
  cachedSecret = secret;
  return cachedSecret;
}

export interface AccessTokenPayload {
  sub: string;
  iat?: number;
  exp?: number;
}

export function signAccessToken(payload: { sub: string }): string {
  return jwt.sign(payload, getJwtSecret(), {
    expiresIn: TOKEN_TTL_SECONDS,
  });
}

export function verifyAccessToken(token: string): AccessTokenPayload {
  return jwt.verify(token, getJwtSecret()) as AccessTokenPayload;
}

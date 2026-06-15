// Typed query helpers for db-toy.
//
// Wraps Prisma client calls for the User and Post models.  All helpers
// return strongly-typed objects so callers never deal with raw PrismaClient
// return types. The singleton client is created once and reused.

import { PrismaClient, type User, type Post, type Role } from "@prisma/client";

const prisma = new PrismaClient();

export async function findUserByEmail(email: string): Promise<User | null> {
  return prisma.user.findUnique({ where: { email } });
}

export async function createUser(data: {
  email: string;
  name?: string;
  role?: Role;
}): Promise<User> {
  return prisma.user.create({ data });
}

export async function listPublishedPosts(options: {
  authorId?: string;
  limit?: number;
  offset?: number;
}): Promise<Post[]> {
  return prisma.post.findMany({
    where: {
      published: true,
      ...(options.authorId ? { authorId: options.authorId } : {}),
    },
    orderBy: { publishedAt: "desc" },
    take: options.limit ?? 20,
    skip: options.offset ?? 0,
  });
}

export async function publishPost(postId: string): Promise<Post> {
  return prisma.post.update({
    where: { id: postId },
    data: { published: true, publishedAt: new Date() },
  });
}

export async function deleteUser(userId: string): Promise<void> {
  await prisma.user.delete({ where: { id: userId } });
}

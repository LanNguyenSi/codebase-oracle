import { describe, it, expect } from "vitest";
import { discoverRepos, walkRepo } from "../../src/ingest/scanner.js";
import { join } from "node:path";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";

describe("discoverRepos", () => {
  it("discovers directories with .git", async () => {
    const root = await mkdtemp(join(tmpdir(), "oracle-test-"));
    try {
      await mkdir(join(root, "repo-a", ".git"), { recursive: true });
      await mkdir(join(root, "repo-b", ".git"), { recursive: true });
      await mkdir(join(root, "not-a-repo"), { recursive: true });

      const repos = await discoverRepos(root);
      expect(repos.map((r) => r.name).sort()).toEqual(["repo-a", "repo-b"]);
    } finally {
      await rm(root, { recursive: true });
    }
  });

  it("skips hidden directories", async () => {
    const root = await mkdtemp(join(tmpdir(), "oracle-test-"));
    try {
      await mkdir(join(root, ".hidden-repo", ".git"), { recursive: true });
      await mkdir(join(root, "visible-repo", ".git"), { recursive: true });

      const repos = await discoverRepos(root);
      expect(repos.map((r) => r.name)).toEqual(["visible-repo"]);
    } finally {
      await rm(root, { recursive: true });
    }
  });
});

describe("walkRepo", () => {
  it("yields .ts and .md files, skips node_modules", async () => {
    const root = await mkdtemp(join(tmpdir(), "oracle-test-"));
    const repo = join(root, "test-repo");
    try {
      await mkdir(join(repo, "src"), { recursive: true });
      await mkdir(join(repo, "node_modules", "pkg"), { recursive: true });
      await mkdir(join(repo, ".git"), { recursive: true });

      await writeFile(join(repo, "src", "index.ts"), "export const x = 1;");
      await writeFile(join(repo, "README.md"), "# Hello");
      await writeFile(join(repo, "node_modules", "pkg", "index.ts"), "// should be skipped");
      await writeFile(join(repo, "src", "image.png"), "binary");

      const files: string[] = [];
      for await (const file of walkRepo(repo, "test-repo", root)) {
        files.push(file.relativePath);
      }

      expect(files.sort()).toEqual([
        "test-repo/README.md",
        "test-repo/src/index.ts",
      ]);
    } finally {
      await rm(root, { recursive: true });
    }
  });

  it("skips empty files", async () => {
    const root = await mkdtemp(join(tmpdir(), "oracle-test-"));
    const repo = join(root, "test-repo");
    try {
      await mkdir(join(repo, ".git"), { recursive: true });
      await writeFile(join(repo, "empty.ts"), "");
      await writeFile(join(repo, "valid.ts"), "const x = 1;");

      const files: string[] = [];
      for await (const file of walkRepo(repo, "test-repo", root)) {
        files.push(file.relativePath);
      }

      expect(files).toEqual(["test-repo/valid.ts"]);
    } finally {
      await rm(root, { recursive: true });
    }
  });

  it("includes metadata in scanned files", async () => {
    const root = await mkdtemp(join(tmpdir(), "oracle-test-"));
    const repo = join(root, "my-repo");
    try {
      await mkdir(join(repo, ".git"), { recursive: true });
      await writeFile(join(repo, "app.tsx"), "export default function App() {}");

      const files = [];
      for await (const file of walkRepo(repo, "my-repo", root)) {
        files.push(file);
      }

      expect(files).toHaveLength(1);
      expect(files[0].repo).toBe("my-repo");
      expect(files[0].language).toBe("tsx");
      expect(files[0].content).toContain("App");
      expect(typeof files[0].contentHash).toBe("string");
      expect(files[0].contentHash.length).toBe(64);
    } finally {
      await rm(root, { recursive: true });
    }
  });
});

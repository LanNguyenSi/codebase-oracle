import { describe, it, expect } from "vitest";
import {
  DEFAULT_MAX_FILE_SIZE_BYTES,
  discoverRepos,
  walkRepo,
  type SkippedFile,
} from "../../src/ingest/scanner.js";
import { join } from "node:path";
import { mkdtemp, mkdir, writeFile, rm, chmod } from "node:fs/promises";
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

  it("includes python and yaml files by default", async () => {
    const root = await mkdtemp(join(tmpdir(), "oracle-test-"));
    const repo = join(root, "poly-repo");
    try {
      await mkdir(join(repo, ".git"), { recursive: true });
      await writeFile(join(repo, "main.py"), "print('hi')");
      await writeFile(join(repo, "docker-compose.yml"), "services: {}");
      await writeFile(join(repo, "Cargo.toml"), "[package]");
      await writeFile(join(repo, "secret.pem"), "-----BEGIN KEY-----");

      const files: string[] = [];
      for await (const file of walkRepo(repo, "poly-repo", root)) {
        files.push(file.relativePath);
      }

      expect(files.sort()).toEqual([
        "poly-repo/Cargo.toml",
        "poly-repo/docker-compose.yml",
        "poly-repo/main.py",
      ]);
    } finally {
      await rm(root, { recursive: true });
    }
  });

  it("bypasses the JSON manifest allowlist when override includes .json", async () => {
    const root = await mkdtemp(join(tmpdir(), "oracle-test-"));
    const repo = join(root, "json-repo");
    try {
      await mkdir(join(repo, ".git"), { recursive: true });
      await writeFile(join(repo, "package.json"), "{}");
      await writeFile(join(repo, "openapi.json"), "{}");

      const defaultFiles: string[] = [];
      for await (const file of walkRepo(repo, "json-repo", root)) {
        defaultFiles.push(file.relativePath);
      }
      expect(defaultFiles).toEqual(["json-repo/package.json"]);

      const overrideFiles: string[] = [];
      for await (const file of walkRepo(repo, "json-repo", root, {
        extensions: new Set([".json"]),
      })) {
        overrideFiles.push(file.relativePath);
      }
      expect(overrideFiles.sort()).toEqual([
        "json-repo/openapi.json",
        "json-repo/package.json",
      ]);
    } finally {
      await rm(root, { recursive: true });
    }
  });

  it("honours the extensions override (only .rb included)", async () => {
    const root = await mkdtemp(join(tmpdir(), "oracle-test-"));
    const repo = join(root, "ruby-repo");
    try {
      await mkdir(join(repo, ".git"), { recursive: true });
      await writeFile(join(repo, "app.rb"), "puts 'hi'");
      await writeFile(join(repo, "ignored.ts"), "const x = 1;");
      await writeFile(join(repo, "README.md"), "# Ruby");

      const files: string[] = [];
      for await (const file of walkRepo(repo, "ruby-repo", root, {
        extensions: new Set([".rb"]),
      })) {
        files.push(file.relativePath);
      }

      expect(files).toEqual(["ruby-repo/app.rb"]);
    } finally {
      await rm(root, { recursive: true });
    }
  });

  it("skips vendor caches (.bun, .opencode-home, .cache, .yarn) by default", async () => {
    const root = await mkdtemp(join(tmpdir(), "oracle-test-"));
    const repo = join(root, "vendor-repo");
    try {
      await mkdir(join(repo, ".git"), { recursive: true });
      await mkdir(join(repo, "src"), { recursive: true });
      await mkdir(join(repo, ".bun", "install", "cache"), { recursive: true });
      await mkdir(join(repo, ".opencode-home", ".bun", "install", "cache"), { recursive: true });
      await mkdir(join(repo, ".cache", "tsc"), { recursive: true });
      await mkdir(join(repo, ".yarn", "cache"), { recursive: true });

      await writeFile(join(repo, "src", "real.ts"), "export const real = 1;");
      await writeFile(join(repo, ".bun", "install", "cache", "vendored.ts"), "// should be skipped");
      await writeFile(join(repo, ".opencode-home", ".bun", "install", "cache", "vendored.ts"), "// should be skipped");
      await writeFile(join(repo, ".cache", "tsc", "cached.ts"), "// should be skipped");
      await writeFile(join(repo, ".yarn", "cache", "pkg.ts"), "// should be skipped");

      const files: string[] = [];
      for await (const file of walkRepo(repo, "vendor-repo", root)) {
        files.push(file.relativePath);
      }

      expect(files).toEqual(["vendor-repo/src/real.ts"]);
    } finally {
      await rm(root, { recursive: true });
    }
  });

  it("honours the skipDirs override (additional names skipped)", async () => {
    const root = await mkdtemp(join(tmpdir(), "oracle-test-"));
    const repo = join(root, "custom-skip-repo");
    try {
      await mkdir(join(repo, ".git"), { recursive: true });
      await mkdir(join(repo, "src"), { recursive: true });
      await mkdir(join(repo, "generated"), { recursive: true });

      await writeFile(join(repo, "src", "real.ts"), "export const real = 1;");
      await writeFile(join(repo, "generated", "auto.ts"), "// caller-skipped");

      const filesWithoutOverride: string[] = [];
      for await (const file of walkRepo(repo, "custom-skip-repo", root)) {
        filesWithoutOverride.push(file.relativePath);
      }
      expect(filesWithoutOverride.sort()).toEqual([
        "custom-skip-repo/generated/auto.ts",
        "custom-skip-repo/src/real.ts",
      ]);

      const filesWithOverride: string[] = [];
      for await (const file of walkRepo(repo, "custom-skip-repo", root, {
        skipDirs: new Set(["node_modules", ".git", "generated"]),
      })) {
        filesWithOverride.push(file.relativePath);
      }
      expect(filesWithOverride).toEqual(["custom-skip-repo/src/real.ts"]);
    } finally {
      await rm(root, { recursive: true });
    }
  });

  it("does NOT prune when .codebase-oracle-skip is a directory, not a file", async () => {
    const root = await mkdtemp(join(tmpdir(), "oracle-test-"));
    const repo = join(root, "weird-repo");
    try {
      await mkdir(join(repo, ".git"), { recursive: true });
      // Sentinel name as a directory should be ignored (the predicate
      // requires entry.isFile()).
      await mkdir(join(repo, ".codebase-oracle-skip", "inner"), { recursive: true });
      await writeFile(join(repo, "src.ts"), "export const x = 1;");
      await writeFile(join(repo, ".codebase-oracle-skip", "inner", "ignored.ts"), "// inside the dot-dir");

      const files: string[] = [];
      for await (const file of walkRepo(repo, "weird-repo", root)) {
        files.push(file.relativePath);
      }
      // src.ts is yielded; the dot-prefixed dir is not in SKIP_DIRS but
      // its name starts with `.` which is not a special filter inside
      // walk(); however its NAME is `.codebase-oracle-skip` and we deliberately
      // only check file-shaped sentinels, so the subtree walks. The inner .ts
      // file IS yielded because nothing prunes the directory.
      expect(files.sort()).toEqual([
        "weird-repo/.codebase-oracle-skip/inner/ignored.ts",
        "weird-repo/src.ts",
      ]);
    } finally {
      await rm(root, { recursive: true });
    }
  });

  it("prunes the entire repo when the sentinel is at the repo root", async () => {
    const root = await mkdtemp(join(tmpdir(), "oracle-test-"));
    const repo = join(root, "skip-everything");
    try {
      await mkdir(join(repo, ".git"), { recursive: true });
      await mkdir(join(repo, "src"), { recursive: true });
      await writeFile(join(repo, ".codebase-oracle-skip"), "skip");
      await writeFile(join(repo, "src", "would-be-indexed.ts"), "export const x = 1;");

      const files: string[] = [];
      for await (const file of walkRepo(repo, "skip-everything", root)) {
        files.push(file.relativePath);
      }
      expect(files).toEqual([]);
    } finally {
      await rm(root, { recursive: true });
    }
  });

  it("prunes any subtree containing a .codebase-oracle-skip sentinel", async () => {
    const root = await mkdtemp(join(tmpdir(), "oracle-test-"));
    const repo = join(root, "skip-repo");
    try {
      await mkdir(join(repo, ".git"), { recursive: true });
      await mkdir(join(repo, "src"), { recursive: true });
      await mkdir(join(repo, "vendored-fixtures", "sample"), { recursive: true });

      await writeFile(join(repo, "src", "real.ts"), "export const real = 1;");
      await writeFile(join(repo, "vendored-fixtures", ".codebase-oracle-skip"), "skip");
      await writeFile(join(repo, "vendored-fixtures", "ignored.ts"), "// skipped");
      await writeFile(join(repo, "vendored-fixtures", "sample", "deep.ts"), "// also skipped");

      const files: string[] = [];
      for await (const file of walkRepo(repo, "skip-repo", root)) {
        files.push(file.relativePath);
      }

      expect(files).toEqual(["skip-repo/src/real.ts"]);
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

describe("walkRepo size limit and skip reporting", () => {
  it("has a default limit of 500_000 bytes", () => {
    expect(DEFAULT_MAX_FILE_SIZE_BYTES).toBe(500_000);
  });

  it("does not yield a file over the (custom) limit, and reports it via onSkip", async () => {
    const root = await mkdtemp(join(tmpdir(), "oracle-test-"));
    const repo = join(root, "size-repo");
    try {
      await mkdir(join(repo, ".git"), { recursive: true });
      const bigContent = "x".repeat(1000);
      const smallContent = "const ok = 1;";
      await writeFile(join(repo, "big.ts"), bigContent);
      await writeFile(join(repo, "small.ts"), smallContent);

      const skips: SkippedFile[] = [];
      const files: string[] = [];
      for await (const file of walkRepo(repo, "size-repo", root, {
        maxFileSizeBytes: 500,
        onSkip: (s) => skips.push(s),
      })) {
        files.push(file.relativePath);
      }

      // Negative control: the file under the limit IS yielded.
      expect(files).toEqual(["size-repo/small.ts"]);

      expect(skips).toHaveLength(1);
      expect(skips[0]).toMatchObject({
        repo: "size-repo",
        relativePath: "size-repo/big.ts",
        reason: "too-large",
        sizeBytes: 1000,
        limitBytes: 500,
      });
    } finally {
      await rm(root, { recursive: true });
    }
  });

  it("respects the default limit (500_000 bytes) when no override is given", async () => {
    const root = await mkdtemp(join(tmpdir(), "oracle-test-"));
    const repo = join(root, "default-size-repo");
    try {
      await mkdir(join(repo, ".git"), { recursive: true });
      // One byte over the default ceiling, one comfortably under it.
      await writeFile(join(repo, "over.ts"), "x".repeat(DEFAULT_MAX_FILE_SIZE_BYTES + 1));
      await writeFile(join(repo, "under.ts"), "const ok = 1;");

      const skips: SkippedFile[] = [];
      const files: string[] = [];
      for await (const file of walkRepo(repo, "default-size-repo", root, {
        onSkip: (s) => skips.push(s),
      })) {
        files.push(file.relativePath);
      }

      expect(files).toEqual(["default-size-repo/under.ts"]);
      expect(skips).toHaveLength(1);
      expect(skips[0]).toMatchObject({
        relativePath: "default-size-repo/over.ts",
        reason: "too-large",
        sizeBytes: DEFAULT_MAX_FILE_SIZE_BYTES + 1,
        limitBytes: DEFAULT_MAX_FILE_SIZE_BYTES,
      });
    } finally {
      await rm(root, { recursive: true });
    }
  }, 15_000);

  it("reports a read error via onSkip and keeps walking the sibling file", async () => {
    // Running as root bypasses file permission bits entirely, which would
    // make the chmod(0o000) below a no-op and this test a false pass.
    if (process.getuid && process.getuid() === 0) {
      return;
    }

    const root = await mkdtemp(join(tmpdir(), "oracle-test-"));
    const repo = join(root, "unreadable-repo");
    const unreadablePath = join(repo, "locked.ts");
    try {
      await mkdir(join(repo, ".git"), { recursive: true });
      await writeFile(unreadablePath, "export const locked = 1;");
      await writeFile(join(repo, "sibling.ts"), "export const sibling = 1;");
      await chmod(unreadablePath, 0o000);

      const skips: SkippedFile[] = [];
      const files: string[] = [];
      for await (const file of walkRepo(repo, "unreadable-repo", root, {
        onSkip: (s) => skips.push(s),
      })) {
        files.push(file.relativePath);
      }

      // The walk continues past the unreadable file and still yields the
      // sibling — one bad file must not kill the whole scan.
      expect(files).toEqual(["unreadable-repo/sibling.ts"]);

      expect(skips).toHaveLength(1);
      expect(skips[0].relativePath).toBe("unreadable-repo/locked.ts");
      expect(skips[0].reason).toBe("read-error");
      expect(typeof skips[0].message).toBe("string");
    } finally {
      // Restore permissions before rm, otherwise the recursive delete of
      // the 0o000 file can itself fail.
      await chmod(unreadablePath, 0o644).catch(() => {});
      await rm(root, { recursive: true });
    }
  });
});

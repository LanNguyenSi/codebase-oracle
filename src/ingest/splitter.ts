import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";
import { parse as parseYaml } from "yaml";
import type { ScannedFile } from "./scanner.js";

// Language-specific separators for better chunk boundaries
const TS_SEPARATORS = [
  "\nexport ",
  "\nfunction ",
  "\nclass ",
  "\ninterface ",
  "\ntype ",
  "\nconst ",
  "\n\n",
  "\n",
];

const MD_SEPARATORS = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n"];

const DEFAULT_SEPARATORS = ["\n\n", "\n", " "];

function getSeparators(language: string): string[] {
  switch (language) {
    case "ts":
    case "tsx":
    case "js":
    case "jsx":
      return TS_SEPARATORS;
    case "md":
      return MD_SEPARATORS;
    default:
      return DEFAULT_SEPARATORS;
  }
}

export async function splitFile(file: ScannedFile): Promise<Document[]> {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1500,
    chunkOverlap: 200,
    separators: getSeparators(file.language),
  });

  const metadata: Record<string, unknown> = {
    repo: file.repo,
    filePath: file.relativePath,
    language: file.language,
    absolutePath: file.absolutePath,
    fileHash: file.contentHash,
  };

  if (file.language === "md") {
    Object.assign(
      metadata,
      extractFrontmatterMetadata(file.content, file.repo, file.relativePath),
    );
  }

  // The splitter still sees the full, unstripped content: frontmatter text
  // carries semantic signal, and stripping it would shift attachLineNumbers'
  // line accounting out of sync with the on-disk file.
  const docs = await splitter.createDocuments([file.content], [metadata]);

  attachLineNumbers(docs, file.content);

  return docs;
}

interface FrontmatterBlock {
  yamlText: string;
}

// Detects a LEADING frontmatter block: the first line must be exactly `---`
// (a trailing \r is tolerated for CRLF files), and the block ends at the
// next line that is exactly `---`. If there is no closing delimiter, no
// block is recognized at all (same as a file with no leading `---`).
function detectFrontmatterBlock(content: string): FrontmatterBlock | null {
  const lines = content.split("\n");
  if (lines.length === 0) return null;
  if (lines[0].replace(/\r$/, "") !== "---") return null;

  for (let i = 1; i < lines.length; i++) {
    if (lines[i].replace(/\r$/, "") === "---") {
      const yamlLines = lines
        .slice(1, i)
        .map((line) => line.replace(/\r$/, ""));
      return { yamlText: yamlLines.join("\n") };
    }
  }
  return null;
}

function isStringArray(value: unknown): value is string[] {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every((v) => typeof v === "string" && v.length > 0)
  );
}

// Extracts exactly four flat, `fm`-prefixed metadata keys from a markdown
// file's leading YAML frontmatter block, if present and valid. Never
// throws: parse errors or a non-object result are fail-soft (one
// console.warn, no fm keys), and files with no leading `---` are untouched.
export function extractFrontmatterMetadata(
  content: string,
  repo: string,
  filePath: string,
): Record<string, unknown> {
  const block = detectFrontmatterBlock(content);
  if (!block) return {};

  let parsed: unknown;
  try {
    parsed = parseYaml(block.yamlText);
  } catch (err) {
    const reason = err instanceof Error ? err.message : String(err);
    console.warn(`frontmatter parse failed in ${repo}/${filePath}: ${reason}`);
    return {};
  }

  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    console.warn(
      `frontmatter parse failed in ${repo}/${filePath}: frontmatter block did not parse to a mapping`,
    );
    return {};
  }

  const obj = parsed as Record<string, unknown>;
  const fm: Record<string, unknown> = {};
  if (typeof obj.type === "string" && obj.type.length > 0) fm.fmType = obj.type;
  if (typeof obj.title === "string" && obj.title.length > 0)
    fm.fmTitle = obj.title;
  if (isStringArray(obj.tags)) fm.fmTags = obj.tags;
  if (isStringArray(obj.sources)) fm.fmSources = obj.sources;
  return fm;
}

// Annotate each chunk with 1-indexed lineStart / lineEnd in the source file.
// Chunks come back in source order, so we walk forward through the content
// with a moving cursor and locate each chunk by indexOf. A chunk that doesn't
// resolve cleanly (e.g. whitespace normalization edge case) inherits its
// line_start from the previous chunk's tail so the metadata is still useful.
function attachLineNumbers(docs: Document[], content: string): void {
  let cursor = 0;
  let fallbackLine = 1;
  for (const doc of docs) {
    const chunk = doc.pageContent;
    const idx = content.indexOf(chunk, cursor);
    let lineStart: number;
    let lineEnd: number;
    if (idx === -1) {
      lineStart = fallbackLine;
      lineEnd = fallbackLine + countNewlines(chunk);
    } else {
      lineStart = countNewlines(content.slice(0, idx)) + 1;
      lineEnd = lineStart + countNewlines(chunk);
      // Advance cursor past the start of this chunk but not past the whole
      // chunk — chunkOverlap means the next chunk legitimately overlaps with
      // this one's tail, and we want indexOf to find that overlap.
      // TODO: files with verbatim-duplicated regions larger than ~half a
      // chunk (e.g. two copies of the same license header back-to-back)
      // can confuse the indexOf walk and reuse the first occurrence's line
      // numbers for the second chunk. Degrades gracefully — line metadata
      // is still useful — but worth revisiting if real-world repos hit it.
      cursor = idx + Math.max(1, Math.floor(chunk.length / 2));
    }
    doc.metadata.lineStart = lineStart;
    doc.metadata.lineEnd = lineEnd;
    fallbackLine = lineEnd;
  }
}

function countNewlines(s: string): number {
  let n = 0;
  for (let i = 0; i < s.length; i++) {
    if (s.charCodeAt(i) === 10) n++;
  }
  return n;
}

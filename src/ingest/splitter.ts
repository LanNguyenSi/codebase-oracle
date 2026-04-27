import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";
import type { ScannedFile } from "./scanner.js";

// Language-specific separators for better chunk boundaries
const TS_SEPARATORS = [
  "\nexport ", "\nfunction ", "\nclass ", "\ninterface ", "\ntype ",
  "\nconst ", "\n\n", "\n",
];

const MD_SEPARATORS = [
  "\n## ", "\n### ", "\n#### ", "\n\n", "\n",
];

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

  const docs = await splitter.createDocuments(
    [file.content],
    [
      {
        repo: file.repo,
        filePath: file.relativePath,
        language: file.language,
        absolutePath: file.absolutePath,
        fileHash: file.contentHash,
      },
    ],
  );

  attachLineNumbers(docs, file.content);

  return docs;
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

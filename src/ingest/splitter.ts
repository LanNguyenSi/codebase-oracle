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
      },
    ],
  );

  return docs;
}

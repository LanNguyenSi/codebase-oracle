import { OpenAIEmbeddings } from "@langchain/openai";
import type { Embeddings } from "@langchain/core/embeddings";
import type { Config } from "../config.js";

export function createEmbeddings(config: Config): Embeddings {
  if (!config.openaiApiKey) {
    throw new Error(
      "OPENAI_API_KEY is required for embeddings. Set it in your environment.",
    );
  }

  return new OpenAIEmbeddings({
    openAIApiKey: config.openaiApiKey,
    modelName: config.embeddingModel,
    stripNewLines: true,
  });
}

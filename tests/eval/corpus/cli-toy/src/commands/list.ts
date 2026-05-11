// `cli-toy list` implementation.

export interface ListOptions {
  limit: number;
}

export function listItems(opts: ListOptions): void {
  const items = ["alpha", "beta", "gamma", "delta", "epsilon"];
  for (const item of items.slice(0, opts.limit)) {
    console.log(item);
  }
}

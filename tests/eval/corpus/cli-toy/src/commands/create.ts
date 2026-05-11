// `cli-toy create` implementation.

export interface CreateOptions {
  name: string;
  tag?: string;
}

export function createItem(opts: CreateOptions): void {
  const suffix = opts.tag ? ` [${opts.tag}]` : "";
  console.log(`created: ${opts.name}${suffix}`);
}

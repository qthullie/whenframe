// A description embedded in the browser as the tool embeds it with the small model's text tower (PE core's, its
// 8-bit weights: scripts/small.py web), with onnxruntime-web.
//
// The tokenizer is CLIP's, as timm's tokenizer.json for PE core sets it: Unicode NFC, each run of white space one
// space, lower case; then words, single digits and runs of other symbols, each one's UTF-8 bytes written with GPT-2's
// characters for bytes and merged pair by pair in the merges' order, "</w>" ending the word; the start and end tokens
// around at most 30 tokens, zeros after them, 32 in all. CLIP's vocabulary follows from its merges: the 256 bytes,
// the same ending a word, then each merge's result, then the two special tokens.
const START = 49406, END = 49407, LENGTH = 32;
const PIECES = /'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/gu;

function byteCharacters() {
  // GPT-2's: printable bytes stand for themselves, the others for the characters from U+0100 on, in byte order
  const kept = [];
  for (let b = 0x21; b <= 0x7e; b++) kept.push(b);
  for (let b = 0xa1; b <= 0xac; b++) kept.push(b);
  for (let b = 0xae; b <= 0xff; b++) kept.push(b);
  const order = [...kept], chars = new Array(256);
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!kept.includes(b)) order.push(b);
  }
  order.forEach((b, i) => {
    chars[b] = String.fromCharCode(i < kept.length ? b : 256 + n++);
  });
  return { chars, order };
}

export class Tokenizer {
  constructor(merges) {
    const { chars, order } = byteCharacters();
    const vocab = order.map((b) => chars[b]);
    vocab.push(...vocab.map((c) => `${c}</w>`));
    this.ranks = new Map();
    merges.forEach((m, i) => {
      this.ranks.set(m, i);
      vocab.push(m.replace(" ", ""));
    });
    this.ids = new Map(vocab.map((t, i) => [t, i]));
    this.chars = chars;
    this.utf8 = new TextEncoder();
    this.cache = new Map();
  }

  static async load(url) {
    const text = await (await fetch(url)).text();
    return new Tokenizer(text.split("\n").filter((line) => line.length > 0));
  }

  bpe(word) {
    const known = this.cache.get(word);
    if (known) return known;
    let parts = Array.from(word);
    parts[parts.length - 1] += "</w>";
    while (parts.length > 1) {
      let best = -1, rank = Infinity;
      for (let i = 0; i + 1 < parts.length; i++) {
        const r = this.ranks.get(`${parts[i]} ${parts[i + 1]}`);
        if (r !== undefined && r < rank) {
          rank = r;
          best = i;
        }
      }
      if (best < 0) break;
      const a = parts[best], b = parts[best + 1], merged = [];
      for (let i = 0; i < parts.length; ) {
        if (i + 1 < parts.length && parts[i] === a && parts[i + 1] === b) {
          merged.push(a + b);
          i += 2;
        } else {
          merged.push(parts[i]);
          i += 1;
        }
      }
      parts = merged;
    }
    const ids = parts.map((p) => this.ids.get(p) ?? END);  // the end token is the tokenizer's unknown one
    this.cache.set(word, ids);
    return ids;
  }

  // the 32 token ids of a description, as BigInt64Array (the tower's int64 input)
  encode(text) {
    const clean = text.normalize("NFC").replace(/\s+/gu, " ").toLowerCase();
    const tokens = [];
    for (const [piece] of clean.matchAll(PIECES)) {
      let word = "";
      for (const byte of this.utf8.encode(piece)) word += this.chars[byte];
      tokens.push(...this.bpe(word));
    }
    const ids = new BigInt64Array(LENGTH), n = Math.min(tokens.length, LENGTH - 2);
    ids[0] = BigInt(START);
    for (let i = 0; i < n; i++) ids[i + 1] = BigInt(tokens[i]);
    ids[n + 1] = BigInt(END);
    return ids;
  }
}

export class TextTower {
  // ort: onnxruntime-web's module, its WebAssembly files where ort.env.wasm.wasmPaths says
  static async load(ort, modelUrl, mergesUrl) {
    const [tokenizer, session] = await Promise.all([
      Tokenizer.load(mergesUrl),
      ort.InferenceSession.create(modelUrl, { executionProviders: ["wasm"], graphOptimizationLevel: "all" }),
    ]);
    const tower = new TextTower();
    Object.assign(tower, { ort, tokenizer, session });
    return tower;
  }

  // the description's embedding, normalised, as the tool's search compares it with the frames; the tower is causal
  // and reads the end token's output, so only the tokens up to it are computed (the zeros after it change nothing)
  async embed(text) {
    const ids = this.tokenizer.encode(text), n = ids.indexOf(BigInt(END)) + 1;
    const out = await this.session.run({ input_ids: new this.ort.Tensor("int64", ids.subarray(0, n), [1, n]) });
    const v = Float32Array.from(out.text_embeds.data);
    let norm = 0;
    for (const x of v) norm += x * x;
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < v.length; i++) v[i] /= norm;
    return v;
  }
}

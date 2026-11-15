# Building an Inference Engine from Scratch

**밑바닥부터 만드는 추론 엔진: Rust로 배우는 AI 인퍼런스 엔지니어링**

A book (written in Korean) that builds an AI inference engine from scratch
in Rust, runs well-known models end to end, and compares the result against
production engines such as llama.cpp and LiteRT.

- Read online: <https://chosungmann.github.io/building-an-inference-engine-from-scratch/>
- Proposal: [docs/proposal.md](docs/proposal.md)

The book is a work in progress. Chapters are published as they are completed.

## Repository layout

| Path | Contents |
| --- | --- |
| `book/` | mdBook sources for the book |
| `crates/` | The `badak` inference engine built in the book (Rust) |
| `scripts/figures/` | Scripts that generate the book's SVG figures |
| `docs/` | Proposal, progress, and next_edition notes |
| `AGENTS.md` | Writing philosophy and working rules |

## Building and testing the code

```bash
cargo test --workspace
```

## Building the book locally

Install [mdBook](https://rust-lang.github.io/mdBook/) 0.4.48, then:

```bash
mdbook serve book --open
```

## License

The source code (`crates/`, `scripts/`) is released under the
[MIT License](LICENSE). The book text and figures (`book/`, `docs/`) are
not covered by that license; all rights are reserved by the author.

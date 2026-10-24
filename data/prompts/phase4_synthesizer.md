You are a technical book author compiling a Living Codex from a RepoMap.
Write a clear markdown chapter for human readers.

The RepoMap has two parts:
- Part 1 (Core Concepts): node NAMES and PageRank scores only. These are
  graph entity ids (concepts), NOT document ids.
- Part 2 (Grounding Context): source passages already tagged as
  [DOC_ID: <id>]. Those <id> values are the only legal citations.

MUST:
1. Use markdown headings (# / ##) and prose paragraphs.
2. Cite sources inline exactly as [DOC_ID: <id>] using IDs printed in
   Part 2 (Grounding Context), copied verbatim.
3. Do not invent document IDs that are not in Part 2.
4. Do not wrap the entire chapter in a single fenced code block.
5. Start with a single H1 matching the given chapter title.
6. Every prose paragraph must include at least one inline [DOC_ID: <id>]
   (do not park all citations in a single leftover line).
7. HARD CONSTRAINT — this is not a suggestion, it has caused real failures:
   Part 1 (Core Concepts) node names, chapter/book titles, and any string
   that is not printed verbatim inside a "[DOC_ID: ...]" tag in Part 2 are
   NEVER valid citations. Study the examples below before writing:

   CORRECT:
   Part 2 shows: [DOC_ID: labor_law_jo_12] "정당한 사유 없이 해고..."
   Chapter text: "...정당한 사유가 필요하다 [DOC_ID: labor_law_jo_12]."

   WRONG (Part 1 concept node used as a citation):
   Part 1 shows: - PROGRAMMER (PageRank: 0.18)
   Chapter text: "...프로그래머는... [DOC_ID: PROGRAMMER]"
   -> PROGRAMMER is a concept name, not a DOC_ID. Never cite it.

   WRONG (invented string, title, or stray token used as a citation):
   Chapter text: "...『어떻게 살 것인가』에서... [DOC_ID: 어떻게살인가]"
   -> "어떻게살인가" never appears as a DOC_ID in Part 2. Do not invent IDs
      from titles, headings, or any text you generate yourself, and never
      let typos or stray model tokens leak into a [DOC_ID: ...] tag.

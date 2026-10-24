You extract domain-specific entities and relationships from knowledge memos.

MUST rules:
1. Never extract generic nouns (e.g. system, company, user, data) as nodes.
   Extract only domain-specific proper nouns / named concepts. (anti super-node)
2. Normalize every node id: replace spaces with underscore (_), UPPERCASE.
3. Node type must be exactly one of: CONCEPT, PERSON, SYSTEM.
4. Include the given chunk_id in each node's source_chunk_ids.
5. Edge source/target must reference node ids present in the nodes list.
6. One concept = one id. Do not emit near-duplicates that differ only by
   spacing, hyphen, or underscore (e.g. NA_JASIN vs NAJASIN).
7. Edges are as important as nodes. Densely connect the extracted nodes:
   every node MUST appear in at least one edge. An isolated node is a
   FAILURE — actively link each node to the chunk's core theme or to its
   sibling nodes via a meaningful relation (e.g. USES, CAUSES, PART_OF,
   CONTRASTS_WITH, RELATES_TO).
8. Prefer edge-rich output: target edge count >= node count. Do not stop
   after listing entities; always ask "how does this node relate to the
   others in this chunk?" before finishing.

# Parking Lot

## From escape-stripe research
- S2: 48-cell archive — might help with diversity, test as T3 after finding good fitness config
- S8: 48-cell + dir_balance — superseded by new tortuosity approach
- S9: CONTEXT_SIZE=2 (24 inputs) — more spatial context, could help learn maze structure

## New ideas
- Dead-end pruning before measuring: remove all floor tiles with <2 floor neighbors recursively → measure path in pruned maze
- Maze density control: enforce minimum floor percentage to prevent sparse maps
- Bottleneck penalty: penalize maps where there's exactly 1 path between P and E (no redundancy)

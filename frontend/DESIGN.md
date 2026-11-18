# Frontend design notes — Fireforce research console

Assumption: the audience is a researcher or developer comparing three known
checkpoints on one sample. The job is inspection and fair comparison, not
persuasion or marketing. The brief in `wildfire-prototype-opencode-docs/DESIGN.md`
is the source of truth.

## Concept card

```
Idea:        A laboratory comparison bench for 64×64 fire masks.
Metaphor:    Instrument panel: numbered sections, hairline rules, mono readouts.
Signature:   One inspector pixel. Hover any mask and the same row/column lights
             up in all five masks, with per-model values and hit/miss in the readout.
Emotion:     Clinical, exact, unhurried — the data is the drama.
Motion verb: Settle. States change in ~120 ms; nothing moves for decoration.
Screenshot:  Three equal masks under one view switch, cursor square on a disputed pixel.
```

## Palette

| Role | Hex |
|---|---|
| Ground | `#0d0f0d` |
| Panel | `#121512` |
| Ink | `#ece9e2` |
| Muted | `#9c9f96` |
| Faint | `#6e7369` |
| Line | `#242823` / `#363b34` |
| Accent (fire) | `#e4572e` |
| Agreement / ready | `#a9b99a` |
| Warning | `#d7a05c` |

Fire orange is limited to the primary action, selected catalog row, fire pixels,
and one live-state dot. It is never a background theme.

## Type

- IBM Plex Sans — body, headings, figure titles.
- IBM Plex Mono — IDs, counts, metrics, labels, metadata.
- Display: sample ID at 26–40 px mono, tracking `-0.035em`.
- Labels: 9–10 px mono, uppercase, `0.12–0.18em` tracking.
- Numbers use tabular figures so metric columns do not shift.

## Grid and shape

- Header 57 px, catalog rail 296 px, workspace fluid, max width 1720 px.
- Three equal columns for both preview masks and model masks, so every mask is
  rendered at the same scale.
- One radius rule: 2 px on buttons, inputs, and panels. No floating cards, no
  shadows, no gradients in the chrome.
- Section numbering (`01 Catalog`, `02 Inputs & target`, `03 Comparison`) gives
  orientation in one scan.

## Motion

120 ms colour/border transitions on actionable elements only. No entrance
animations, no loading shimmer. `prefers-reduced-motion` collapses transitions.

## Composition

```
┌ header: mark / title ───────────── status · device · samples · fingerprint ┐
├ catalog 01 ──────────┬ sample header  h1 id ───────── [Run predictions] ───┤
│ split · search       │ 02 Inputs & target                                  │
│ 50 IDs / page        │ [previous fire] [ground truth] [facts]              │
│ pager                │ 03 Comparison   [Binary][Probability][Consensus]    │
│                      │ [M1 1ch] [M2 8ch] [M3 13ch]                         │
│                      │ metrics under each · inspector readout               │
└──────────────────────┴─────────────────────────────────────────────────────┘
```

At <820 px the three-column grids become one column and masks keep their square
aspect; the catalog becomes a bounded list above the workspace. At <620 px the
header drops secondary facts and metric cells become a 2×2 grid.

## Deliberate deviations from a generic dashboard

- No rounded cards, drop shadows, gradients, or hero banner.
- The comparison is a single instrument grid, not three independent cards.
- The synchronized pixel inspector and consensus map are specific to this
  dataset; they would not make sense on another dashboard.
- Honest empty, partial, and all-unknown states are designed rather than hidden
  behind a spinner or zero-filled metrics.

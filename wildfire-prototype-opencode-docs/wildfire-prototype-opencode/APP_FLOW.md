# App Flow / User Journey

## Primary journey
Open dashboard → choose split → search/select sample ID → inspect sample → run predictions → compare M1/M2/M3 → inspect metrics/ground truth → select another sample.

## Screens
1. **Dashboard:** title, backend readiness, device, indexed sample counts.
2. **Sample browser:** split filter, searchable/paginated sample ID list, label availability.
3. **Sample preview:** selected ID, previous-fire mask, target mask if valid, valid/unknown pixel counts, available input features.
4. **Comparison:** three equal model panels with predicted mask, optional probability view, threshold, checkpoint ID, inference time, and metrics. Show ground truth separately.

## States
- Initial loading / backend offline.
- Empty manifest / no search results.
- Sample loading / ready.
- Inference running.
- Full success.
- Partial model failure.
- Total failure.
- Ground truth unavailable.
- All target pixels unknown (metrics unavailable).

## Interaction rules
Changing sample clears old results. Prevent duplicate in-flight submissions. Keep sample ID visible. Never show metrics if there are zero valid target pixels. Never treat target `-1` as non-fire. Do not display date, location, or event names unless verified metadata exists.

## Accessibility
Keyboard-accessible controls, textual legends in addition to color, same mask dimensions and scale across model panels, and numeric metrics alongside visuals.

# UI/UX Design Brief

## Direction
A precise scientific dashboard, not a consumer weather app. Prioritize fair visual comparison, data validity, and legibility over decoration.

## Visual system
Use one coherent neutral theme (dark charcoal or light). Use restrained fire-orange for predicted fire and primary actions, gray for non-fire, and a distinct pattern/outline for unknown target pixels. Never rely on red/green alone.

## Layout
Header: app title + backend/device status.
Controls: split + searchable sample ID.
Preview: previous-fire input and ground truth.
Results: three equal-width model cards.
Each card: model name, input features, mask, threshold, inference time, checkpoint ID, metrics/status.

## Mask viewer
Render 64×64 pixels with nearest-neighbor scaling so boundaries remain exact. Keep dimensions and scale identical across M1/M2/M3/ground truth. Include a legend and optional zoom.

## Metrics
Show Dice, IoU, precision, recall consistently as decimals or percentages. Use N/A when target labels are unavailable. Do not headline accuracy because of severe class imbalance.

## Responsive behavior
Desktop is the primary research workflow. On mobile, stack model cards vertically and preserve readable masks.

## Motion and copy
Minimal transitions; honest progress indicators (no fake percentages). Use labels such as “Predicted next-day fire mask” and “Ground-truth next-day mask.” Avoid claims such as “fire will occur here” or “real-time forecast.”

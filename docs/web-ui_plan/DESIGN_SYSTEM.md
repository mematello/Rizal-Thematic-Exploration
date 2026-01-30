# Rizal Semantic Search - Design System (v2.0)

## 1. Color Palette (Accessibility Verified)
| Name | Hex | Role | Contrast (on Cream) |
| :--- | :--- | :--- | :--- |
| **Papel de Hapon** | `#FFF8E1` | Background | N/A |
| **Intramuros Brown**| `#3E2723` | Primary Brand, Text | 13.6:1 (AAA) |
| **Academic Blue** | `#1A237E` | Links, Buttons | 12.5:1 (AAA) |
| **Tinta (Text)** | `#261612` | Body Copy | 15.2:1 (AAA) |
| **Teal Insight** | `#00695C` | Semantic Score/Highlight| 6.8:1 (AA) |
| **Ibarra Gold** | `#F57F17` | Noli Accent | 3.1:1 (UI Only)* |
| **Simoun Magenta** | `#AD1457` | Fili Accent | 5.4:1 (AA) |
| **Amber Highlight** | `#FFF59D` | Lexical Background | (Background color) |

*> Note: Ibarra Gold is used for borders/graphics, never for text on light backgrounds.*

## 2. Typography
**Font Stack:**
* Headers: `Roboto`, sans-serif (Weights: 500, 700)
* Body: `Crimson Text`, serif (Weights: 400, 600, 400i)

**Scale (Mobile/Desktop):**
* **H1:** 28px/36px (Bold)
* **H2:** 22px/28px (Bold)
* **Body:** 18px (Regular, 1.6 LH) - *Critical for readability*
* **Caption:** 14px (Regular)
* **Micro:** 12px (Mono/Bold)

## 3. Spacing Grid
Base unit: **4px**
* **p-1 (4px):** Tight adjustments
* **p-4 (16px):** Standard card padding
* **gap-6 (24px):** Section separation

## 4. Accessibility Checklist (WCAG 2.1 AA)
- [ ] All images have `alt` text or `role="presentation"`.
- [ ] Result Cards use `<article>` and unique `aria-labelledby`.
- [ ] Score Visualizers have `aria-label` describing values.
- [ ] Color is never the only means of conveying information (Underlines used for semantic matches).
- [ ] Focus rings visible on all interactive elements.

## 5. Performance Budget (Total < 500KB)
* **Fonts:** 60KB (Subsetting: Latin + Ext)
* **React/Next Bundle:** 180KB (Gzipped)
* **Styles (Tailwind):** 12KB
* **Icons (SVG):** 5KB
* **Data (First Contentful Paint):** ~15KB (JSON)
* **Buffer:** ~200KB for images/other
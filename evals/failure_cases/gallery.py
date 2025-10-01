"""
evals/failure_cases/gallery.py

Builds an annotated HTML failure gallery from collected FailureCase objects.

The gallery renders each failure as a card with:
    - Observed vs reconstructed spectrum overlay
    - Physics residual (shaded area)
    - Predicted vs true SMILES
    - ECR, Tanimoto, GF recall metrics
    - Failure type badge + description
    - Manual annotation field

Output: evals/failure_cases/gallery.html
        evals/failure_cases/figures/*.png  (one per case)

Usage:
    builder = FailureGalleryBuilder()
    builder.build(cases, output_dir="evals/failure_cases")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spectralm.physics.group_frequencies import WAVENUMBER_AXIS
from evals.failure_cases.collector import FailureCase, FAILURE_TYPES


# ── Colour scheme per failure type ────────────────────────────────────────────
FAILURE_COLOURS = {
    "solvent_interference":  "#e8a020",
    "regioisomer_confusion": "#7B5EA7",
    "novel_scaffold":        "#3d6b9e",
    "overtone_band":         "#1D9E75",
    "invalid_smiles":        "#c94040",
    "low_confidence":        "#888780",
    "unknown":               "#444441",
}

SEVERITY_COLOURS = {
    "critical": "#c94040",
    "major":    "#e8a020",
    "minor":    "#1D9E75",
}


class FailureGalleryBuilder:
    """
    Renders annotated failure case gallery as HTML + individual PNG plots.
    """

    def __init__(self, figsize: tuple = (10, 3.5), dpi: int = 120):
        self.figsize = figsize
        self.dpi     = dpi

        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "font.family":       "serif",
            "axes.labelsize":    9,
            "axes.titlesize":    10,
        })

    def build(
        self,
        cases: list[FailureCase],
        output_dir: str | Path,
        title: str = "SpectraLM — Failure Case Gallery",
    ):
        """
        Build full gallery: individual PNG plots + HTML report.
        """
        output_dir = Path(output_dir)
        figs_dir   = output_dir / "figures"
        figs_dir.mkdir(parents=True, exist_ok=True)

        print(f"Building gallery for {len(cases)} failure cases...")

        # ── 1. Individual plots ────────────────────────────────────────────
        plot_paths: dict[str, Path] = {}
        for case in cases:
            path = figs_dir / f"{case.case_id}.png"
            self._plot_case(case, path)
            plot_paths[case.case_id] = path

        # ── 2. Summary taxonomy plot ───────────────────────────────────────
        taxonomy_path = figs_dir / "taxonomy_summary.png"
        self._plot_taxonomy(cases, taxonomy_path)

        # ── 3. HTML gallery ────────────────────────────────────────────────
        html_path = output_dir / "gallery.html"
        html      = self._render_html(cases, plot_paths, taxonomy_path, title)
        html_path.write_text(html, encoding="utf-8")

        # ── 4. Save index JSON ─────────────────────────────────────────────
        index = [{
            "case_id":      c.case_id,
            "failure_type": c.failure_type,
            "severity":     c.severity_label,
            "ecr":          round(c.ecr, 4),
            "tanimoto":     round(c.tanimoto, 4),
        } for c in cases]
        with open(output_dir / "gallery_index.json", "w") as f:
            json.dump(index, f, indent=2)

        print(f"✓ Gallery → {html_path}")
        print(f"✓ Figures → {figs_dir}  ({len(cases)} plots)")

    # ── Plotting ───────────────────────────────────────────────────────────

    def _plot_case(self, case: FailureCase, output_path: Path):
        """
        Three-panel plot for a single failure case:
            Left  : observed vs reconstructed spectrum
            Centre: residual spectrum with annotated peaks
            Right : metrics summary text
        """
        obs   = np.array(case.observed_spectrum)
        recon = np.array(case.reconstructed_spectrum)
        resid = np.array(case.residual_spectrum)
        wn    = WAVENUMBER_AXIS

        fig, axes = plt.subplots(1, 3, figsize=self.figsize,
                                 gridspec_kw={"width_ratios": [2.5, 2, 1]})

        fc = FAILURE_COLOURS.get(case.failure_type, "#444441")
        sc = SEVERITY_COLOURS.get(case.severity_label, "#888780")

        # ── Panel 1: Observed vs Reconstructed ────────────────────────────
        ax1 = axes[0]
        ax1.plot(wn, obs,   lw=1.0, color="#2d4a7a",
                 label="Observed", zorder=3)
        ax1.plot(wn, recon, lw=0.9, color="#c94040",
                 linestyle="--", label="Reconstructed", alpha=0.85)
        ax1.fill_between(wn, obs, recon, alpha=0.20,
                         color="#e8a020", label="Residual")
        ax1.invert_xaxis()
        ax1.set_xlim(4000, 400)
        ax1.set_ylim(-0.05, 1.15)
        ax1.set_xlabel("Wavenumber (cm⁻¹)")
        ax1.set_ylabel("Normalised Absorbance")
        ax1.set_title("Observed vs Reconstructed", fontsize=9)
        ax1.legend(fontsize=7, loc="upper right")

        # Annotate worst residual region
        worst_wn_idx = int(np.argmax(resid))
        worst_wn     = float(wn[worst_wn_idx])
        ax1.axvline(worst_wn, color=sc, lw=0.8, linestyle=":",
                    alpha=0.7)
        ax1.text(worst_wn + 30, 0.95, f"Δmax\n{worst_wn:.0f}",
                 fontsize=6.5, color=sc,
                 ha="left" if worst_wn < 2200 else "right")

        # ── Panel 2: Residual spectrum ────────────────────────────────────
        ax2 = axes[1]
        ax2.fill_between(wn, resid, alpha=0.55, color=fc)
        ax2.plot(wn, resid, lw=0.7, color=fc)
        ax2.axhline(0.25, color="#c94040", lw=0.8, linestyle="--",
                    label="ECR threshold")
        ax2.invert_xaxis()
        ax2.set_xlim(4000, 400)
        ax2.set_ylim(0, max(0.5, resid.max() * 1.15))
        ax2.set_xlabel("Wavenumber (cm⁻¹)")
        ax2.set_ylabel("|Observed − Reconstructed|")
        ax2.set_title("Physics Residual", fontsize=9)
        ax2.legend(fontsize=7)

        # Annotate GF violations
        from spectralm.physics.group_frequencies import GROUP_FREQUENCY_TABLE
        for gf in GROUP_FREQUENCY_TABLE:
            if gf.name in case.gf_violations:
                mid = (gf.low + gf.high) / 2
                ax2.axvspan(gf.low, gf.high, alpha=0.10,
                            color="#c94040", zorder=0)
                ax2.text(mid, resid.max() * 0.85,
                         gf.name.split("(")[0][:8],
                         fontsize=5.5, ha="center",
                         color="#c94040", rotation=90)

        # ── Panel 3: Metrics card ─────────────────────────────────────────
        ax3 = axes[2]
        ax3.axis("off")

        lines = [
            ("Case",       case.case_id),
            ("ECR",        f"{case.ecr:.4f}"),
            ("Tanimoto",   f"{case.tanimoto:.3f}"),
            ("GF Recall",  f"{case.gf_recall:.3f}"),
            ("Log-prob",   f"{case.log_prob:.2f}"),
            ("MW pred",    f"{case.mw_predicted:.0f}" if case.mw_predicted >= 0 else "N/A"),
            ("MW true",    f"{case.mw_true:.0f}" if case.mw_true >= 0 else "N/A"),
            ("Formula P",  case.formula_predicted or "N/A"),
            ("Formula T",  case.formula_true or "N/A"),
            ("Valid SMILES", "Yes" if case.valid_smiles else "No"),
        ]

        y = 0.98
        for label, value in lines:
            ax3.text(0.05, y, f"{label}:", fontsize=7.5, fontweight="bold",
                     transform=ax3.transAxes, va="top")
            colour = "#c94040" if (
                (label == "ECR" and case.ecr > 0.25) or
                (label == "Valid SMILES" and not case.valid_smiles)
            ) else "#2d2d2d"
            ax3.text(0.55, y, value, fontsize=7.5, color=colour,
                     transform=ax3.transAxes, va="top")
            y -= 0.085

        # Failure type badge
        badge_y = y - 0.02
        ax3.add_patch(plt.Rectangle(
            (0.0, badge_y - 0.07), 1.0, 0.08,
            transform=ax3.transAxes,
            facecolor=fc, alpha=0.2, linewidth=0
        ))
        ax3.text(0.5, badge_y - 0.03,
                 case.failure_type.replace("_", " ").upper(),
                 fontsize=6.5, ha="center", color=fc,
                 fontweight="bold", transform=ax3.transAxes, va="center")

        # Predicted SMILES (truncated)
        pred_display = (case.predicted_smiles[:22] + "…"
                        if len(case.predicted_smiles) > 22
                        else case.predicted_smiles)
        true_display = (case.true_smiles[:22] + "…"
                        if len(case.true_smiles) > 22
                        else case.true_smiles)
        y2 = badge_y - 0.12
        ax3.text(0.05, y2, "Pred:", fontsize=6.5, color="#c94040",
                 fontweight="bold", transform=ax3.transAxes, va="top")
        ax3.text(0.05, y2 - 0.07, pred_display, fontsize=6,
                 color="#c94040", transform=ax3.transAxes, va="top",
                 family="monospace")
        ax3.text(0.05, y2 - 0.15, "True:", fontsize=6.5, color="#2d4a7a",
                 fontweight="bold", transform=ax3.transAxes, va="top")
        ax3.text(0.05, y2 - 0.22, true_display, fontsize=6,
                 color="#2d4a7a", transform=ax3.transAxes, va="top",
                 family="monospace")

        # ── Title ──────────────────────────────────────────────────────────
        severity_label = case.severity_label.upper()
        fig.suptitle(
            f"{case.case_id}  |  {case.failure_type.replace('_', ' ')}  "
            f"|  Severity: {severity_label}",
            fontsize=9, fontweight="bold", color=fc,
            x=0.42,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_taxonomy(self, cases: list[FailureCase], output_path: Path):
        """Summary pie + bar chart of failure taxonomy."""
        from collections import Counter

        type_counts    = Counter(c.failure_type for c in cases)
        severity_counts = Counter(c.severity_label for c in cases)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Pie chart — failure types
        labels = [k.replace("_", "\n") for k in type_counts.keys()]
        colours = [FAILURE_COLOURS.get(k, "#888780") for k in type_counts.keys()]
        ax1.pie(
            type_counts.values(),
            labels=labels,
            colors=colours,
            autopct="%1.0f%%",
            startangle=140,
            textprops={"fontsize": 8},
        )
        ax1.set_title("Failure Type Distribution", fontweight="bold")

        # Bar chart — severity
        sev_order = ["critical", "major", "minor"]
        sev_vals  = [severity_counts.get(s, 0) for s in sev_order]
        sev_cols  = [SEVERITY_COLOURS[s] for s in sev_order]
        bars = ax2.bar(sev_order, sev_vals, color=sev_cols,
                       edgecolor="white", lw=0.5)
        for bar, v in zip(bars, sev_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     v + 0.3, str(v), ha="center", fontsize=9)
        ax2.set_ylabel("Count"); ax2.set_title("Severity Distribution", fontweight="bold")

        plt.suptitle(f"Failure Gallery Summary  (n={len(cases)})",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    # ── HTML renderer ──────────────────────────────────────────────────────

    def _render_html(
        self,
        cases: list[FailureCase],
        plot_paths: dict[str, Path],
        taxonomy_path: Path,
        title: str,
    ) -> str:
        """Render full HTML gallery page."""

        def card(case: FailureCase, img_path: Path) -> str:
            fc   = FAILURE_COLOURS.get(case.failure_type, "#444441")
            sc   = SEVERITY_COLOURS.get(case.severity_label, "#888780")
            desc = FAILURE_TYPES.get(case.failure_type, "")
            rel_img = img_path.name

            return f"""
<div class="case-card" id="{case.case_id}">
  <div class="card-header" style="border-left: 4px solid {fc};">
    <span class="case-id">{case.case_id}</span>
    <span class="badge" style="background:{fc}20;color:{fc};border:1px solid {fc}40;">
      {case.failure_type.replace('_',' ')}
    </span>
    <span class="severity" style="color:{sc};">
      {case.severity_label.upper()}  (score: {case.severity_score:.3f})
    </span>
  </div>
  <img src="figures/{rel_img}" alt="{case.case_id}" class="spectrum-plot"/>
  <div class="metrics-row">
    <div class="metric {'bad' if case.ecr > 0.25 else 'ok'}">
      <span class="mlabel">ECR</span>
      <span class="mval">{case.ecr:.4f}</span>
    </div>
    <div class="metric {'bad' if case.tanimoto < 0.3 else 'ok'}">
      <span class="mlabel">Tanimoto</span>
      <span class="mval">{case.tanimoto:.3f}</span>
    </div>
    <div class="metric">
      <span class="mlabel">GF Recall</span>
      <span class="mval">{case.gf_recall:.3f}</span>
    </div>
    <div class="metric">
      <span class="mlabel">Log-prob</span>
      <span class="mval">{case.log_prob:.2f}</span>
    </div>
  </div>
  <div class="smiles-row">
    <div class="smiles pred">
      <span class="slabel">Predicted</span>
      <code>{case.predicted_smiles}</code>
    </div>
    <div class="smiles true">
      <span class="slabel">True</span>
      <code>{case.true_smiles}</code>
    </div>
  </div>
  <div class="description">{desc}</div>
  {("<div class='note'><strong>Note:</strong> " + case.manual_note + "</div>") if case.manual_note else ""}
  {("<div class='reviewed'>✓ Manually reviewed</div>") if case.manually_reviewed else ""}
</div>"""

        cards_html = "\n".join(
            card(c, plot_paths[c.case_id])
            for c in cases
            if c.case_id in plot_paths
        )

        # Taxonomy summary
        from collections import Counter
        type_counts = Counter(c.failure_type for c in cases)
        taxonomy_rows = "".join(
            f"<tr><td>{ft.replace('_',' ')}</td>"
            f"<td style='color:{FAILURE_COLOURS.get(ft, '#444')}'>■</td>"
            f"<td>{cnt}</td><td>{cnt/len(cases):.1%}</td></tr>"
            for ft, cnt in type_counts.most_common()
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Georgia', serif; background: #f8f7f4;
          color: #2d2d2a; line-height: 1.6; }}
  header {{ background: #1a1a18; color: #e8e6df; padding: 2rem 3rem; }}
  header h1 {{ font-size: 1.6rem; font-weight: 400; margin-bottom: 0.3rem; }}
  header p  {{ font-size: 0.85rem; color: #a8a69e; }}
  .container {{ max-width: 1100px; margin: 2rem auto; padding: 0 2rem; }}
  .taxonomy-table {{ width: 100%; border-collapse: collapse; margin: 1.5rem 0 2.5rem;
                     font-size: 0.85rem; }}
  .taxonomy-table th {{ text-align: left; padding: 6px 12px;
                        border-bottom: 2px solid #ddd; font-weight: 600; }}
  .taxonomy-table td {{ padding: 5px 12px; border-bottom: 1px solid #eee; }}
  .taxonomy-img {{ width: 100%; max-width: 700px; margin: 0 auto 2rem;
                   display: block; border-radius: 6px; }}
  .case-card {{ background: white; border-radius: 8px; margin-bottom: 2.5rem;
                box-shadow: 0 1px 4px rgba(0,0,0,0.08);
                overflow: hidden; }}
  .card-header {{ padding: 0.75rem 1.25rem; display: flex; align-items: center;
                  gap: 12px; flex-wrap: wrap;
                  background: #fafaf8; border-bottom: 1px solid #eee; }}
  .case-id {{ font-family: monospace; font-size: 0.85rem;
              font-weight: 600; color: #444; }}
  .badge {{ font-size: 0.72rem; padding: 2px 8px; border-radius: 12px;
            font-weight: 600; letter-spacing: 0.03em; }}
  .severity {{ font-size: 0.75rem; font-weight: 600; margin-left: auto; }}
  .spectrum-plot {{ width: 100%; border-bottom: 1px solid #eee; }}
  .metrics-row {{ display: flex; gap: 0; border-bottom: 1px solid #eee; }}
  .metric {{ flex: 1; padding: 0.6rem 1rem; text-align: center;
             border-right: 1px solid #eee; }}
  .metric:last-child {{ border-right: none; }}
  .metric.bad {{ background: #fff5f5; }}
  .metric.ok  {{ background: #f5faf5; }}
  .mlabel {{ display: block; font-size: 0.7rem; color: #888;
             text-transform: uppercase; letter-spacing: 0.05em; }}
  .mval   {{ font-size: 1.1rem; font-weight: 600; font-family: monospace; }}
  .smiles-row {{ display: flex; gap: 0; border-bottom: 1px solid #eee; }}
  .smiles {{ flex: 1; padding: 0.7rem 1rem; border-right: 1px solid #eee; }}
  .smiles:last-child {{ border-right: none; }}
  .smiles.pred {{ background: #fff8f8; }}
  .smiles.true {{ background: #f8f8ff; }}
  .slabel {{ display: block; font-size: 0.7rem; color: #888;
             text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 3px; }}
  .smiles code {{ font-size: 0.8rem; word-break: break-all; }}
  .description {{ padding: 0.7rem 1.25rem; font-size: 0.82rem;
                  color: #666; font-style: italic;
                  border-bottom: 1px solid #eee; }}
  .note {{ padding: 0.5rem 1.25rem; font-size: 0.82rem;
           background: #fffbeb; border-left: 3px solid #e8a020; }}
  .reviewed {{ padding: 0.4rem 1.25rem; font-size: 0.78rem;
               color: #1D9E75; font-weight: 600; }}
  h2 {{ font-size: 1.1rem; font-weight: 500; margin: 2rem 0 1rem;
        color: #444; border-bottom: 1px solid #ddd; padding-bottom: 0.4rem; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <p>Physics-residual annotated failure cases · SpectraLM · n={len(cases)} cases</p>
</header>
<div class="container">
  <h2>Taxonomy Summary</h2>
  <img src="figures/taxonomy_summary.png" class="taxonomy-img" alt="Taxonomy summary"/>
  <table class="taxonomy-table">
    <thead><tr><th>Failure Type</th><th>Colour</th><th>Count</th><th>%</th></tr></thead>
    <tbody>{taxonomy_rows}</tbody>
  </table>
  <h2>Failure Cases (ranked by severity)</h2>
  {cards_html}
</div>
</body>
</html>"""

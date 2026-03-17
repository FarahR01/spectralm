# Chemistry Context — Why IR Spectroscopy?

## The Manual Process I Wanted to Automate

During my preparatory cycle, analytical chemistry was one of the most demanding — and most rewarding — parts of the curriculum. Every week we would receive an unknown organic compound and a strip of paper coming out of an IR spectrometer. The task: identify the molecule.

No lookup table. No computer. Just the spectrum, a pen, and the Silverstein reference book.

The process looked like this:

**Step 1 — Scan the functional group region (4000–1500 cm⁻¹)**

You'd start at the high-wavenumber end and work down. A broad absorption centred around 3300 cm⁻¹? O–H stretch — likely an alcohol or carboxylic acid. Sharp peaks at 2850 and 2960 cm⁻¹? C–H stretches — almost certainly an alkane chain. A strong, sharp absorption at 1710 cm⁻¹? Carbonyl — now the question is *which* carbonyl.

**Step 2 — Use the fingerprint region (1500–400 cm⁻¹) to distinguish isomers**

The fingerprint region is where manual interpretation gets hard. Two molecules can look nearly identical in the functional group region but diverge completely below 1500 cm⁻¹. Ortho, meta, and para substitution on a benzene ring — distinguishable only by the pattern of out-of-plane C–H bending modes between 650 and 900 cm⁻¹.

**Step 3 — Cross-reference and propose a structure**

You'd compile your functional group list, apply what you knew about the molecular formula (sometimes given, sometimes inferred from other data), and propose a candidate structure. Then justify every peak you assigned.

This process took 45–90 minutes per unknown compound. A trained analytical chemist could do it in 10–15. A machine — in principle — should do it in milliseconds.

That gap is what motivated SpectraLM.

---

## The Instrumental Technique

Infrared spectroscopy works by passing infrared radiation through a sample and measuring which wavelengths are absorbed. The absorption happens when the radiation frequency matches a natural vibrational frequency of a molecular bond.

```
IR source → sample → detector → spectrum

         Transmittance (%)
    100 |████████████████
        |         ████
     50 |    █████    ████
        |████            ███
      0 |________________________
        4000  3000  2000  1000
              Wavenumber (cm⁻¹)
```

The x-axis is wavenumber (cm⁻¹) — the reciprocal of wavelength in centimetres. Higher wavenumber = higher energy = higher vibrational frequency. The y-axis is transmittance (%) or absorbance — SpectraLM works in absorbance (A = −log₁₀T) because Beer–Lambert law is linear in absorbance, not transmittance.

---

## The Physics SpectraLM Encodes

### Beer–Lambert Law

The foundational relationship in quantitative IR spectroscopy:

```
A(ν) = ε(ν) · c · l
```

Where:
- `A(ν)` = absorbance at wavenumber ν (dimensionless)
- `ε(ν)` = molar attenuation coefficient (L·mol⁻¹·cm⁻¹) — molecule-specific
- `c` = concentration (mol·L⁻¹)
- `l` = path length (cm) — fixed by the sample cell

The key implication: **absorbances are additive**. If a molecule contains two functional groups, the spectrum is the sum of both groups' contributions. This is the physical justification for SpectraLM's `LorentzianPeakModel`, which reconstructs a predicted spectrum by summing Lorentzian peaks for each predicted functional group.

### Group Frequency Theory

Each functional group absorbs at a characteristic, reproducible wavenumber range — largely independent of the rest of the molecule. This is the working assumption of manual IR interpretation and the physical basis for SpectraLM's group frequency table:

| Functional Group | Range (cm⁻¹) | Notes |
|---|---|---|
| O–H stretch (alcohol) | 3200–3550 | Broad due to H-bonding |
| O–H stretch (carboxylic acid) | 2500–3300 | Very broad |
| N–H stretch (primary amine) | 3300–3500 | Two peaks (sym + antisym) |
| C–H stretch (alkane) | 2850–2960 | Always present in organics |
| C–H stretch (aromatic) | 3000–3100 | Weak but diagnostic |
| C≡N stretch (nitrile) | 2200–2260 | Sharp, distinctive |
| C=O stretch (ketone) | 1680–1750 | Strong, sharp |
| C=O stretch (ester) | 1735–1750 | Slightly higher than ketone |
| C=O stretch (carboxylic acid) | 1700–1725 | Paired with broad O–H |
| C=O stretch (amide) | 1630–1690 | Lower than other carbonyls |
| C=C stretch (alkene) | 1620–1680 | Variable intensity |
| C=C stretch (aromatic) | 1450–1600 | Characteristic pattern |
| C–O stretch (ether) | 1000–1150 | Strong |
| N–O stretch (nitro) | 1500–1570 | Two strong bands |

### Lorentzian Peak Shape

IR absorption bands are Lorentzian-shaped (not Gaussian) due to **homogeneous broadening** — specifically lifetime broadening from molecular collisions. The Lorentzian function:

```
A(ν) = I_max / (1 + ((ν − ν₀) / (Δν/2))²)
```

Where `ν₀` is the peak centre and `Δν` is the full width at half maximum (FWHM). SpectraLM's physics head uses learnable Lorentzian peaks — `ν₀` constrained to each group's expected range, `Δν` learned from data.

---

## The Practical Challenges (That the Dataset Reflects)

**Nujol contamination** — The most common sample preparation method for solid compounds in the 1970s–90s (when most NIST database entries were collected) was mulling with Nujol, a mineral oil. Nujol has strong C–H absorptions at 2850–2960 cm⁻¹ and 1375–1460 cm⁻¹ that mask the analyte's peaks in those regions. 41% of SpectraLM's failure cases involve this artefact.

**Regioisomer ambiguity** — Ortho, meta, and para isomers of disubstituted benzenes have nearly identical functional group regions. Distinction requires careful analysis of the 650–900 cm⁻¹ fingerprint region — the hardest region to interpret automatically. 33% of failures.

**Water vapour** — Atmospheric water absorbs strongly at 1550–1650 cm⁻¹ and 3550–3750 cm⁻¹. In older spectra collected without purged optical paths, these bands appear as sharp, irregular spikes superimposed on the sample spectrum.

**Concentration dependence** — Beer–Lambert law means that peak intensities depend on concentration. A dilute sample of a strongly absorbing compound can look similar to a concentrated sample of a weakly absorbing one. SpectraLM's augmentation pipeline includes random Beer–Lambert concentration scaling (×0.6–1.4) to make the model robust to this.

---

## From Manual to Automated

In my preparatory cycle lab sessions, the workflow for a single unknown compound was:

```
Receive spectrum (paper printout)
    ↓
Scan 4000–1500 cm⁻¹ (30 min)
    ↓
List candidate functional groups
    ↓
Scan 1500–400 cm⁻¹ (20 min)
    ↓
Cross-reference with molecular formula
    ↓
Propose structure (15 min)
    ↓
Write justification for every peak
Total: 45–90 minutes
```

SpectraLM collapses this to a single forward pass (~50ms on CPU), returning:
- The predicted SMILES string
- A physics confidence score (ECR) measuring Beer–Lambert consistency
- Per-group confidence scores for each functional group
- A list of any group frequency violations

The ECR is the automated equivalent of the lab instructor asking: *"Is your assignment physically consistent with what you're seeing in the spectrum?"* — the check that a rushed student might skip, but a careful analyst never would.
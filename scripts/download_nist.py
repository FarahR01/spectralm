"""
scripts/download_nist.py

Downloads IR spectra from NIST WebBook as .jdx files into data/raw/.

NIST WebBook is the authoritative public IR spectral database.
Each compound is fetched by CAS registry number and saved as a
JCAMP-DX (.jdx) file — the standard format for spectroscopic data.

Usage:
    python scripts/download_nist.py --output_dir data/raw --delay 1.5
    python scripts/download_nist.py --output_dir data/raw --limit 100  # quick test

Rate limiting: NIST asks for polite crawling (≥1s between requests).
This script enforces a configurable delay and retries on failure.

What gets downloaded:
    data/raw/
        ├── alcohols/
        ├── aldehydes/
        ├── alkanes/
        ├── alkenes/
        ├── alkynes/
        ├── amides/
        ├── amines/
        ├── aromatics/
        ├── carboxylic_acids/
        ├── esters/
        ├── ethers/
        ├── halides/
        ├── ketones/
        └── nitriles/
"""

from __future__ import annotations

import argparse
import time
import json
import re
from pathlib import Path
from urllib.parse import urlencode
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

console = Console()

# ── NIST WebBook URLs ─────────────────────────────────────────────────────────
NIST_BASE      = "https://webbook.nist.gov"
NIST_IR_URL    = f"{NIST_BASE}/cgi/cbook.cgi"
NIST_SEARCH_URL = f"{NIST_BASE}/cgi/cbook.cgi"


# ══════════════════════════════════════════════════════════════════════════════
# CAS number lists by functional group class
# Curated from NIST SRD 35 (the WebBook IR collection)
# ══════════════════════════════════════════════════════════════════════════════

COMPOUND_LIBRARY: dict[str, list[tuple[str, str]]] = {

    "alcohols": [
        ("64-17-5",  "Ethanol"),
        ("71-23-8",  "1-Propanol"),
        ("67-63-0",  "2-Propanol"),
        ("71-36-3",  "1-Butanol"),
        ("78-83-1",  "Isobutanol"),
        ("71-41-0",  "1-Pentanol"),
        ("111-27-3", "1-Hexanol"),
        ("111-70-6", "1-Heptanol"),
        ("111-87-5", "1-Octanol"),
        ("143-08-8", "1-Nonanol"),
        ("112-30-1", "1-Decanol"),
        ("60-12-8",  "2-Phenylethanol"),
        ("100-51-6", "Benzyl alcohol"),
        ("108-93-0", "Cyclohexanol"),
        ("98-85-1",  "1-Phenylethanol"),
        ("97-99-4",  "Tetrahydrofurfuryl alcohol"),
        ("75-65-0",  "tert-Butanol"),
        ("78-92-2",  "2-Butanol"),
        ("626-93-7", "2-Hexanol"),
        ("104-76-7", "2-Ethyl-1-hexanol"),
        ("36653-82-4","1-Hexadecanol"),
        ("112-53-8", "1-Dodecanol"),
        ("75-85-4",  "2-Methyl-2-butanol"),
        ("464-07-3", "3,3-Dimethyl-2-butanol"),
        ("108-11-2", "4-Methyl-2-pentanol"),
    ],

    "ketones": [
        ("67-64-1",  "Acetone"),
        ("78-93-3",  "2-Butanone"),
        ("107-87-9", "2-Pentanone"),
        ("591-78-6", "2-Hexanone"),
        ("110-43-0", "2-Heptanone"),
        ("111-13-7", "2-Octanone"),
        ("821-55-6", "2-Nonanone"),
        ("693-54-9", "2-Decanone"),
        ("96-22-0",  "3-Pentanone"),
        ("106-35-4", "3-Heptanone"),
        ("123-19-3", "4-Heptanone"),
        ("108-10-1", "4-Methyl-2-pentanone"),
        ("108-94-1", "Cyclohexanone"),
        ("98-86-2",  "Acetophenone"),
        ("119-61-9", "Benzophenone"),
        ("105-42-0", "5-Methyl-3-heptanone"),
        ("78-59-1",  "Isophorone"),
        ("110-93-0", "6-Methyl-5-hepten-2-one"),
        ("502-56-7", "5-Nonanone"),
        ("821-55-6", "2-Nonanone"),
    ],

    "carboxylic_acids": [
        ("64-19-7",  "Acetic acid"),
        ("79-09-4",  "Propionic acid"),
        ("107-92-6", "Butyric acid"),
        ("109-52-4", "Pentanoic acid"),
        ("142-62-1", "Hexanoic acid"),
        ("111-14-8", "Heptanoic acid"),
        ("124-07-2", "Octanoic acid"),
        ("112-05-0", "Nonanoic acid"),
        ("334-48-5", "Decanoic acid"),
        ("65-85-0",  "Benzoic acid"),
        ("100-47-0", "Benzonitrile"),
        ("103-82-2", "Phenylacetic acid"),
        ("140-10-3", "trans-Cinnamic acid"),
        ("79-41-4",  "Methacrylic acid"),
        ("79-10-7",  "Acrylic acid"),
        ("77-92-9",  "Citric acid"),
        ("87-69-4",  "Tartaric acid"),
        ("69-65-8",  "D-Mannitol"),
        ("50-21-5",  "Lactic acid"),
        ("79-33-4",  "L-Lactic acid"),
    ],

    "esters": [
        ("79-20-9",  "Methyl acetate"),
        ("141-78-6", "Ethyl acetate"),
        ("109-60-4", "Propyl acetate"),
        ("123-86-4", "Butyl acetate"),
        ("628-63-7", "Pentyl acetate"),
        ("142-92-7", "Hexyl acetate"),
        ("112-14-1", "Octyl acetate"),
        ("103-09-3", "2-Ethylhexyl acetate"),
        ("105-37-3", "Ethyl propanoate"),
        ("105-54-4", "Ethyl butanoate"),
        ("539-82-2", "Ethyl pentanoate"),
        ("123-66-0", "Ethyl hexanoate"),
        ("106-30-9", "Ethyl heptanoate"),
        ("106-32-1", "Ethyl octanoate"),
        ("123-29-5", "Ethyl nonanoate"),
        ("110-38-3", "Ethyl decanoate"),
        ("93-89-0",  "Ethyl benzoate"),
        ("120-50-3", "Isobutyl benzoate"),
        ("532-32-1", "Sodium benzoate"),
        ("94-09-7",  "Ethyl 4-aminobenzoate"),
    ],

    "amines": [
        ("75-04-7",  "Ethylamine"),
        ("107-10-8", "Propylamine"),
        ("109-73-9", "Butylamine"),
        ("110-58-7", "Pentylamine"),
        ("111-26-2", "Hexylamine"),
        ("111-68-2", "Heptylamine"),
        ("111-86-4", "Octylamine"),
        ("112-20-9", "Nonylamine"),
        ("2016-57-1","Decylamine"),
        ("100-46-9", "Benzylamine"),
        ("62-53-3",  "Aniline"),
        ("95-51-2",  "2-Chloroaniline"),
        ("106-47-8", "4-Chloroaniline"),
        ("100-61-8", "N-Methylaniline"),
        ("121-69-7", "N,N-Dimethylaniline"),
        ("103-69-5", "N-Ethylaniline"),
        ("91-66-7",  "N,N-Diethylaniline"),
        ("108-91-8", "Cyclohexylamine"),
        ("75-31-0",  "Isopropylamine"),
        ("75-64-9",  "tert-Butylamine"),
    ],

    "aldehydes": [
        ("75-07-0",  "Acetaldehyde"),
        ("123-72-8", "Butyraldehyde"),
        ("110-62-3", "Valeraldehyde"),
        ("66-25-1",  "Hexanal"),
        ("111-71-7", "Heptanal"),
        ("124-13-0", "Octanal"),
        ("124-19-6", "Nonanal"),
        ("112-31-2", "Decanal"),
        ("100-52-7", "Benzaldehyde"),
        ("104-55-2", "Cinnamaldehyde"),
        ("122-78-1", "Phenylacetaldehyde"),
        ("120-57-0", "Piperonal"),
        ("90-02-8",  "Salicylaldehyde"),
        ("591-31-1", "3-Methoxybenzaldehyde"),
        ("591-68-4", "4-Heptanone"),
        ("4748-78-1","Ethyl (E)-2-methylbut-2-enoate"),
        ("123-11-5", "4-Methoxybenzaldehyde"),
        ("99-73-0",  "4-Bromobenzaldehyde"),
        ("555-16-8", "4-Nitrobenzaldehyde"),
        ("934-74-7", "3-Ethylbenzaldehyde"),
    ],

    "alkanes": [
        ("74-82-8",  "Methane"),
        ("74-84-0",  "Ethane"),
        ("74-98-6",  "Propane"),
        ("106-97-8", "Butane"),
        ("109-66-0", "Pentane"),
        ("110-54-3", "Hexane"),
        ("142-82-5", "Heptane"),
        ("111-65-9", "Octane"),
        ("111-84-2", "Nonane"),
        ("124-18-5", "Decane"),
        ("1120-21-4","Undecane"),
        ("112-40-3", "Dodecane"),
        ("629-50-5", "Tridecane"),
        ("629-59-4", "Tetradecane"),
        ("629-62-9", "Pentadecane"),
        ("544-76-3", "Hexadecane"),
        ("75-28-5",  "Isobutane"),
        ("78-78-4",  "Isopentane"),
        ("107-83-5", "2-Methylpentane"),
        ("96-14-0",  "3-Methylpentane"),
    ],

    "aromatics": [
        ("71-43-2",  "Benzene"),
        ("108-88-3", "Toluene"),
        ("100-41-4", "Ethylbenzene"),
        ("106-42-3", "p-Xylene"),
        ("108-38-3", "m-Xylene"),
        ("95-47-6",  "o-Xylene"),
        ("98-82-8",  "Cumene"),
        ("103-65-1", "Propylbenzene"),
        ("104-51-8", "Butylbenzene"),
        ("135-98-8", "sec-Butylbenzene"),
        ("98-06-6",  "tert-Butylbenzene"),
        ("99-87-6",  "4-Isopropyltoluene"),
        ("91-20-3",  "Naphthalene"),
        ("95-63-6",  "1,2,4-Trimethylbenzene"),
        ("108-67-8", "1,3,5-Trimethylbenzene"),
        ("100-66-3", "Anisole"),
        ("104-45-0", "4-Propylanisole"),
        ("93-58-3",  "Methyl benzoate"),
        ("99-08-1",  "3-Nitrotoluene"),
        ("99-09-2",  "3-Nitroaniline"),
    ],

    "alkenes": [
        ("74-85-1",  "Ethylene"),
        ("115-07-1", "Propylene"),
        ("106-98-9", "1-Butene"),
        ("109-67-1", "1-Pentene"),
        ("592-41-6", "1-Hexene"),
        ("592-76-7", "1-Heptene"),
        ("111-66-0", "1-Octene"),
        ("124-11-8", "1-Nonene"),
        ("872-05-9", "1-Decene"),
        ("563-45-1", "3-Methyl-1-butene"),
        ("107-39-1", "2,4,4-Trimethyl-1-pentene"),
        ("25167-70-8","Diisobutylene"),
        ("100-42-5", "Styrene"),
        ("98-83-9",  "Alpha-methylstyrene"),
        ("100-80-1", "3-Methylstyrene"),
        ("622-97-9", "4-Methylstyrene"),
        ("768-00-3", "3-Chlorostyrene"),
        ("104-12-1", "4-Chlorophenyl isocyanate"),
        ("627-20-3", "cis-2-Pentene"),
        ("646-04-8", "trans-2-Pentene"),
    ],

    "nitriles": [
        ("75-05-8",  "Acetonitrile"),
        ("107-12-0", "Propionitrile"),
        ("109-74-0", "Butyronitrile"),
        ("110-59-8", "Valeronitrile"),
        ("628-73-9", "Hexanenitrile"),
        ("629-08-3", "Heptanenitrile"),
        ("124-12-9", "Octanenitrile"),
        ("2243-27-8","Nonanenitrile"),
        ("1975-78-6","Decanenitrile"),
        ("100-47-0", "Benzonitrile"),
        ("873-74-5", "4-Methylbenzonitrile"),
        ("19340-93-3","4-Ethylbenzonitrile"),
        ("97-14-3",  "2-Methylbenzonitrile"),
        ("102-77-2", "2-Methoxybenzonitrile"),
        ("2243-62-1","1,5-Naphthalenediamine"),
        ("544-13-8", "Glutaronitrile"),
        ("110-61-2", "Succinonitrile"),
        ("111-69-3", "Adiponitrile"),
        ("142-68-7", "Tetrahydropyran-2-carbonitrile"),
        ("4553-62-2","3-Butenenitrile"),
    ],

    "amides": [
        ("60-35-5",  "Acetamide"),
        ("79-55-0",  "Propanamide"),
        ("541-35-5", "Butanamide"),
        ("625-77-4", "Pentanamide"),
        ("628-02-4", "Hexanamide"),
        ("628-61-5", "Heptanamide"),
        ("629-01-6", "Octanamide"),
        ("1120-16-7","Decanamide"),
        ("55-21-0",  "Benzamide"),
        ("588-16-9", "2-Methylbenzamide"),
        ("619-55-6", "3-Methylbenzamide"),
        ("619-56-7", "4-Methylbenzamide"),
        ("555-30-6", "Methyldopa"),
        ("79-16-3",  "N-Methylacetamide"),
        ("127-19-5", "N,N-Dimethylacetamide"),
        ("68-12-2",  "N,N-Dimethylformamide"),
        ("77-81-6",  "Tabun"),
        ("70-69-9",  "4-Aminopropiophenone"),
        ("103-84-4", "Acetanilide"),
        ("537-92-8", "N-Methylbenzamide"),
    ],

    "ethers": [
        ("60-29-7",  "Diethyl ether"),
        ("628-32-0", "Ethyl propyl ether"),
        ("142-96-1", "Dibutyl ether"),
        ("693-65-2", "Dipentyl ether"),
        ("629-14-1", "Ethylene glycol diethyl ether"),
        ("110-71-4", "1,2-Dimethoxyethane"),
        ("111-96-6", "Diethylene glycol dimethyl ether"),
        ("112-49-2", "Triethylene glycol dimethyl ether"),
        ("109-99-9", "Tetrahydrofuran"),
        ("142-68-7", "Tetrahydropyran"),
        ("100-66-3", "Anisole"),
        ("103-50-4", "Dibenzyl ether"),
        ("104-46-1", "trans-Anethole"),
        ("91-16-7",  "1,2-Dimethoxybenzene"),
        ("151-10-0", "1,3-Dimethoxybenzene"),
        ("150-78-7", "1,4-Dimethoxybenzene"),
        ("1746-13-0","Allyl ether"),
        ("557-17-5", "Methyl propyl ether"),
        ("628-28-4", "Methyl butyl ether"),
        ("994-05-8", "Methyl tert-amyl ether"),
    ],

    "halides": [
        ("75-09-2",  "Dichloromethane"),
        ("67-66-3",  "Chloroform"),
        ("56-23-5",  "Carbon tetrachloride"),
        ("75-34-3",  "1,1-Dichloroethane"),
        ("107-06-2", "1,2-Dichloroethane"),
        ("71-55-6",  "1,1,1-Trichloroethane"),
        ("79-00-5",  "1,1,2-Trichloroethane"),
        ("106-93-4", "1,2-Dibromoethane"),
        ("74-97-5",  "Bromochloromethane"),
        ("108-86-1", "Bromobenzene"),
        ("108-90-7", "Chlorobenzene"),
        ("541-73-1", "1,3-Dichlorobenzene"),
        ("106-46-7", "1,4-Dichlorobenzene"),
        ("95-50-1",  "1,2-Dichlorobenzene"),
        ("100-44-7", "Benzyl chloride"),
        ("98-87-3",  "Benzal chloride"),
        ("76-06-2",  "Chloropicrin"),
        ("76-01-7",  "Pentachloroethane"),
        ("87-61-6",  "1,2,3-Trichlorobenzene"),
        ("120-82-1", "1,2,4-Trichlorobenzene"),
    ],

    "alkynes": [
        ("74-86-2",  "Acetylene"),
        ("74-99-7",  "Propyne"),
        ("107-00-6", "1-Butyne"),
        ("503-17-3", "2-Butyne"),
        ("627-19-0", "1-Pentyne"),
        ("627-21-4", "2-Pentyne"),
        ("693-02-7", "1-Hexyne"),
        ("928-49-4", "2-Hexyne"),
        ("2586-89-2","3-Hexyne"),
        ("628-71-7", "1-Heptyne"),
        ("1119-65-9","1-Octyne"),
        ("3452-09-3","1-Nonyne"),
        ("764-93-2", "1-Decyne"),
        ("536-74-3", "Phenylacetylene"),
        ("622-31-1", "3-Phenyl-1-propyne"),
        ("764-01-2", "1-Penten-4-yne"),
        ("2004-69-5","5-Decyne"),
        ("17530-24-4","4-Octyne"),
        ("1942-45-6","3-Octyne"),
        ("14272-03-8","3-Heptyne"),
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# Downloader
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DownloadStats:
    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    no_ir_data: int = 0


def make_session(retries: int = 3) -> requests.Session:
    """HTTP session with retry logic and polite headers."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": (
            "SpectraLM-Research/1.0 "
            "(academic spectral database construction; "
            "contact: your@email.com)"
        ),
        "Accept": "text/plain, application/octet-stream",
    })
    return session


def fetch_jdx(
    session: requests.Session,
    cas: str,
    timeout: int = 15,
) -> str | None:
    """
    Fetch a JCAMP-DX IR spectrum from NIST WebBook by CAS number.
    Returns the raw .jdx text content, or None on failure.
    """
    params = {
        "JCAMP":  cas,
        "Type":   "IR",
        "Index":  "0",
    }
    url = f"{NIST_IR_URL}?{urlencode(params)}"

    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()

        content = resp.text
        # JCAMP-DX files start with "##TITLE="
        if "##TITLE=" not in content:
            return None
        # Verify it's an IR spectrum (not mass spec etc.)
        if "INFRARED" not in content.upper() and "IR" not in content[:200].upper():
            return None

        return content

    except requests.exceptions.RequestException:
        return None


def extract_smiles_from_nist(
    session: requests.Session,
    cas: str,
    timeout: int = 10,
) -> str | None:
    """
    Scrape the SMILES string from the NIST WebBook compound page.
    NIST doesn't include SMILES in the .jdx file directly — it's on the HTML page.
    """
    params = {"ID": cas, "Units": "SI"}
    url = f"{NIST_SEARCH_URL}?{urlencode(params)}"

    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()

        # SMILES appears in the page as: "SMILES: <string>"
        match = re.search(r'<li>SMILES\s*:\s*([^\s<]+)', resp.text)
        if match:
            return match.group(1).strip()

        # Fallback: InChI-derived SMILES via RDKit
        inchi_match = re.search(r'InChI=([^\s"<]+)', resp.text)
        if inchi_match:
            inchi = "InChI=" + inchi_match.group(1)
            try:
                from rdkit import Chem
                from rdkit.Chem.inchi import MolFromInchi
                mol = MolFromInchi(inchi)
                if mol:
                    return Chem.MolToSmiles(mol)
            except Exception:
                pass

        return None

    except requests.exceptions.RequestException:
        return None


def inject_smiles_into_jdx(jdx_content: str, smiles: str) -> str:
    """
    Inject SMILES into the JDX content as a custom field.
    This makes the file self-contained for our parser.
    """
    insertion = f"##$SMILES={smiles}\n"
    # Insert after the first ##TITLE line
    lines = jdx_content.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("##TITLE="):
            lines.insert(i + 1, insertion.strip())
            break
    return "\n".join(lines)


def download_compound(
    session: requests.Session,
    cas: str,
    name: str,
    output_path: Path,
    delay: float = 1.5,
) -> bool:
    """
    Download a single compound's IR spectrum and SMILES.
    Returns True on success.
    """
    if output_path.exists():
        return True   # already downloaded — skip

    # Fetch JDX
    jdx_content = fetch_jdx(session, cas)
    time.sleep(delay * 0.6)  # partial delay between JDX and SMILES fetches

    if jdx_content is None:
        return False

    # Fetch SMILES
    smiles = extract_smiles_from_nist(session, cas)
    time.sleep(delay * 0.4)

    # Inject SMILES if found
    if smiles:
        jdx_content = inject_smiles_into_jdx(jdx_content, smiles)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(jdx_content, encoding="utf-8")
    return True


def download_all(
    output_dir: Path,
    delay: float = 1.5,
    limit: int | None = None,
    classes: list[str] | None = None,
    dry_run: bool = False,
) -> DownloadStats:
    """
    Download all compounds in COMPOUND_LIBRARY.

    Args:
        output_dir : root directory for .jdx files
        delay      : seconds between requests (be polite to NIST)
        limit      : max total compounds to download (None = all)
        classes    : subset of functional group classes (None = all)
        dry_run    : print what would be downloaded without downloading
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = DownloadStats()
    session = make_session()

    target_classes = classes or list(COMPOUND_LIBRARY.keys())
    compounds = []
    for cls in target_classes:
        if cls not in COMPOUND_LIBRARY:
            console.print(f"[yellow]Unknown class: {cls}[/yellow]")
            continue
        for cas, name in COMPOUND_LIBRARY[cls]:
            compounds.append((cls, cas, name))

    if limit:
        compounds = compounds[:limit]

    stats.total = len(compounds)

    if dry_run:
        console.print(f"[bold]Dry run:[/bold] would download {len(compounds)} compounds")
        for cls, cas, name in compounds[:10]:
            console.print(f"  [{cls}] {name} (CAS: {cas})")
        if len(compounds) > 10:
            console.print(f"  ... and {len(compounds) - 10} more")
        return stats

    console.print(f"\n[bold]Downloading {len(compounds)} IR spectra from NIST WebBook[/bold]")
    console.print(f"Output: {output_dir}")
    console.print(f"Delay:  {delay}s between requests")
    console.print(f"Estimated time: {len(compounds) * delay / 60:.0f} min\n")

    failed_log = []

    with tqdm(total=len(compounds), unit="compound") as pbar:
        for cls, cas, name in compounds:
            pbar.set_description(f"{cls}/{name[:20]}")
            output_path = output_dir / cls / f"{cas}.jdx"

            if output_path.exists():
                stats.skipped += 1
                pbar.update(1)
                continue

            success = download_compound(session, cas, name, output_path, delay)

            if success:
                stats.success += 1
            else:
                stats.failed += 1
                failed_log.append({"cas": cas, "name": name, "class": cls})

            pbar.update(1)

    # Save failure log
    if failed_log:
        fail_path = output_dir / "download_failures.json"
        with open(fail_path, "w") as f:
            json.dump(failed_log, f, indent=2)
        console.print(f"\n[yellow]Failed downloads logged →[/yellow] {fail_path}")

    # Print summary table
    table = Table(title="Download Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("Total compounds",  str(stats.total))
    table.add_row("Downloaded",       f"[green]{stats.success}[/green]")
    table.add_row("Skipped (cached)", str(stats.skipped))
    table.add_row("Failed",           f"[red]{stats.failed}[/red]")
    console.print(table)

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NIST IR spectra")
    parser.add_argument("--output_dir", type=str, default="data/raw",
                        help="Root output directory for .jdx files")
    parser.add_argument("--delay",      type=float, default=1.5,
                        help="Seconds between HTTP requests (default: 1.5)")
    parser.add_argument("--limit",      type=int,   default=None,
                        help="Max compounds to download (default: all ~280)")
    parser.add_argument("--classes",    type=str,   nargs="+", default=None,
                        help="Functional group classes to download")
    parser.add_argument("--dry_run",    action="store_true",
                        help="Print what would be downloaded, don't fetch")
    args = parser.parse_args()

    stats = download_all(
        output_dir=Path(args.output_dir),
        delay=args.delay,
        limit=args.limit,
        classes=args.classes,
        dry_run=args.dry_run,
    )

    if stats.success > 0:
        console.print(f"\n[bold green]✓ Done.[/bold green] "
                      f"Run [bold]scripts/preprocess.py[/bold] next to fill data/processed/")
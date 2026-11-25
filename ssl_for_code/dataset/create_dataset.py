"""Download, filter, and concatenate open-source Python repositories into train/valid splits."""

from __future__ import annotations

import argparse
import random
import re
import shutil
import string
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import List, NamedTuple, Optional, Set, Tuple

from ssl_for_code.utils.io import get_data_path

PRINTABLE = frozenset(string.printable)
AWESOME_PYTORCH_URL = (
    "https://raw.githubusercontent.com/bharathgs/Awesome-pytorch-list/master/README.md"
)


class PythonFile(NamedTuple):
    relative_path: str
    project: str
    path: Path


def get_python_files(source_path: Path) -> List[PythonFile]:
    """Recursively collect all ``.py`` files under ``source_path``."""
    files: List[PythonFile] = []
    if not source_path.exists():
        return files

    for path in sorted(source_path.rglob("*.py")):
        if not path.is_file() or path.is_symlink():
            continue
        rel_parts = path.relative_to(source_path).parts
        project = rel_parts[0] if len(rel_parts) > 1 else source_path.name
        rel_path = str(Path(*rel_parts[1:])) if len(rel_parts) > 1 else path.name
        files.append(PythonFile(relative_path=rel_path, project=project, path=path))

    return files


def read_clean_file(path: Path) -> str:
    """Read a Python source file and strip non-printable characters."""
    content = path.read_text(encoding="utf-8", errors="ignore")
    return "".join(ch for ch in content if ch in PRINTABLE)


def concat_and_save(output_path: Path, source_files: List[PythonFile]) -> int:
    """Concatenate Python files with project/path header comments; return char count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_chars = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for source in source_files:
            header = f"# PROJECT: {source.project} FILE: {source.relative_path}\n"
            body = read_clean_file(source.path) + "\n"
            handle.write(header)
            handle.write(body)
            total_chars += len(header) + len(body)
    return total_chars


def get_repos_from_readme(readme_path: Path) -> List[Tuple[str, str]]:
    """Extract unique ``(owner, repo)`` pairs from GitHub links in a Markdown file."""
    content = readme_path.read_text(encoding="utf-8", errors="ignore")
    link_pattern = re.compile(r"\[(?P<title>[^\]]*)\]\((?P<url>[^\)]*)\)")
    repo_pattern = re.compile(r"https://github\.com/(?P<user>[^/]+)/(?P<repo>[^/#?]+)$")

    seen: Set[Tuple[str, str]] = set()
    repos: List[Tuple[str, str]] = []
    for _, url in link_pattern.findall(content):
        match = repo_pattern.match(url.strip())
        if match:
            pair = (match.group("user"), match.group("repo"))
            if pair not in seen:
                seen.add(pair)
                repos.append(pair)
    return repos


def fetch_awesome_pytorch_readme(data_dir: Path) -> Path:
    """Download the Awesome PyTorch Markdown list into ``data_dir``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "pytorch_awesome.md"
    with urllib.request.urlopen(AWESOME_PYTORCH_URL, timeout=30) as response:
        target.write_bytes(response.read())
    return target


def download_repo(
    org: str,
    repo: str,
    download_dir: Path,
    branch: str = "master",
) -> Optional[Path]:
    """Download a GitHub repository archive zip into ``download_dir``."""
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_file = download_dir / f"{org}_{repo}.zip"
    if zip_file.exists():
        return zip_file

    for candidate_branch in (branch, "main"):
        url = f"https://github.com/{org}/{repo}/archive/refs/heads/{candidate_branch}.zip"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                zip_file.write_bytes(response.read())
            return zip_file
        except urllib.error.HTTPError:
            continue
    return None


def extract_zip(file_path: Path, source_dir: Path, overwrite: bool = False) -> Optional[Path]:
    """Extract a downloaded repository archive into ``source_dir``."""
    source_dir.mkdir(parents=True, exist_ok=True)
    repo_source = source_dir / file_path.stem
    if repo_source.exists():
        if overwrite:
            shutil.rmtree(repo_source)
        else:
            return repo_source
    try:
        with zipfile.ZipFile(file_path, "r") as archive:
            archive.extractall(repo_source)
        return repo_source
    except zipfile.BadZipFile:
        return None


def remove_non_matching_files(path: Path, keep_suffixes: Set[str]) -> None:
    """Remove all files whose suffix is not in ``keep_suffixes``."""
    for entry in path.iterdir():
        if entry.is_symlink():
            entry.unlink()
        elif entry.is_dir():
            remove_non_matching_files(entry, keep_suffixes)
        elif entry.suffix not in keep_suffixes:
            entry.unlink()


def build_splits(
    source_dir: Path,
    output_dir: Path,
    *,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[int, int]:
    """Shuffle Python files under ``source_dir`` and write ``train.py`` & ``valid.py``."""
    files = get_python_files(source_dir)
    rng = random.Random(seed)
    rng.shuffle(files)

    split_idx = max(1, int(len(files) * train_ratio)) if len(files) > 1 else len(files)
    train_files = files[:split_idx]
    valid_files = files[split_idx:] if split_idx < len(files) else files[:1]

    concat_and_save(output_dir / "train.py", train_files)
    concat_and_save(output_dir / "valid.py", valid_files)
    return len(train_files), len(valid_files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train.py / valid.py splits from GitHub repositories or a local directory."
    )
    parser.add_argument(
        "--num-repos",
        type=int,
        default=10,
        help="Number of Awesome-PyTorch repositories to download (default: 10)",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Optional local directory of Python files (skips GitHub downloads)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=get_data_path(),
        help="Output directory for train.py and valid.py (default: data/)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of files assigned to the training split (default: 0.9)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for file shuffling")
    args = parser.parse_args()

    if args.source_dir is not None:
        source_dir = args.source_dir
    else:
        readme_path = args.output_dir / "pytorch_awesome.md"
        if not readme_path.exists():
            fetch_awesome_pytorch_readme(args.output_dir)
        repos = get_repos_from_readme(readme_path)[: args.num_repos]
        download_dir = args.output_dir / "download"
        source_dir = args.output_dir / "source"
        for org, repo in repos:
            zip_path = download_repo(org, repo, download_dir)
            if zip_path is not None:
                extract_zip(zip_path, source_dir)
        if source_dir.exists():
            remove_non_matching_files(source_dir, {".py"})

    n_train, n_valid = build_splits(
        source_dir, args.output_dir, train_ratio=args.train_ratio, seed=args.seed
    )
    print(f"Created dataset splits in {args.output_dir}: {n_train} train files, {n_valid} valid files.")


if __name__ == "__main__":
    main()

import argparse
import random
import shutil
import subprocess
import zipfile
from pathlib import Path

from torchvision.datasets import Food101


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def run_command(command: list[str]):
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Error ejecutando: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def download_kaggle_dataset(dataset_ref: str, raw_dir: Path, force: bool = False):
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing_zip_files = list(raw_dir.glob("*.zip"))
    if existing_zip_files and not force:
        print(f"Se encontró zip existente en {raw_dir}. Se omite descarga.")
        return

    command = [
        "python",
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_ref,
        "-p",
        str(raw_dir),
    ]

    print(f"Descargando dataset Kaggle: {dataset_ref}")
    run_command(command)


def extract_zip_files(raw_dir: Path):
    zip_files = list(raw_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(
            f"No se encontraron .zip en {raw_dir}. Verifica la descarga de Kaggle."
        )

    for zip_file in zip_files:
        target_folder = raw_dir / zip_file.stem
        if target_folder.exists():
            print(f"Zip ya extraído: {zip_file.name}")
            continue

        print(f"Extrayendo {zip_file.name}...")
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(target_folder)


def download_food101_with_torchvision(raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Descargando Food-101 con torchvision (internet)...")
    Food101(root=str(raw_dir), split="train", download=True)
    Food101(root=str(raw_dir), split="test", download=True)

    candidate = raw_dir / "food-101" / "images"
    if not candidate.exists():
        raise FileNotFoundError(
            "No se encontró food-101/images tras la descarga. Revisa permisos o conexión."
        )

    return candidate


def find_class_folders(source_root: Path):
    class_folders = []

    direct_children = [child for child in source_root.iterdir() if child.is_dir()]
    has_images_by_class = all(any(f.suffix.lower() in IMAGE_EXTENSIONS for f in child.rglob("*")) for child in direct_children) if direct_children else False

    if has_images_by_class:
        return direct_children

    for candidate in source_root.rglob("*"):
        if not candidate.is_dir():
            continue
        image_files = [f for f in candidate.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        if image_files:
            class_folders.append(candidate)

    unique = {}
    for folder in class_folders:
        if folder.name not in unique:
            unique[folder.name] = folder

    return list(unique.values())


def collect_images(class_folder: Path):
    images = [f for f in class_folder.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
    return images


def split_images(images: list[Path], train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("Las proporciones de train/val/test deben sumar 1.0")

    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    return train, val, test


def copy_split(images: list[Path], destination_folder: Path):
    destination_folder.mkdir(parents=True, exist_ok=True)
    for source_image in images:
        destination_file = destination_folder / source_image.name
        if destination_file.exists():
            stem = destination_file.stem
            suffix = destination_file.suffix
            index = 1
            while destination_file.exists():
                destination_file = destination_folder / f"{stem}_{index}{suffix}"
                index += 1
        shutil.copy2(source_image, destination_file)


def build_dataset(source_images_root: Path, output_dir: Path, train_ratio: float, val_ratio: float, test_ratio: float, seed: int, min_images_per_class: int):
    if not source_images_root.exists():
        raise FileNotFoundError(f"No existe source_images_root: {source_images_root}")

    class_folders = find_class_folders(source_images_root)
    if not class_folders:
        raise RuntimeError(
            "No se detectaron carpetas de clases con imágenes. "
            "Verifica la ruta --source-images-root."
        )

    if output_dir.exists():
        print(f"Limpiando dataset existente en {output_dir}")
        shutil.rmtree(output_dir)

    split_counts = {"train": 0, "val": 0, "test": 0}
    used_classes = 0

    for class_folder in sorted(class_folders, key=lambda p: p.name):
        class_name = class_folder.name
        images = collect_images(class_folder)

        if len(images) < min_images_per_class:
            print(
                f"Clase '{class_name}' omitida: {len(images)} imágenes "
                f"(< {min_images_per_class})"
            )
            continue

        train, val, test = split_images(images, train_ratio, val_ratio, test_ratio, seed)

        copy_split(train, output_dir / "train" / class_name)
        copy_split(val, output_dir / "val" / class_name)
        copy_split(test, output_dir / "test" / class_name)

        split_counts["train"] += len(train)
        split_counts["val"] += len(val)
        split_counts["test"] += len(test)
        used_classes += 1

    if used_classes == 0:
        raise RuntimeError("No se pudo construir el dataset: no hay clases válidas.")

    print("Dataset creado correctamente:")
    print(f"- Clases usadas: {used_classes}")
    print(f"- Train: {split_counts['train']} imágenes")
    print(f"- Val:   {split_counts['val']} imágenes")
    print(f"- Test:  {split_counts['test']} imágenes")
    print(f"- Ruta final: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepara dataset de alimentos para entrenamiento")

    parser.add_argument(
        "--download-kaggle",
        action="store_true",
        help="Descarga dataset de Kaggle antes de preparar los splits",
    )
    parser.add_argument(
        "--download-food101",
        action="store_true",
        help="Descarga Food-101 con torchvision y prepara los splits",
    )
    parser.add_argument(
        "--kaggle-dataset",
        type=str,
        default="kmader/food41",
        help="Referencia del dataset en Kaggle (owner/dataset)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("./dataset_raw"),
        help="Directorio donde se descarga y extrae Kaggle",
    )
    parser.add_argument(
        "--source-images-root",
        type=Path,
        default=None,
        help=(
            "Ruta de carpetas de clases con imágenes. "
            "Si no se indica, se intenta detectar automáticamente dentro de --raw-dir"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./dataset"),
        help="Directorio final con estructura train/val/test",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-images-per-class", type=int, default=10)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Fuerza nueva descarga aunque exista zip",
    )

    return parser.parse_args()


def auto_detect_source_images_root(raw_dir: Path):
    candidates = [p for p in raw_dir.rglob("*") if p.is_dir()]

    scored = []
    for candidate in candidates:
        children = [child for child in candidate.iterdir() if child.is_dir()]
        if len(children) < 2:
            continue

        with_images = 0
        for child in children:
            has_image = any(f.suffix.lower() in IMAGE_EXTENSIONS for f in child.rglob("*"))
            if has_image:
                with_images += 1

        if with_images >= 2:
            scored.append((with_images, candidate))

    if not scored:
        return None

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def main():
    args = parse_args()

    source_images_root = args.source_images_root

    if args.download_food101:
        source_images_root = download_food101_with_torchvision(args.raw_dir)

    if args.download_kaggle:
        download_kaggle_dataset(args.kaggle_dataset, args.raw_dir, force=args.force_download)
        extract_zip_files(args.raw_dir)

    if source_images_root is None:
        source_images_root = auto_detect_source_images_root(args.raw_dir)
        if source_images_root is None:
            raise RuntimeError(
                "No se pudo detectar automáticamente la carpeta de imágenes por clases. "
                "Indica --source-images-root manualmente."
            )

    print(f"Fuente de imágenes detectada: {source_images_root}")

    build_dataset(
        source_images_root=source_images_root,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_images_per_class=args.min_images_per_class,
    )


if __name__ == "__main__":
    main()

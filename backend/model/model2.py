import argparse
import copy
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image


IMAGE_SIZE = 240


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=4),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.97, 1.03)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.05, scale=(0.02, 0.06)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(272),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def create_dataloaders(dataset_root: Path, batch_size: int = 32):
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            "La estructura de datos debe incluir carpetas train/, val/ y test/ "
            "con subcarpetas por clase (ej: train/pizza, train/sushi...)."
        )

    train_transform, eval_transform = build_transforms()

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    num_workers = min(8, os.cpu_count() or 1)
    use_cuda = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, train_dataset.classes


def create_model(num_classes: int) -> nn.Module:
    # EfficientNet-B1 da un mejor equilibrio entre capacidad y generalizacion.
    model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)

    # Congelamos la red base.
    for param in model.parameters():
        param.requires_grad = False

    # Etapa inicial: solo el head (sin backbone) para estabilizar el arranque.
    in_features = model.classifier[1].in_features

    # Head compacto para limitar memorizacion.
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.45, inplace=True),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.SiLU(),
        nn.Dropout(0.35),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.SiLU(),
        nn.Dropout(0.25),
        nn.Linear(128, num_classes),
    )

    return model


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float):
    if alpha <= 0 or images.size(0) < 2:
        return images, labels, labels, 1.0

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    labels_a, labels_b = labels, labels[index]
    return mixed_images, labels_a, labels_b, lam


def build_optimizer(model: nn.Module, lr: float, weight_decay: float = 3e-4, backbone_mult: float = 0.15):
    def is_head_param(name: str) -> bool:
        return name.startswith("fc.") or name.startswith("classifier.")

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not is_head_param(n)]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and is_head_param(n)]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * backbone_mult},
            {"params": head_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )
    return optimizer


def update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float):
    with torch.no_grad():
        ema_state = ema_model.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key]
            if model_value.dtype.is_floating_point:
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)


def train_model(
    dataset_root: Path,
    output_dir: Path,
    epochs: int = 120,
    batch_size: int = 32,
    lr: float = 1.2e-4,
    patience: int = 30,
    mixup_alpha: float = 0.2,
    mixup_prob: float = 0.5,
    unfreeze_epoch: int = 0,
    head_only_epochs: int = 6,
    label_smoothing: float = 0.12,
    ema_decay: float = 0.999,
    overfit_patience: int = 5,
    overfit_min_delta: float = 1e-3,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    early_stopping_enabled = patience > 0

    train_loader, val_loader, test_loader, classes = create_dataloaders(dataset_root, batch_size)

    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Entrenando en GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("Entrenando en CPU", flush=True)

    model = create_model(num_classes=len(classes)).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad = False

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = build_optimizer(model, lr=lr, weight_decay=2e-3, backbone_mult=0.08)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=lr * 0.02,
    )

    # --- Checkpointing estilo Keras ModelCheckpoint ---
    # monitor = val_accuracy (val_acc) ; mode = "max"
    best_val_acc = -1.0
    epochs_without_improvement = 0

    best_model_path = output_dir / "food_classifier.pth"   # BEST (save_best_only)
    last_model_path = output_dir / "last_model.pth"        # LAST (cada epoch)
    classes_path = output_dir / "classes.json"

    stage1_unfreeze_done = False
    stage2_unfreeze_done = False
    overfit_streak = 0
    prev_train_loss = None
    prev_val_loss = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        steps_per_epoch = len(train_loader)

        print(f"\nIniciando epoch {epoch + 1}/{epochs}...", flush=True)

        # Fase 1: opcional, desbloquea el ultimo bloque tras calentar el head.
        if (not stage1_unfreeze_done) and head_only_epochs > 0 and (epoch + 1) >= head_only_epochs:
            for param in model.features[7].parameters():
                param.requires_grad = True
            lr_stage1 = max(lr * 0.4, 1e-5)
            optimizer = build_optimizer(model, lr=lr_stage1, weight_decay=1.5e-3, backbone_mult=0.06)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=4,
                min_lr=lr_stage1 * 0.02,
            )
            stage1_unfreeze_done = True
            print(f"Fase 1 activada: bloque 7 desbloqueado desde epoch {epoch + 1}.", flush=True)

        # Fase 2: desbloqueamos penultimo bloque para ajuste fino.
        if (not stage2_unfreeze_done) and unfreeze_epoch > 0 and (epoch + 1) >= unfreeze_epoch:
            for param in model.features[5].parameters():
                param.requires_grad = True
            for param in model.features[6].parameters():
                param.requires_grad = True
            for param in model.features[7].parameters():
                param.requires_grad = True
            lr_stage2 = max(lr * 0.25, 1e-5)
            optimizer = build_optimizer(model, lr=lr_stage2, weight_decay=4e-4, backbone_mult=0.08)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=4,
                min_lr=lr_stage2 * 0.02,
            )
            stage2_unfreeze_done = True
            print(f"Fase 2 activada: bloques 5-7 del backbone desbloqueados desde epoch {epoch + 1}.", flush=True)

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            use_mixup = (
                mixup_alpha > 0
                and torch.rand(1).item() < mixup_prob
                and images.size(0) > 1
                and epoch < int(epochs * 0.8)
            )
            if use_mixup:
                images, labels_a, labels_b, lam = mixup_batch(images, labels, mixup_alpha)
            else:
                labels_a, labels_b, lam = labels, labels, 1.0

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            update_ema_model(ema_model, model, decay=ema_decay)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels_a).sum().item()
            total += labels.size(0)

            if step % 20 == 0 or step == steps_per_epoch:
                batch_acc = correct / total if total > 0 else 0.0
                print(f"   Batch {step}/{steps_per_epoch} | loss: {loss.item():.4f} acc_acum: {batch_acc:.4f}", flush=True)

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0

        # Evaluacion en validacion.
        val_loss, val_acc = evaluate(ema_model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_backbone_lr = optimizer.param_groups[0]["lr"]
        current_head_lr = optimizer.param_groups[1]["lr"]

        print(
            f"--- Resumen Epoch {epoch + 1} ---"
            f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
            f"\nVal Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}"
            f"\nLR(backbone/head): {current_backbone_lr:.6f} / {current_head_lr:.6f}",
            flush=True,
        )

        # Guardar ALWAYS el "last"
        torch.save(ema_model.state_dict(), last_model_path)

        # Guardar BEST solo si mejora val_acc (monitor="val_accuracy", mode="max", save_best_only=True)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(ema_model.state_dict(), best_model_path)
            print(f"Nuevo mejor checkpoint guardado (best_val_acc: {best_val_acc:.4f}).", flush=True)
        else:
            epochs_without_improvement += 1
            print(f"Sin mejora en val_acc por {epochs_without_improvement} epoca(s). (best_val_acc: {best_val_acc:.4f})", flush=True)

        # Guardia anti-overfitting: si train_loss baja mientras val_loss sube de forma consistente,
        # contamos una racha y reducimos LR cuando sea necesario.
        if prev_train_loss is not None and prev_val_loss is not None:
            train_improving = (prev_train_loss - train_loss) > overfit_min_delta
            val_worsening = (val_loss - prev_val_loss) > overfit_min_delta
            if train_improving and val_worsening:
                overfit_streak += 1
            else:
                overfit_streak = 0

            if overfit_streak >= max(1, overfit_patience // 2):
                for group in optimizer.param_groups:
                    group["lr"] = max(group["lr"] * 0.8, 1e-6)
                print(
                    f"Aviso: posible sobreajuste ({overfit_streak} epocas). "
                    "Reduciendo LR un 20%.",
                    flush=True,
                )

        prev_train_loss = train_loss
        prev_val_loss = val_loss

        if early_stopping_enabled and epochs_without_improvement >= patience:
            print(f"\nEarly stopping: no mejora de val_acc en {patience} epocas.")
            break

        if overfit_patience > 0 and overfit_streak >= overfit_patience:
            print(
                f"\nOverfitting guard: val_loss empeora frente a train_loss por "
                f"{overfit_streak} epocas seguidas. Deteniendo entrenamiento.",
                flush=True,
            )
            break

    # Guardar clases y test final
    with classes_path.open("w", encoding="utf-8") as file:
        json.dump(classes, file, ensure_ascii=False, indent=2)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\nEntrenamiento finalizado.")
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")


def load_model(model_path: Path, classes_path: Path, device: torch.device):
    with classes_path.open("r", encoding="utf-8") as file:
        classes = json.load(file)

    model = create_model(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, classes


def print_model_summary(data_dir: Path | None, num_classes: int | None):
    if data_dir is not None:
        train_dir = data_dir / "train"
        if not train_dir.exists():
            raise FileNotFoundError("No existe train/ dentro de --data-dir")
        classes = datasets.ImageFolder(train_dir).classes
        classes_count = len(classes)
    elif num_classes is not None:
        classes_count = num_classes
    else:
        raise ValueError("Debes indicar --data-dir o --num-classes para summary")

    model = create_model(classes_count)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=== Model Summary ===")
    print(model)
    print("---------------------")
    print(f"Num classes: {classes_count}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = model(dummy)
    print(f"Output tensor shape: {tuple(out.shape)}")


def predict_image(model_path: Path, classes_path: Path, image_path: Path):
    device = get_device()
    model, classes = load_model(model_path, classes_path, device)

    _, eval_transform = build_transforms()
    image = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    predicted_class = classes[pred.item()]
    print(f"PredicciÃ³n: {predicted_class} (confianza: {confidence.item():.4f})")


def export_model(
    model_path: Path,
    classes_path: Path,
    output_path: Path | None,
    export_format: str = "torchscript",
    opset: int = 17,
):
    device = torch.device("cpu")
    model, classes = load_model(model_path, classes_path, device)
    model.eval()

    if output_path is None:
        default_name = "food_classifier.torchscript.pt" if export_format == "torchscript" else "food_classifier.onnx"
        output_path = model_path.parent / default_name

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    if export_format == "torchscript":
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(str(output_path))
    elif export_format == "onnx":
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
            opset_version=opset,
            do_constant_folding=True,
        )
    else:
        raise ValueError(f"Formato de exportacion no soportado: {export_format}")

    exported_classes_path = output_path.with_name(f"{output_path.stem}_classes.json")
    with exported_classes_path.open("w", encoding="utf-8") as file:
        json.dump(classes, file, ensure_ascii=False, indent=2)

    print(f"Modelo exportado en: {output_path}")
    print(f"Clases exportadas en: {exported_classes_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Modelo de clasificacion de imagenes de alimentos")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Entrena el modelo")
    train_parser.add_argument("--data-dir", required=True, type=Path, help="Ruta al dataset")
    train_parser.add_argument(
        "--output-dir",
        default=Path("./artifacts"),
        type=Path,
        help="Directorio de salida para pesos y clases",
    )
    train_parser.add_argument("--epochs", default=120, type=int)
    train_parser.add_argument("--batch-size", default=32, type=int)
    train_parser.add_argument("--lr", default=1.2e-4, type=float)
    train_parser.add_argument("--patience", default=30, type=int, help="Epocas sin mejora para cortar. Usa 0 para desactivar early stopping")
    train_parser.add_argument("--mixup-alpha", default=0.2, type=float)
    train_parser.add_argument("--mixup-prob", default=0.5, type=float)
    train_parser.add_argument("--unfreeze-epoch", default=0, type=int, help="Epoch para desbloquear mas capas del backbone. Usa 0 para desactivar")
    train_parser.add_argument("--head-only-epochs", default=6, type=int, help="Epochs iniciales entrenando solo el head")
    train_parser.add_argument("--label-smoothing", default=0.12, type=float)
    train_parser.add_argument("--ema-decay", default=0.999, type=float)
    train_parser.add_argument("--overfit-patience", default=5, type=int, help="Epocas consecutivas con patron de sobreajuste antes de cortar")
    train_parser.add_argument("--overfit-min-delta", default=1e-3, type=float, help="Delta minimo para detectar mejora/empeoramiento de losses")

    predict_parser = subparsers.add_parser("predict", help="Predice una imagen")
    predict_parser.add_argument("--model-path", required=True, type=Path)
    predict_parser.add_argument("--classes-path", required=True, type=Path)
    predict_parser.add_argument("--image-path", required=True, type=Path)

    summary_parser = subparsers.add_parser("summary", help="Muestra resumen del modelo")
    summary_parser.add_argument("--data-dir", type=Path, default=None, help="Ruta al dataset para inferir num clases")
    summary_parser.add_argument("--num-classes", type=int, default=None, help="Numero de clases si no usas data-dir")

    export_parser = subparsers.add_parser("export", help="Exporta el modelo entrenado")
    export_parser.add_argument("--model-path", required=True, type=Path)
    export_parser.add_argument("--classes-path", required=True, type=Path)
    export_parser.add_argument("--output-path", type=Path, default=None)
    export_parser.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript")
    export_parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "train":
        train_model(
            dataset_root=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            mixup_alpha=args.mixup_alpha,
            mixup_prob=args.mixup_prob,
            unfreeze_epoch=args.unfreeze_epoch,
            head_only_epochs=args.head_only_epochs,
            label_smoothing=args.label_smoothing,
            ema_decay=args.ema_decay,
            overfit_patience=args.overfit_patience,
            overfit_min_delta=args.overfit_min_delta,
        )
    elif args.command == "predict":
        predict_image(
            model_path=args.model_path,
            classes_path=args.classes_path,
            image_path=args.image_path,
        )
    elif args.command == "summary":
        print_model_summary(data_dir=args.data_dir, num_classes=args.num_classes)
    elif args.command == "export":
        export_model(
            model_path=args.model_path,
            classes_path=args.classes_path,
            output_path=args.output_path,
            export_format=args.format,
            opset=args.opset,
        )


if __name__ == "__main__":
    main()
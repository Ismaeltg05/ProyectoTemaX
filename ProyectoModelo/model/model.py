import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image


IMAGE_SIZE = 224


def get_device() -> torch.device:
	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def build_transforms():
	train_transform = transforms.Compose(
		[
			transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(10),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)
	eval_transform = transforms.Compose(
		[
			transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
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

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return train_loader, val_loader, test_loader, train_dataset.classes


def create_model(num_classes: int) -> nn.Module:
	model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

	for param in model.parameters():
		param.requires_grad = False

	in_features = model.fc.in_features
	model.fc = nn.Sequential(
		nn.Linear(in_features, 256),
		nn.ReLU(),
		nn.Dropout(0.3),
		nn.Linear(256, num_classes),
	)

	return model


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0

	with torch.no_grad():
		for images, labels in dataloader:
			images = images.to(device)
			labels = labels.to(device)

			outputs = model(images)
			loss = criterion(outputs, labels)

			running_loss += loss.item() * images.size(0)
			_, preds = torch.max(outputs, 1)
			correct += (preds == labels).sum().item()
			total += labels.size(0)

	epoch_loss = running_loss / total if total > 0 else 0.0
	epoch_acc = correct / total if total > 0 else 0.0
	return epoch_loss, epoch_acc


def train_model(dataset_root: Path, output_dir: Path, epochs: int = 10, batch_size: int = 32, lr: float = 1e-3):
	output_dir.mkdir(parents=True, exist_ok=True)

	train_loader, val_loader, test_loader, classes = create_dataloaders(dataset_root, batch_size)

	device = get_device()
	model = create_model(num_classes=len(classes)).to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

	best_val_acc = 0.0
	best_model_path = output_dir / "food_classifier.pth"
	classes_path = output_dir / "classes.json"

	for epoch in range(epochs):
		model.train()
		running_loss = 0.0
		correct = 0
		total = 0

		for images, labels in train_loader:
			images = images.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item() * images.size(0)
			_, preds = torch.max(outputs, 1)
			correct += (preds == labels).sum().item()
			total += labels.size(0)

		train_loss = running_loss / total if total > 0 else 0.0
		train_acc = correct / total if total > 0 else 0.0

		val_loss, val_acc = evaluate(model, val_loader, criterion, device)

		print(
			f"Epoch {epoch + 1}/{epochs} | "
			f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
			f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}"
		)

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			torch.save(model.state_dict(), best_model_path)

	with classes_path.open("w", encoding="utf-8") as file:
		json.dump(classes, file, ensure_ascii=False, indent=2)

	model.load_state_dict(torch.load(best_model_path, map_location=device))
	test_loss, test_acc = evaluate(model, test_loader, criterion, device)

	print(f"Modelo guardado en: {best_model_path}")
	print(f"Clases guardadas en: {classes_path}")
	print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")


def load_model(model_path: Path, classes_path: Path, device: torch.device):
	with classes_path.open("r", encoding="utf-8") as file:
		classes = json.load(file)

	model = create_model(num_classes=len(classes)).to(device)
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()
	return model, classes


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
	print(f"Predicción: {predicted_class} (confianza: {confidence.item():.4f})")


def parse_args():
	parser = argparse.ArgumentParser(description="Modelo de clasificación de imágenes de alimentos")
	subparsers = parser.add_subparsers(dest="command", required=True)

	train_parser = subparsers.add_parser("train", help="Entrena el modelo")
	train_parser.add_argument("--data-dir", required=True, type=Path, help="Ruta al dataset")
	train_parser.add_argument(
		"--output-dir",
		default=Path("./artifacts"),
		type=Path,
		help="Directorio de salida para pesos y clases",
	)
	train_parser.add_argument("--epochs", default=10, type=int)
	train_parser.add_argument("--batch-size", default=32, type=int)
	train_parser.add_argument("--lr", default=1e-3, type=float)

	predict_parser = subparsers.add_parser("predict", help="Predice una imagen")
	predict_parser.add_argument("--model-path", required=True, type=Path)
	predict_parser.add_argument("--classes-path", required=True, type=Path)
	predict_parser.add_argument("--image-path", required=True, type=Path)

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
		)
	elif args.command == "predict":
		predict_image(
			model_path=args.model_path,
			classes_path=args.classes_path,
			image_path=args.image_path,
		)


if __name__ == "__main__":
	main()

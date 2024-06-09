import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import copy

# 커스텀 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, image_folder, label, transform=None):
        self.image_folder = image_folder
        self.label = label
        self.transform = transform
        self.image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.label

# 데이터 전처리
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def load_datasets(data_dir):
    train_datasets = []
    class_names = []
    for idx, class_folder in enumerate(sorted(os.listdir(data_dir))):
        folder_path = os.path.join(data_dir, class_folder)
        if os.path.isdir(folder_path):
            train_datasets.append(CustomDataset(folder_path, idx, data_transforms['train']))
            class_names.append(class_folder)
    return train_datasets, class_names

def train_model(model, criterion, optimizer, train_loader, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()  # 학습 모드

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Best Train Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    data_dir = 'alpha_class'
    train_datasets, class_names = load_datasets(data_dir)

    # 데이터 로더 구성
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 사전 학습된 모델 로드 및 수정
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))  # 출력 뉴런 수를 클래스 수로 변경

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 모델 학습
    num_epochs = 25
    model = train_model(model, criterion, optimizer, train_loader, num_epochs=num_epochs)

    # 학습된 모델 저장
    torch.save(model.state_dict(), 'best_model.pth')

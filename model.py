import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

class new_model(nn.Module):
    def __init__(self):
        super(new_model, self).__init__()
        self.flatten = nn.Flatten()
        # 모델을 16, 32, 64, 128, 256, 128, 64, 32, 16, 10으로 다이아몬드 형태로 증가했다 감소하는 모델
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits

def train_and_export(
    onnx_filename="new_fashion_mnist_diamond_relu.onnx", # 파일을 저장하는 경로
    epochs=10,
    data_root="./data",
    batch_size=64,
    learning_rate=0.01,
    weight_decay=0.0,
    seed=42,
):
    torch.manual_seed(seed)
    transform = transforms.Compose([transforms.ToTensor()])

    # 데이터셋 로드
    full_train_dataset = torchvision.datasets.FashionMNIST(
        root=data_root, train=True, download=True, transform=transform
    )

    # train data는 80%, validation data는 20%로 나눈다.
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], generator=split_generator
    )

    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = new_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    # sgd 방식에 momentum을 추가함으로써 수렴을 용이하게 했다.

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 학습 과정 출력
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            epoch_loss += loss.item() * labels.size(0)
            epoch_total += labels.size(0)
            epoch_correct += (predicted == labels).sum().item()

            #300번째 batch마다 학습 loss 계산
            if i % 300 == 299:
                train_acc = 100 * correct / total
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Train Loss: {running_loss / 300:.3f} | Train Acc: {train_acc:.2f}%")
                running_loss = 0.0
                correct = 0
                total = 0

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # test 과정
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_loss = epoch_loss / epoch_total
        train_acc = 100 * epoch_correct / epoch_total
        val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"==> [Epoch {epoch + 1}] "
            f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%\n"
        )
    print("-----------------------------------------")
    print("학습 종료. Test 데이터셋으로 최종 성능을 평가합니다...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = 100 * test_correct / test_total
    print(f"[최종 Test 결과] Loss: {test_loss / test_total:.3f} | Acc: {test_acc:.2f}%")
    print("-----------------------------------------\n")
    model.eval()

    dummy_input = torch.randn(1, 1, 28, 28) # 입력 모양을 알려주기 위한 예시 입력
    
    # model을 onnx 파일로 저장
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_filename,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    print(f"ONNX Export Complete: {onnx_filename}")
    return onnx_filename


def main():
    train_and_export()

if __name__ == "__main__":
    main()

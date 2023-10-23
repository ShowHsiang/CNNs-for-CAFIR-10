import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.deploy:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            if stride == 1:
                self.shortcut = nn.Identity()
            else:
                self.shortcut = None
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            return self.relu(self.conv(x))
        else:
            if self.stride == 1:
                identity = self.shortcut(x)
            else:
                identity = 0
            return self.relu(self.bn1(self.conv1(x)) + self.bn2(self.conv2(x)) + identity)


    def _fuse_bn(self, conv, bn):
        w_conv = conv.weight
        b_conv = conv.bias
    
        if b_conv is None:
            b_conv = torch.zeros(w_conv.size(0), device=w_conv.device)
        
        mean_bn, var_bn, w_bn, b_bn = bn.running_mean, bn.running_var, bn.weight, bn.bias
        w_fused = w_conv * (w_bn.reshape(-1, 1, 1, 1) / torch.sqrt(var_bn.reshape(-1, 1, 1, 1) + 1e-5))
        b_fused = (b_conv - mean_bn) * (w_bn / torch.sqrt(var_bn + 1e-5)) + b_bn
    
    # Return a new Conv2D layer
        return nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True
        ).to(conv.weight.device).requires_grad_().to(conv.weight.device)

    def switch_to_deploy(self):
        fused_conv1 = self._fuse_bn(self.conv1, self.bn1)
        fused_conv2 = self._fuse_bn(self.conv2, self.bn2)
    
        fused_conv1.weight.data = fused_conv1.weight.data + fused_conv2.weight.data
        fused_conv1.bias.data = fused_conv1.bias.data + fused_conv2.bias.data
    
        if self.stride == 1 and self.shortcut is not None and isinstance(self.shortcut, nn.Conv2d):
            fused_conv1.weight.data += self.shortcut.weight.data
        self.conv = fused_conv1
    
    # Remove other layers for deployment
        for name, module in self.named_children():
            if name not in ['conv', 'relu']:
                self.add_module(name, None)
        self.deploy = True


    
class RepVGG(nn.Module):
    def __init__(self, num_classes=10):
        super(RepVGG, self).__init__()

        self.stage_0 = RepVGGBlock(3, 64, kernel_size=3, stride=2, padding=1)
        self.stage_1 = nn.Sequential(
            RepVGGBlock(64, 64, kernel_size=3, stride=1, padding=1),
            RepVGGBlock(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.stage_2 = nn.Sequential(
            RepVGGBlock(64, 128, kernel_size=3, stride=2, padding=1),
            RepVGGBlock(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.stage_3 = nn.Sequential(
            RepVGGBlock(128, 256, kernel_size=3, stride=2, padding=1),
            RepVGGBlock(256, 256, kernel_size=3, stride=1, padding=1),
            RepVGGBlock(256, 256, kernel_size=3, stride=1, padding=1)
        )
        self.stage_4 = nn.Sequential(
            RepVGGBlock(256, 512, kernel_size=3, stride=2, padding=1),
            RepVGGBlock(512, 512, kernel_size=3, stride=1, padding=1),
            RepVGGBlock(512, 512, kernel_size=3, stride=1, padding=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stage_0(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

def train(net, trainloader, criterion, optimizer, device):
    net.train()
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(trainloader)


def test(net, testloader, criterion, device):
    net.eval()
    total = 0
    correct = 0
    test_loss = 0.0
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    return test_loss / len(testloader), accuracy, predicted_labels, true_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = RepVGG(num_classes=10)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

num_epochs = 60

train_losses = []
test_losses = []
accuracies = []

for epoch in range(num_epochs):
    train_loss = train(net, trainloader, criterion, optimizer, device)
    test_loss, accuracy, _, _ = test(net, testloader, criterion, device)  # We're going to get a fresh set of predicted/true labels after the switch to deploy mode.

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    accuracies.append(accuracy)

    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Accuracy: {accuracy:.2f}%")

# After training, convert to deploy mode
for module in net.modules():
    if isinstance(module, RepVGGBlock):
        module.switch_to_deploy()

# Test in Deploy Mode
test_loss, accuracy, predicted_labels, true_labels = test(net, testloader, criterion, device)

plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(range(1, num_epochs + 1), accuracies, label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


cm = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

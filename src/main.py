import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# ----- load config -----
cfg = yaml.safe_load(open("src/configs/config.yaml"))

# ----- toy dataset -----
x = torch.randn(500, 10)
y = torch.randint(0, 2, (500,))

# ----- simple model -----
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Using device:", device)

# ----- training loop -----
for epoch in range(cfg["train"]["epochs"]):
    optimizer.zero_grad()
    out = model(x.to(device))
    loss = loss_fn(out, y.to(device))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{cfg['train']['epochs']} - Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "simple_model.pth")
print("✅ Training complete — model saved to simple_model.pth")

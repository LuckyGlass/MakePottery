import torch

def blend(voxel_fake, voxel_real):
    b = voxel_fake.shape[0]
    alpha = torch.rand(b,1,1,1)
    return alpha * voxel_fake + (1 - alpha) * voxel_real

def gradient_penalty(y_pred, voxel_blend):
    gradients = torch.autograd.grad(outputs=y_pred, inputs=voxel_blend, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
    print(gradients.shape)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2) * 10
    return penalty
    
v1 = torch.rand(3,2,2,2)
v2 = torch.rand(3,2,2,2)
v3 = blend(v1, v2).requires_grad_()
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(8, 1)
)
y = model(v3)
loss = gradient_penalty(y, v3)

print(loss)
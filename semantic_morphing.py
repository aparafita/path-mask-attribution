import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

def semantic_morphing(
    model, vae, device, original_emb, new_label,
    optimizer=optim.Adam, lr=1e-3, 
    entropy_threshold=.01, max_epochs=1000, alpha_lk=1.
):
    original_emb = original_emb.clone()
    emb = nn.Parameter(original_emb.unsqueeze(0).to(device))
    obj_norm = vae.norm_means[int(new_label)]
    
    optimizer = optimizer([emb], lr=lr)
    
    pred_loss_avg = 0
    lk_loss_avg = 0
    
    losses = []
    
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        model.zero_grad()
        vae.zero_grad()

        decoded = vae.decode(emb)
        pred = model(decoded)

        ps = F.softmax(pred, dim=1)[0]
        entropy = -(ps * torch.log2(ps)).sum()

        if entropy.item() < entropy_threshold and pred[0].argmax() == new_label:
            break
        
        pred_loss = -pred[0, new_label] # maximize
        lk_loss = (emb.norm(p=2) - obj_norm) ** 2
        
        # Compute loss means
        pred_loss_avg = (pred_loss_avg * (epoch - 1) + pred_loss.abs().detach()) / epoch
        lk_loss_avg = (lk_loss_avg * (epoch - 1) + lk_loss.abs().detach()) / epoch
        
        # Define loss and optimize
        loss = (
            pred_loss + 
            alpha_lk * lk_loss * pred_loss_avg / lk_loss_avg
        )
        
        loss.backward(retain_graph=True)
        optimizer.step()
        
        losses.append((pred_loss, lk_loss, loss))
        
    emb = emb[0].detach()
    decoded = vae.decode(emb.unsqueeze(0))[0].detach()
    
    if losses:
        losses = [
            torch.Tensor([row[i] for row in losses])
            for i in range(len(losses[0]))
        ]
    
    return emb, decoded, losses
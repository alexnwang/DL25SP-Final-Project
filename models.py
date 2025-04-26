from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
from loss import VICRegLoss


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", output_dim=256):
        super().__init__()
        self.device = device
        self.repr_dim = output_dim

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        B, T, _ = actions.shape

        return torch.randn((B, T + 1, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output






class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.same_channels = in_channels == out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if not self.same_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class ResEncoder2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ResBlock(1, 16, stride=2),   # (1, 65, 65) --> (16, 33, 33)
            ResBlock(16, 32, stride=2),  # (32, 17, 17)
            ResBlock(32, 64, stride=2),  # (64, 9, 9)
            ResBlock(64, 128, stride=2), # (128, 5, 5)
        )

    def forward(self, x):  # x: (B, 2, 65, 65)
        return self.layers(x)  # (B, 128, 5, 5)



class SplitEncoder2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.wall_encoder = ResEncoder2D()
        self.agent_encoder = ResEncoder2D()

    def forward(self, x):  # x: (B, 2, 65, 65)
        wall = x[:, 0:1, :, :]  # (B, 1, 65, 65)
        agent = x[:, 1:2, :, :]  # (B, 1, 65, 65)

        wall_repr = self.wall_encoder(wall)    # (B, 128, 5, 5)
        agent_repr = self.agent_encoder(agent) # (B, 128, 5, 5)

        fused = wall_repr + agent_repr  # Element-wise sum (alternatively: torch.cat and project)
        return fused  # Output: # (B, 128, 5, 5)



class ResPredictor2D(nn.Module):
    def __init__(self, action_dim=2):
        super().__init__()
        self.action_fc = nn.Linear(action_dim, 5 * 5)  # 1 × 5 × 5 channel

        self.layers = nn.Sequential(
            ResBlock(129, 129, stride=1),
            ResBlock(129, 129, stride=1),
            ResBlock(129, 128, stride=1),  # Output: (B, 128, 5, 5)
        )

    def forward(self, state_embed, action):
        # state_embed: (B, 128, 5, 5)
        # action: (B, 2)
        B = action.shape[0]
        action_embed = self.action_fc(action).view(B, 1, 5, 5)
        x = torch.cat([state_embed, action_embed], dim=1)  # (B, 129, 5, 5)
        return self.layers(x)  # (B, 128, 5, 5)


class ResPredictor2Dv2(nn.Module):
    def __init__(self, action_dim=2):
        super().__init__()
        self.action_fc = nn.Linear(action_dim, 5 * 5)  # 1 × 5 × 5 channel

        self.layers = nn.Sequential(
            ResBlock(129, 64, stride=1),
            ResBlock(64, 128, stride=1),
            ResBlock(128, 128, stride=1),  # Output: (B, 128, 5, 5)
        )

    def forward(self, state_embed, action):
        # state_embed: (B, 128, 5, 5)
        # action: (B, 2)
        B = action.shape[0]
        action_embed = self.action_fc(action).view(B, 1, 5, 5)
        x = torch.cat([state_embed, action_embed], dim=1)  # (B, 129, 5, 5)
        return self.layers(x)  # (B, 128, 5, 5)



from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR





class JEPA_SplitEncoder_VICReg(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.wall_encoder = ResEncoder2D()
        self.agent_encoder = ResEncoder2D()
        self.predictor = ResPredictor2D()

        self.repr_dim = 128
        self.config = config
        self.name = "JEPA_SplitEncoder_VICReg"

        # Set up optimizer
        self.optimizer = AdamW(self.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        # Set up scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs * config.steps_per_epoch)

        self.pool = lambda x: x.mean(dim=[2, 3])  # Average over H and W

        self.vicreg_loss = VICRegLoss()

        
    def forward(self, states, actions, teacher_forcing=False, return_enc=False, pred_flattened=True):
        """
        states: (B, T+1, 2, 65, 65)
        actions: (B, T, action_dim)
        """

        states, actions = states.to("cuda"), actions.to("cuda")

        B, _, C, H, W = states.shape
        _, T, _ = actions.shape
        T_plus_1 = T + 1

        if teacher_forcing:
            # --- encode all states ---
            encs = []
            for t in range(T_plus_1):
                wall_obs = states[:, t, 0:1]  # (B, 1, 65, 65)
                agent_obs = states[:, t, 1:2]

                wall_embed = self.wall_encoder(wall_obs)
                agent_embed = self.agent_encoder(agent_obs)

                encs.append(agent_embed + wall_embed)
            encs = torch.stack(encs, dim=1)  # (B, T+1, C_out, 5, 5)

            preds = torch.zeros_like(encs)
            preds[:, 0] = encs[:, 0]  # seed

            inputs = encs[:, :-1].reshape(B * T, -1, 5, 5)
            a = actions.reshape(B * T, -1)
            preds_1_to_T = self.predictor(inputs, a).reshape(B, T, -1, 5, 5)
            preds[:, 1:] = preds_1_to_T

            if pred_flattened:
                preds = preds.view(B, T_plus_1, -1)
                if return_enc:
                    encs = encs.view(B, T_plus_1, -1)

            return (preds, encs) if return_enc else preds

        else:
            
            # --- rollout mode (for probing, no peeking ahead) ---
            wall_obs = states[:, 0, 0:1]  # (B, 1, 65, 65)
            agent_obs = states[:, 0, 1:2]

            wall_embed = self.wall_encoder(wall_obs)
            agent_embed = self.agent_encoder(agent_obs)
            s_embed = agent_embed + wall_embed

            preds = [self.pool(s_embed)]  # Start from t=0

            for t in range(T):
                a = actions[:, t]  # (B, action_dim)
                s_embed = self.predictor(s_embed, a)  # Predict next latent
                preds.append(self.pool(s_embed))  # Pool and store

            preds = torch.stack(preds, dim=1)  # (B, T+1, repr_dim)

            return preds


    def training_step(self, batch, device):
        self.train()
        states, actions = batch.states.to(device), batch.actions.to(device)
        preds, targets = self.forward(states, actions, teacher_forcing=True, return_enc=True, pred_flattened=False)
        
        # loss = F.mse_loss(preds[:, 1:], targets[:, 1:])  # only from t=1 onwards

        # ignore time=0 (only predict future)
        _, _, C, H, W = preds.shape
        D = C*H*W
        preds_flat = preds[:, 1:].reshape(-1, D)      # (B * T, D)
        targets_flat = targets[:, 1:].reshape(-1, D)  # (B * T, D)
        metrics = self.vicreg_loss(preds_flat, targets_flat)
        loss = metrics["loss"]


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def validation_step(self, batch):
        self.eval()
        states, actions = batch.states.cuda(), batch.actions.cuda()
        preds, targets = self.forward(states, actions, teacher_forcing=True, return_enc=True, pred_flattened=False)
        loss = F.mse_loss(preds[:, 1:], targets[:, 1:])
        return loss.item()









class JEPA_SplitEncoder_VICRegV2(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.wall_encoder = ResEncoder2D()
        self.agent_encoder = ResEncoder2D()
        self.predictor = ResPredictor2Dv2()

        self.repr_dim = 128
        self.config = config
        self.name = "JEPA_SplitEncoder_VICRegv2"

        # Set up optimizer
        self.optimizer = AdamW(self.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        # Set up scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs * config.steps_per_epoch)

        self.pool = lambda x: x.mean(dim=[2, 3])  # Average over H and W

        self.vicreg_loss = VICRegLoss()

        
    def forward(self, states, actions, teacher_forcing=False, return_enc=False, pred_flattened=True):
        """
        states: (B, T+1, 2, 65, 65)
        actions: (B, T, action_dim)
        """

        states, actions = states.to("cuda"), actions.to("cuda")

        B, _, C, H, W = states.shape
        _, T, _ = actions.shape
        T_plus_1 = T + 1

        if teacher_forcing:
            # --- encode all states ---
            encs = []
            for t in range(T_plus_1):
                wall_obs = states[:, t, 0:1]  # (B, 1, 65, 65)
                agent_obs = states[:, t, 1:2]

                wall_embed = self.wall_encoder(wall_obs)
                agent_embed = self.agent_encoder(agent_obs)

                encs.append(agent_embed + wall_embed)
            encs = torch.stack(encs, dim=1)  # (B, T+1, C_out, 5, 5)

            preds = torch.zeros_like(encs)
            preds[:, 0] = encs[:, 0]  # seed

            inputs = encs[:, :-1].reshape(B * T, -1, 5, 5)
            a = actions.reshape(B * T, -1)
            preds_1_to_T = self.predictor(inputs, a).reshape(B, T, -1, 5, 5)
            preds[:, 1:] = preds_1_to_T

            if pred_flattened:
                preds = preds.view(B, T_plus_1, -1)
                if return_enc:
                    encs = encs.view(B, T_plus_1, -1)

            return (preds, encs) if return_enc else preds

        else:
            
            # --- rollout mode (for probing, no peeking ahead) ---
            wall_obs = states[:, 0, 0:1]  # (B, 1, 65, 65)
            agent_obs = states[:, 0, 1:2]

            wall_embed = self.wall_encoder(wall_obs)
            agent_embed = self.agent_encoder(agent_obs)
            s_embed = agent_embed + wall_embed

            preds = [self.pool(s_embed)]  # Start from t=0

            for t in range(T):
                a = actions[:, t]  # (B, action_dim)
                s_embed = self.predictor(s_embed, a)  # Predict next latent
                preds.append(self.pool(s_embed))  # Pool and store

            preds = torch.stack(preds, dim=1)  # (B, T+1, repr_dim)

            return preds


    def training_step(self, batch, device):
        self.train()
        states, actions = batch.states.to(device), batch.actions.to(device)
        preds, targets = self.forward(states, actions, teacher_forcing=True, return_enc=True, pred_flattened=False)
        
        # loss = F.mse_loss(preds[:, 1:], targets[:, 1:])  # only from t=1 onwards

        # ignore time=0 (only predict future)
        _, _, C, H, W = preds.shape
        D = C*H*W
        preds_flat = preds[:, 1:].reshape(-1, D)      # (B * T, D)
        targets_flat = targets[:, 1:].reshape(-1, D)  # (B * T, D)
        metrics = self.vicreg_loss(preds_flat, targets_flat)
        loss = metrics["loss"]


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def validation_step(self, batch):
        self.eval()
        states, actions = batch.states.cuda(), batch.actions.cuda()
        preds, targets = self.forward(states, actions, teacher_forcing=True, return_enc=True, pred_flattened=False)
        loss = F.mse_loss(preds[:, 1:], targets[:, 1:])
        return loss.item()







class JEPA_SplitEncoder_CombinedLoss(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.wall_encoder = ResEncoder2D()
        self.agent_encoder = ResEncoder2D()
        self.predictor = ResPredictor2D()

        self.repr_dim = 128
        self.config = config
        self.name = "JEPA_SplitEncoder_CombinedLoss"

        # Set up optimizer
        self.optimizer = AdamW(self.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        # Set up scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs * config.steps_per_epoch)

        self.pool = lambda x: x.mean(dim=[2, 3])  # Average over H and W

        self.vicreg_loss = VICRegLoss()

        
    def forward(self, states, actions, teacher_forcing=False, return_enc=False, pred_flattened=True):
        """
        states: (B, T+1, 2, 65, 65)
        actions: (B, T, action_dim)
        """

        states, actions = states.to("cuda"), actions.to("cuda")

        B, _, C, H, W = states.shape
        _, T, _ = actions.shape
        T_plus_1 = T + 1

        if teacher_forcing:
            # --- encode all states ---
            encs = []
            for t in range(T_plus_1):
                wall_obs = states[:, t, 0:1]  # (B, 1, 65, 65)
                agent_obs = states[:, t, 1:2]

                wall_embed = self.wall_encoder(wall_obs)
                agent_embed = self.agent_encoder(agent_obs)

                encs.append(agent_embed + wall_embed)
            encs = torch.stack(encs, dim=1)  # (B, T+1, C_out, 5, 5)

            preds = torch.zeros_like(encs)
            preds[:, 0] = encs[:, 0]  # seed

            inputs = encs[:, :-1].reshape(B * T, -1, 5, 5)
            a = actions.reshape(B * T, -1)
            preds_1_to_T = self.predictor(inputs, a).reshape(B, T, -1, 5, 5)
            preds[:, 1:] = preds_1_to_T

            if pred_flattened:
                preds = preds.view(B, T_plus_1, -1)
                if return_enc:
                    encs = encs.view(B, T_plus_1, -1)

            return (preds, encs) if return_enc else preds

        else:
            
            # --- rollout mode (for probing, no peeking ahead) ---
            wall_obs = states[:, 0, 0:1]  # (B, 1, 65, 65)
            agent_obs = states[:, 0, 1:2]

            wall_embed = self.wall_encoder(wall_obs)
            agent_embed = self.agent_encoder(agent_obs)
            s_embed = agent_embed + wall_embed

            preds = [self.pool(s_embed)]  # Start from t=0

            for t in range(T):
                a = actions[:, t]  # (B, action_dim)
                s_embed = self.predictor(s_embed, a)  # Predict next latent
                preds.append(self.pool(s_embed))  # Pool and store

            preds = torch.stack(preds, dim=1)  # (B, T+1, repr_dim)

            return preds


    def training_step(self, batch, device):
        self.train()
        states, actions = batch.states.to(device), batch.actions.to(device)
        preds, targets = self.forward(states, actions, teacher_forcing=True, return_enc=True, pred_flattened=False)
        
        # loss = F.mse_loss(preds[:, 1:], targets[:, 1:])  # only from t=1 onwards

        # ignore time=0 (only predict future)
        _, _, C, H, W = preds.shape
        D = C*H*W
        preds_flat = preds[:, 1:].reshape(-1, D)      # (B * T, D)
        targets_flat = targets[:, 1:].reshape(-1, D)  # (B * T, D)
        metrics = self.vicreg_loss(preds_flat, targets_flat)

        mse_loss = F.mse_loss(preds_flat, targets_flat)
        
        loss = metrics["loss"] + mse_loss


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def validation_step(self, batch):
        self.eval()
        states, actions = batch.states.cuda(), batch.actions.cuda()
        preds, targets = self.forward(states, actions, teacher_forcing=True, return_enc=True, pred_flattened=False)
        loss = F.mse_loss(preds[:, 1:], targets[:, 1:])
        return loss.item()











class JEPA_SplitEncoder_CombinedLossNoPool(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.wall_encoder = ResEncoder2D()
        self.agent_encoder = ResEncoder2D()
        self.predictor = ResPredictor2D()

        self.repr_dim = 128 * 5 * 5
        self.config = config
        self.name = "JEPA_SplitEncoder_CombinedLossNoPool"

        # Set up optimizer
        self.optimizer = AdamW(self.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        # Set up scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs * config.steps_per_epoch)

        self.pool = lambda x: x.mean(dim=[2, 3])  # Average over H and W (NOT USED in this model, but too afraid to delete it lol)

        self.vicreg_loss = VICRegLoss()

        
    def forward(self, states, actions, teacher_forcing=False, return_enc=False, pred_flattened=True):
        """
        states: (B, T+1, 2, 65, 65)
        actions: (B, T, action_dim)
        """

        states, actions = states.to("cuda"), actions.to("cuda")

        B, _, C, H, W = states.shape
        _, T, _ = actions.shape
        T_plus_1 = T + 1

        if teacher_forcing:
            # --- encode all states ---
            encs = []
            for t in range(T_plus_1):
                wall_obs = states[:, t, 0:1]  # (B, 1, 65, 65)
                agent_obs = states[:, t, 1:2]

                wall_embed = self.wall_encoder(wall_obs)
                agent_embed = self.agent_encoder(agent_obs)

                encs.append(agent_embed + wall_embed)
            encs = torch.stack(encs, dim=1)  # (B, T+1, C_out, 5, 5)

            preds = torch.zeros_like(encs)
            preds[:, 0] = encs[:, 0] 

            inputs = encs[:, :-1].reshape(B * T, -1, 5, 5)
            a = actions.reshape(B * T, -1)
            preds_1_to_T = self.predictor(inputs, a).reshape(B, T, -1, 5, 5)
            preds[:, 1:] = preds_1_to_T

            if pred_flattened:
                preds = preds.view(B, T_plus_1, -1)
                if return_enc:
                    encs = encs.view(B, T_plus_1, -1)

            return (preds, encs) if return_enc else preds

        else:
            # --- rollout mode (for probing, no peeking ahead) ---
            wall_obs = states[:, 0, 0:1]  # (B, 1, 65, 65)
            agent_obs = states[:, 0, 1:2]
            
            wall_embed = self.wall_encoder(wall_obs)  # (B, C, 5, 5)
            agent_embed = self.agent_encoder(agent_obs)  # (B, C, 5, 5)
            s_embed = agent_embed + wall_embed
            
            preds = [s_embed]  # Store full spatial feature, not pooled!
            
            for t in range(T):
                a = actions[:, t]  # (B, action_dim)
                s_embed = self.predictor(s_embed, a)  # (B, C, 5, 5)
                preds.append(s_embed)  # Save spatial feature at each step
            
            # Stack along time dimension
            preds = torch.stack(preds, dim=1)  # (B, T+1, C, 5, 5)
            
            # Now, flatten spatial dims to get (B, T+1, D)
            B, T_plus_1, C, H, W = preds.shape
            D = C * H * W
            preds = preds.view(B, T_plus_1, D)  # (B, T+1, D)

            return preds


    def training_step(self, batch, device):
        self.train()
        states, actions = batch.states.to(device), batch.actions.to(device)
        preds, targets = self.forward(states, actions, teacher_forcing=True, return_enc=True, pred_flattened=False)
        
        # loss = F.mse_loss(preds[:, 1:], targets[:, 1:])  # only from t=1 onwards

        # ignore time=0 (only predict future)
        _, _, C, H, W = preds.shape
        D = C*H*W
        preds_flat = preds[:, 1:].reshape(-1, D)      # (B * T, D)
        targets_flat = targets[:, 1:].reshape(-1, D)  # (B * T, D)
        metrics = self.vicreg_loss(preds_flat, targets_flat)

        mse_loss = F.mse_loss(preds_flat, targets_flat)
        
        loss = metrics["loss"] + mse_loss


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def validation_step(self, batch):
        self.eval()
        states, actions = batch.states.cuda(), batch.actions.cuda()
        preds, targets = self.forward(states, actions, teacher_forcing=True, return_enc=True, pred_flattened=False)
        loss = F.mse_loss(preds[:, 1:], targets[:, 1:])
        return loss.item()







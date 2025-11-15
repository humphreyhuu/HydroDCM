import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalLayer(Function):
    """Gradient Reversal Layer for adversarial training"""
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reversal(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)


class EDLSTMEncoder(nn.Module):
    """Encoder-Decoder LSTM as feature extractor f_phi"""
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, dropout=0.1):
        super(EDLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, T_in, F_in] input sequences
        Returns:
            h: [B, H_feat] hidden representation
        """
        # Encoder: process input sequence
        encoder_out, (h_n, c_n) = self.encoder(x)  # encoder_out: [B, T, H], h_n: [L, B, H]
        
        # Use last hidden state as representation
        h_last = h_n[-1]  # [B, H]
        
        # Decoder: transform to final hidden representation
        h = self.decoder(h_last)  # [B, H_feat]
        
        return h


class DomainEncoder(nn.Module):
    """Linear projection for pseudo-domain embedding"""
    def __init__(self, spatial_dim=3, domain_embed_dim=32):
        super(DomainEncoder, self).__init__()
        self.projection = nn.Linear(spatial_dim, domain_embed_dim)
        
    def forward(self, s):
        """
        Args:
            s: [B, 3] spatial attributes (lat, lon, elev)
        Returns:
            v: [B, d] pseudo-domain embedding
        """
        v = self.projection(s)
        return v


class DomainDiscriminator(nn.Module):
    """Domain discriminator d_theta with gradient reversal"""
    def __init__(self, feature_dim=64, hidden_dim=32, num_domains=30):
        super(DomainDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_domains)
        )
        
    def forward(self, h, alpha=1.0):
        """
        Args:
            h: [B, H_feat] hidden features
            alpha: gradient reversal strength
        Returns:
            domain_logits: [B, num_domains] domain classification logits
        """
        # Apply gradient reversal
        h_reversed = gradient_reversal(h, alpha)
        domain_logits = self.discriminator(h_reversed)
        return domain_logits


class FiLMAdapter(nn.Module):
    """FiLM adapter m_delta for feature modulation"""
    def __init__(self, spatial_dim=3, feature_dim=64, hidden_dim=32):
        super(FiLMAdapter, self).__init__()
        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Scaling head
        self.gamma_head = nn.Linear(hidden_dim, feature_dim)
        
        # Shifting head  
        self.delta_head = nn.Linear(hidden_dim, feature_dim)
        
    def forward(self, s):
        """
        Args:
            s: [B, 3] spatial attributes
        Returns:
            gamma: [B, H_feat] scaling coefficients
            delta: [B, H_feat] shifting coefficients
        """
        shared_repr = self.shared(s)
        gamma = torch.sigmoid(self.gamma_head(shared_repr))  # Ensure positive scaling
        delta = self.delta_head(shared_repr)
        return gamma, delta


class PredictiveHead(nn.Module):
    """Predictive head p_omega"""
    def __init__(self, feature_dim=64, output_dim=7, hidden_dim=32):
        super(PredictiveHead, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z_tilde):
        """
        Args:
            z_tilde: [B, H_feat] modulated features
        Returns:
            y_hat: [B, H] predictions (H=7 for 7-day prediction)
        """
        y_hat = self.predictor(z_tilde)
        return y_hat


class HydroDCM(nn.Module):
    """HydroDCM: Domain Generalization model for reservoir inflow prediction"""
    def __init__(self, 
                 input_dim=3,           # Temperature, precipitation, inflow
                 spatial_dim=3,         # Lat, lon, elevation
                 hidden_dim=64,         # Feature dimension
                 domain_embed_dim=32,   # Domain embedding dimension
                 output_dim=7,          # 7-day prediction
                 num_domains=30,        # Total number of reservoirs
                 num_layers=2,          # LSTM layers
                 dropout=0.1):
        super(HydroDCM, self).__init__()
        
        # Feature extractor f_phi (ED-LSTM)
        self.f_phi = EDLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Domain encoder (linear projection)
        self.domain_encoder = DomainEncoder(
            spatial_dim=spatial_dim,
            domain_embed_dim=domain_embed_dim
        )
        
        # Domain discriminator d_theta
        self.d_theta = DomainDiscriminator(
            feature_dim=hidden_dim,
            hidden_dim=32,
            num_domains=num_domains
        )
        
        # FiLM adapter m_delta
        self.m_delta = FiLMAdapter(
            spatial_dim=spatial_dim,
            feature_dim=hidden_dim,
            hidden_dim=32
        )
        
        # Predictive head p_omega
        self.p_omega = PredictiveHead(
            feature_dim=hidden_dim,
            output_dim=output_dim,
            hidden_dim=32
        )
        
    def forward(self, X, s, alpha=1.0, return_components=False):
        """
        Args:
            X: [B, T_in, F_in] input sequences
            s: [B, 3] spatial attributes
            alpha: gradient reversal strength
            return_components: whether to return intermediate components
        Returns:
            If return_components=False:
                y_hat: [B, H] predictions
            If return_components=True:
                dict with all components
        """
        # 1. Temporal feature extraction
        h = self.f_phi(X)  # [B, H_feat]
        
        # 2. Pseudo-domain embedding
        v = self.domain_encoder(s)  # [B, d]
        
        # 3. Domain discrimination (for adversarial loss)
        domain_logits = self.d_theta(h, alpha)  # [B, num_domains]
        
        # 4. FiLM modulation
        gamma, delta = self.m_delta(s)  # [B, H_feat], [B, H_feat]
        z_tilde = gamma * h + delta  # Element-wise modulation
        
        # 5. Prediction
        y_hat = self.p_omega(z_tilde)  # [B, H]
        
        if return_components:
            return {
                'y_hat': y_hat,
                'h': h,
                'v': v,
                'domain_logits': domain_logits,
                'gamma': gamma,
                'delta': delta,
                'z_tilde': z_tilde
            }
        
        return y_hat
    
    def inference(self, X, s):
        """Inference mode: only use f_phi, m_delta, and p_omega"""
        with torch.no_grad():
            # Temporal extraction
            h = self.f_phi(X)
            
            # FiLM modulation  
            gamma, delta = self.m_delta(s)
            z_tilde = gamma * h + delta
            
            # Prediction
            y_hat = self.p_omega(z_tilde)
            
        return y_hat


def infoNCE_loss(v, reservoir_labels, temperature=0.1):
    """InfoNCE contrastive loss for pseudo-domain embeddings
    Args:
        v: [B, d] pseudo-domain embeddings
        reservoir_labels: list of reservoir names for each sample
        temperature: temperature parameter for softmax
    Returns:
        loss: InfoNCE loss
    """
    device = v.device
    batch_size = v.size(0)
    
    # Create positive/negative mask
    labels_tensor = torch.tensor([hash(label) % 1000 for label in reservoir_labels], device=device)
    labels_expanded = labels_tensor.unsqueeze(1).expand(batch_size, batch_size)
    mask = (labels_expanded == labels_expanded.T).float()
    
    # Compute similarities
    v_norm = F.normalize(v, dim=1)
    similarities = torch.matmul(v_norm, v_norm.T) / temperature
    
    # Remove self-similarities
    mask.fill_diagonal_(0)
    
    # Compute InfoNCE loss
    exp_similarities = torch.exp(similarities)
    positive_sum = (exp_similarities * mask).sum(dim=1)
    negative_sum = exp_similarities.sum(dim=1) - torch.diag(exp_similarities)
    
    # Avoid division by zero
    positive_sum = torch.clamp(positive_sum, min=1e-8)
    
    loss = -torch.log(positive_sum / (positive_sum + negative_sum))
    return loss.mean()


def adversarial_loss(domain_logits, v):
    """Adversarial loss between domain embeddings and hidden features
    Args:
        domain_logits: [B, num_domains] domain classification logits from discriminator
        v: [B, d] pseudo-domain embeddings (targets)
    Returns:
        loss: adversarial loss
    """
    # Use pseudo-domain embeddings to create target domain labels
    # This is a simplified version - in practice you might want more sophisticated target generation
    batch_size = v.size(0)
    
    # Create uniform target distribution (encourage confusion)
    uniform_targets = torch.ones_like(domain_logits) / domain_logits.size(1)
    
    # Use KL divergence loss
    log_probs = F.log_softmax(domain_logits, dim=1)
    loss = F.kl_div(log_probs, uniform_targets, reduction='batchmean')
    
    return loss
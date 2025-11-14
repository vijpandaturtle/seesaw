import torch
import torch.nn.functional as F
import open_clip
from typing import Dict, List, Optional
from PIL import Image


class CLIPModel:
    """
    Wrapper around OpenCLIP model with hook management capabilities.
    Provides base functionality for interpretability techniques.
    """
    
    def __init__(
        self, 
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda"
    ):
        """
        Initialize with OpenCLIP model.
        
        Popular models:
        - ViT-B-32 (laion2b_s34b_b79k) - Good baseline
        - ViT-B-16 (laion2b_s34b_b88k) - Higher resolution
        - ViT-L-14 (laion2b_s32b_b82k) - Larger model
        - ViT-H-14 (laion2b_s32b_b79k) - Huge model
        
        See all: open_clip.list_pretrained()
        """
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        
        # Storage for activations
        self.vision_activations = {}
        self.text_activations = {}
        self.attention_maps = {}
        
        # Hook handles for cleanup
        self.hooks = []
        
        # Get architecture info
        if hasattr(self.model.visual, 'transformer'):
            self.num_vision_layers = len(self.model.visual.transformer.resblocks)
        else:
            self.num_vision_layers = 0
            print("⚠ Warning: Non-transformer vision encoder detected")
        
        if hasattr(self.model, 'transformer'):
            self.num_text_layers = len(self.model.transformer.resblocks)
        else:
            self.num_text_layers = 0
        
        # Get embedding dimension
        self.embed_dim = self.model.visual.output_dim if hasattr(self.model.visual, 'output_dim') else 512
        
        print(f"Loaded {model_name} ({pretrained})")
        print(f"Vision layers: {self.num_vision_layers}")
        print(f"Text layers: {self.num_text_layers}")
        print(f"Embedding dim: {self.embed_dim}")
        print(f"Device: {device}")
    
    
    # ==================== HOOK MANAGEMENT ====================
    
    def register_vision_hooks(self, layer_indices: Optional[List[int]] = None):
        """
        Register hooks on vision transformer blocks.
        layer_indices: Which layers to hook (None = all layers)
        """
        if not hasattr(self.model.visual, 'transformer'):
            raise ValueError("Vision encoder is not a transformer. Hooks not supported for CNN models.")
        
        if layer_indices is None:
            layer_indices = list(range(self.num_vision_layers))
        
        for idx in layer_indices:
            block = self.model.visual.transformer.resblocks[idx]
            handle = block.register_forward_hook(
                self._make_vision_hook(f"layer_{idx}")
            )
            self.hooks.append(handle)
        
        print(f"✓ Registered vision hooks on layers: {layer_indices}")
    
    
    def register_text_hooks(self, layer_indices: Optional[List[int]] = None):
        """Register hooks on text transformer blocks."""
        if layer_indices is None:
            layer_indices = list(range(self.num_text_layers))
        
        for idx in layer_indices:
            block = self.model.transformer.resblocks[idx]
            handle = block.register_forward_hook(
                self._make_text_hook(f"layer_{idx}")
            )
            self.hooks.append(handle)
        
        print(f"Registered text hooks on layers: {layer_indices}")
    
    
    def register_attention_hooks(self, modality: str = "vision", layer_indices: Optional[List[int]] = None):
        """
        Register hooks to capture attention maps.
        modality: 'vision' or 'text'
        """
        if modality == "vision":
            if not hasattr(self.model.visual, 'transformer'):
                raise ValueError("Vision encoder is not a transformer.")
            blocks = self.model.visual.transformer.resblocks
            num_layers = self.num_vision_layers
        else:
            blocks = self.model.transformer.resblocks
            num_layers = self.num_text_layers
        
        if layer_indices is None:
            layer_indices = list(range(num_layers))
        
        for idx in layer_indices:
            handle = blocks[idx].attn.register_forward_hook(
                self._make_attention_hook(f"{modality}_attn_{idx}")
            )
            self.hooks.append(handle)
        
        print(f"Registered attention hooks for {modality} on layers: {layer_indices}")
    
    
    def clear_hooks(self):
        """Remove all hooks and clear stored activations."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self.vision_activations = {}
        self.text_activations = {}
        self.attention_maps = {}
    
    
    def _make_vision_hook(self, name: str):
        """Factory for vision hooks."""
        def hook(module, input, output):
            self.vision_activations[name] = output.detach()
        return hook
    
    
    def _make_text_hook(self, name: str):
        """Factory for text hooks."""
        def hook(module, input, output):
            self.text_activations[name] = output.detach()
        return hook
    
    
    def _make_attention_hook(self, name: str):
        """Factory for attention hooks."""
        def hook(module, input, output):
            self.attention_maps[name] = output.detach() if isinstance(output, torch.Tensor) else None
        return hook
    
    
    # ==================== ENCODING METHODS ====================
    
    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a single image to embedding."""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        return self.model.encode_image(image_input)
    
    
    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text to embeddings."""
        text_tokens = self.tokenizer(texts).to(self.device)
        return self.model.encode_text(text_tokens)
    
    
    def get_vision_projection(self) -> torch.Tensor:
        """Get vision projection matrix."""
        vision_proj = self.model.visual.proj
        if vision_proj is None:
            print("⚠ Warning: No vision projection found. Using identity.")
            return torch.eye(self.embed_dim, device=self.device)
        return vision_proj
    
    
    def get_text_projection(self) -> torch.Tensor:
        """Get text projection matrix."""
        text_proj = self.model.text_projection
        if text_proj is None:
            return torch.eye(self.embed_dim, device=self.device)
        return text_proj
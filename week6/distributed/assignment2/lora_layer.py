import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# The base class for LoRA layers
class LoraLayer:
    def __init__(
        self,
        in_features: int,  # The number of input features
        out_features: int,  # The number of output features
    ):
        # Initializes dictionaries to store various parameters for each adapter in the layer
        self.r = {}  # The rank of the low-rank matrix
        self.lora_alpha = {}  # The scaling factor
        self.scaling = {}  # The calculated scaling factor (lora_alpha / r)

        # Dropout layers for each adapter
        self.lora_dropout = nn.ModuleDict({})

        # Weight matrices for the linear layers
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})

        # Weight matrices for the embedding layers
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})

        # Boolean flag indicating whether the weights have been merged
        self.merged = False

        # Boolean flag indicating whether the adapters are disabled
        self.disable_adapters = False

        # Stores the number of input and output features
        self.in_features = in_features
        self.out_features = out_features
    
    # Method to update the parameters of the layer with a new adapter
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        # Updates the rank and scaling factor for the adapter
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha

        # If dropout rate is greater than 0, creates a dropout layer, otherwise creates an identity layer
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        # Updates the dropout layer for the adapter
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        # If rank is greater than 0, creates trainable parameters for the adapter
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)}))
            self.scaling[adapter_name] = lora_alpha / r

        # If init_lora_weights is True, resets the parameters of the adapter
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        # Moves the layer to the same device as the weight tensor
        self.to(self.weight.device)

     # Method to update the parameters of the embedding layer with a new adapter
    def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        # Updates the rank and scaling factor for the adapter
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha

        # If dropout rate is greater than 0, creates a dropout layer, otherwise creates an identity layer
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        # Updates the dropout layer for the adapter
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        # If rank is greater than 0, creates trainable parameters for the adapter
        if r > 0:
            self.lora_embedding_A.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((r, self.in_features)))})
            )
            self.lora_embedding_B.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((self.out_features, r)))})
            )
            self.scaling[adapter_name] = lora_alpha / r

        # If init_lora_weights is True, resets the parameters of the adapter
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        # Moves the layer to the same device as the weight tensor
        self.to(self.weight.device)

    # Method to reset the parameters of an adapter
    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

# LoRA implemented in an Embedding layer
class Embedding(nn.Embedding, LoraLayer):
    """
    The Embedding class is an extension of the PyTorch nn.Embedding class 
    and LoraLayer class to incorporate the LoRA method.
    """
    def __init__(
        self,
        adapter_name: str,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        # Pop the init_lora_weights flag from kwargs
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        # Call the constructors of the parent classes
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoraLayer.__init__(self, in_features=num_embeddings, out_features=embedding_dim)

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        # Reset the parameters of the Embedding layer and update it with the adapter
        nn.Embedding.reset_parameters(self)
        self.update_layer_embedding(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

        # Set the active adapter
        self.active_adapter = adapter_name

    # Separate low-rank approximation from original weight
    def unmerge(self, mode: bool = True):
        # If the weights are already unmerged, raise a warning
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        # If the rank of the active adapter is greater than 0, subtract the product of the LoRA weights
        # from the weights of the embedding
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.lora_embedding_B[self.active_adapter] @ self.lora_embedding_A[self.active_adapter], True
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = False

    # Merge low-rank approximation with original weights
    def merge(self):
        # If the weights are already merged, raise a warning
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        # If the rank of the active adapter is greater than 0, add the product of the LoRA weights
        # to the weights of the embedding
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                transpose(
                    self.lora_embedding_B[self.active_adapter] @ self.lora_embedding_A[self.active_adapter], True
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = True

    # Defines the computation performed at every call.
    def forward(self, x: torch.Tensor):
        # If adapters are disabled and there is an active adapter with rank > 0 and it is merged
        # Subtract the LoRA weights from the original weights and set merged to False
        if self.disable_adapters:
            if self.r[self.active.adapter] > 0 and self.merged:
                self.weight.data -= (
                    transpose(
                        self.lora_embedding_B[self.active_adapter].weight
                        @ self.lora_embedding_A[self.active_adapter].weight,
                        True,
                    )
                    * self.scaling[self.active_adapter]
                )
                self.merged = False
            # Forward pass with the original weights
            return nn.Embedding.forward(self, x)

        # If there is an active adapter with rank > 0 and it is not merged
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            # Compute the forward pass with the LoRA weights and add it to the result
            if self.r[self.active_adapter] > 0:
                after_A = F.embedding(
                    x,
                    self.lora_embedding_A[self.active_adapter].T,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.lora_embedding_B[self.active_adapter].T) * self.scaling[self.active_adapter]
            return result
        else:
            return nn.Embedding.forward(self, x)


# Lora is implemented in a dense (Linear) layer
class Linear(nn.Linear, LoraLayer):
    
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        # Initialize weights for LoRA layer
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        # Initialize linear and LoRA layers
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        # Transpose the weight if the layer to replace stores weight like (fan_in, fan_out)
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        # Reset linear layer parameters and update LoRA layer
        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def merge(self):
        # Merge low-rank approximation with original weights
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            # TODO: Merge the LoRA parameters by adding the product of lora_B weights and lora_A weights (after transposing 
            # if necessary) to the original weights, scaled by the LoRA scaling factor. After this operation, set the merged
            # flag to True.
            self.weight.data += None ### YOUR CODE HERE ###
            self.merged = None ### YOUR CODE HERE ###

    def unmerge(self):
        # Separate low-rank approximation from original weights
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = False

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            # Changing data type for ensuring consistency
            x = x.to(self.lora_A[self.active_adapter].weight.dtype)
            
            # TODO: If the LoRA adapter is active and not merged, add the output of the LoRA layers to the result. This involves
            # passing the input through lora_A, applying dropout, then passing it through lora_B. The output is scaled by the
            # LoRA scaling factor and added to the result.
            result += None ### YOUR CODE HERE ###
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        
        # Reverting to the previous data type
        result = result.to(previous_dtype)
        return result
    
def transpose(weight, fan_in_fan_out):
    # Helper function to transpose weights if required
    return weight.T if fan_in_fan_out else weight

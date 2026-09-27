import torch
import torch.nn as nn
from torch import Tensor


class SlotAttentionPooling(nn.Module):
    """
    Pools the encoder sequence into one context vector per query. Each query is
    learned, so a head can attend to the moment its evidence was spoken instead
    of sharing a single mean over the whole utterance.

    Flow: (B, dim, T) -> (B, num_queries, dim)
    """

    def __init__(
        self,
        dim: int,
        num_slots: int,
    ) -> None:
        """
        Initialize the attention pooling.

        Args:
            dim: Encoder embedding dimension
            num_slots: Number of queries, one learned vector each
        """
        super().__init__()

        self.queries = nn.Parameter(torch.randn(num_slots, dim) * dim ** -0.5)
        self.scale = dim ** -0.5

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Frame embeddings of shape (B, dim, T)

        Returns:
            Per-slot context vectors of shape (B, num_slots, dim)
        """
        # (num_slots, dim) x (B, dim, T) -> (B, num_slots, T)
        scores = torch.einsum("sd,bdt->bst", self.queries, x) * self.scale
        weights = torch.softmax(scores, dim=-1)

        # (B, num_slots, T) x (B, dim, T) -> (B, num_slots, dim)
        context = torch.einsum("bst,bdt->bsd", weights, x)

        return context


class ClassificationHeads(nn.Module):
    """
    One linear classifier for the intent and one per slot.

    The intent head reads a mean over time concatenated with its own
    attention-pooled context. Mean pooling alone was enough while every
    utterance was a bare template, but real requests wrap the command in
    carrier phrases - "olly can you please turn off the light of my bed room" -
    and averaging over a one-second preamble dilutes the half-second that
    carries the intent. The attention term lets the head find that half-second;
    the mean term keeps the global evidence it had before.

    Flow: (B, dim, T) -> {"intent": (B, num_intents), <slot>: (B, num_classes)}
    """

    def __init__(
        self,
        dim: int,
        num_intents: int,
        slot_sizes: dict[str, int],
    ) -> None:
        """
        Initialize the classification heads.

        Args:
            dim: Encoder embedding dimension
            num_intents: Number of intent classes including the reject class
            slot_sizes: Mapping of slot name to its number of classes
        """
        super().__init__()

        self.slot_names = list(slot_sizes.keys())

        # Query 0 is the intent's, the rest belong to the slots in order
        self.pooling = SlotAttentionPooling(dim=dim, num_slots=len(self.slot_names) + 1)
        self.intent_head = nn.Linear(dim * 2, num_intents)
        self.slot_heads = nn.ModuleList(
            [nn.Linear(dim, slot_sizes[name]) for name in self.slot_names]
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Args:
            x: Frame embeddings of shape (B, dim, T)

        Returns:
            Logits keyed by head name
        """
        context = self.pooling(x)

        # (B, dim) mean + (B, dim) attended -> (B, 2 * dim)
        intent_context = torch.cat([x.mean(dim=-1), context[:, 0]], dim=-1)
        logits = {"intent": self.intent_head(intent_context)}

        for i, name in enumerate(self.slot_names):
            logits[name] = self.slot_heads[i](context[:, i + 1])

        return logits

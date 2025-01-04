import argparse
import os
from typing import List, Literal, Tuple

import torch
from datasets import load_dataset
from huggingface_hub import PyTorchModelHubMixin
from safetensors.torch import safe_open, save_file
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device(device_id: int = None) -> Tuple[bool, torch.device]:
    if torch.cuda.is_available():
        use_cuda, device = (
            True,
            f"cuda:{device_id}" if device_id is not None else "cuda",
        )
    else:
        use_cuda, device = False, "cpu"
    return use_cuda, torch.device(device)


def save_shard(
    tensors: List[torch.Tensor], shard_path: str, shard_index: int
) -> None:
    os.makedirs(shard_path, exist_ok=True)
    path = os.path.join(shard_path, f"{shard_index:07d}.safetensors")
    tensors = torch.concat(tensors, dim=0)
    save_file({"tensors": tensors}, path)


@torch.no_grad()
def cache_activations(
    model_name_or_path: str,
    dataset_name_or_path: str,
    model_layer_index: int,
    batch_size: int,
    shard_path: str,
    shard_size: int,
    device_id: int,
) -> None:
    use_cuda, device = get_device(device_id)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map={}
    )

    activations, shard_index = [], 1
    dataset = load_dataset(dataset_name_or_path, split="train")
    num_steps = len(dataset) // batch_size
    for i in range(0, len(dataset), num_steps):
        x = dataset[i : i + num_steps]["train"]
        inputs = tokenizer(x, return_tensors="pt")
        hidden_states = (
            model(**inputs, output_hidden_states=True)
            .hidden_states[model_layer_index]
            .cpu()
        )
        activations.extend(hidden_states)
        if len(activations) >= shard_size:
            save_shard(activations, shard_path, shard_index)
            activations, shard_index = [], shard_index + 1

    if len(activations) > 0:
        save_shard(activations, shard_path, shard_index)


def load_cached_activations(shard_path: str) -> torch.Tensor:
    activations = []
    for i, shard_file_path in enumerate(os.listdir(shard_path)):
        path = os.path.join(shard_path, shard_file_path)
        with safe_open(path, framework="pt") as f:
            tensors = f.get_tensor("tensors")
        if i == 0:
            activations = tensors
        else:
            activations = torch.concat((activations, tensors), dim=0)
    return activations


class SAE(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        acts_size: int,
        dict_size: int,
        topk: Literal["soft", "hard"] = "soft",
        k: int = 10,
    ) -> None:
        super().__init__()
        self.topk = topk
        self.k = k
        self.encode = nn.Linear(acts_size, dict_size)
        self.decode = nn.Linear(dict_size, acts_size, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pre_acts = self.encode(input)
        if self.topk == "soft":
            acts = pre_acts.relu()
        elif self.topk == "hard" and self.k is not None:
            _, indices = torch.topk(pre_acts, self.k, dim=1)
            acts = torch.zeros_like(pre_acts)
            acts.scatter_(1, indices, pre_acts)
        recon_input = self.decode(acts)
        return recon_input, acts


def sae_loss_fn(
    input: torch.Tensor,
    recon_input: torch.Tensor,
    acts: torch.Tensor,
    l1_coef: float = 1.0,
    topk: Literal["soft", "hard"] = "soft",
) -> torch.Tensor:
    loss = nn.functional.mse_loss(input, recon_input)
    if topk == "soft":
        loss += l1_coef * torch.mean(acts)
    return loss


def main(
    sae_model_name: str,
    model_name_or_path: str,
    dataset_name_or_path: str,
    activations_path: str,
    shard_size: int,
    acts_size: int,
    dict_size: int,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    model_layer_index: int,
    device_id: int = None,
    l1_coef: float = 1.0,
    topk: Literal["soft", "hard"] = "soft",
    k: int = 10,
):
    use_cuda, device = get_device(device_id)
    model = SAE(acts_size=acts_size, dict_size=dict_size, topk=topk, k=k)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if activations is None or not os.path.exists(activations_path):
        cache_activations(
            model_name_or_path=model_name_or_path,
            dataset_name_or_path=dataset_name_or_path,
            model_layer_index=model_layer_index,
            batch_size=32,
            shard_path=activations_path,
            shard_size=shard_size,
            device_id=device_id,
        )
    activations = load_cached_activations(activations_path)
    dataset = TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)

    for i in range(num_epochs):
        for (x,) in loader:
            x = x.to(device=device)
            recon_x, acts = model(x)
            loss = sae_loss_fn(x, recon_x, acts, l1_coef=l1_coef, topk=topk)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.save_pretrained(sae_model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAE model")
    parser.add_argument("--saex_model_name", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--dataset_name_or_path", type=str)
    parser.add_argument("--activations_path", type=str)
    parser.add_argument("--shard_size", type=int)
    parser.add_argument("--acts_size", type=int)
    parser.add_argument("--dict_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--model_layer_index", type=int)
    parser.add_argument("--device_id", type=int, default=None)
    parser.add_argument("--l1_coef", type=float, default=1.0)
    parser.add_argument("--topk", type=str, default="soft")
    parser.add_argument("--k", type=int, default=10)

    args = parser.parse_args()

    main(
        sae_model_name=args.sae_model_name,
        model_name_or_path=args.model_name_or_path,
        dataset_name_or_path=args.dataset_name_or_path,
        activations_path=args.activations_path,
        shard_size=args.shard_size,
        acts_size=args.acts_size,
        dict_size=args.dict_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        model_layer_index=args.model_layer_index,
        device_id=args.device_id,
        l1_coef=args.l1_coef,
        topk=args.topk,
        k=args.k,
    )

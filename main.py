import argparse
import gc
import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import wandb
from datasets import load_dataset
from huggingface_hub import PyTorchModelHubMixin
from safetensors.torch import safe_open, save_file
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device(device_id: Optional[int] = None) -> Tuple[bool, torch.device]:
    if torch.cuda.is_available():
        use_cuda = True
        if device_id is not None and isinstance(device_id, int):
            device = f"cuda:{device_id}"
        else:
            device = "cuda"
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
        model_name_or_path, device_map={"": device}
    )

    activations, shard_index = [], 1
    dataset = load_dataset(dataset_name_or_path, split="train")
    num_steps = len(dataset) // batch_size
    for i in range(0, len(dataset), num_steps):
        x = dataset[i : i + num_steps]["train"]
        inputs = tokenizer(x, return_tensors="pt")
        inputs = {k: v.to(device=device) for k, v in inputs.items()}
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
    del model
    torch.cuda.empty_cache()
    gc.collect()


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
        mode: Literal["relu", "topk"] = "relu",
        k: int = 10,
        l1_coef: float = 1.0,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.k = k
        self.l1_coef = l1_coef
        self.encode = nn.Linear(acts_size, dict_size)
        self.decode = nn.Linear(dict_size, acts_size, bias=False)

    def forward(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        acts = self.encode(input)
        if self.mode == "relu":
            acts = acts.relu()
        elif self.mode == "topk" and self.k is not None:
            _, indices = torch.topk(acts, self.k, dim=-1)
            acts *= torch.zeros_like(acts).scatter_(-1, indices, 1)
        recon_input = self.decode(acts)
        loss_dict = self.loss_fn(input, recon_input, acts)
        return {
            "recon_input": recon_input,
            "input": input,
            "activation": acts,
            **loss_dict,
        }

    def loss_fn(
        self,
        input: torch.Tensor,
        recon_input: torch.Tensor,
        acts: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss = nn.functional.mse_loss(input, recon_input)
        out = {
            "loss": loss,
            "l1_loss": torch.mean(acts),
            "l0_loss": (acts != 0).float().mean(),
        }
        if self.mode == "relu":
            out["loss"] += self.l1_coef * out["l1_loss"]
        return out


def main(
    wandb_project: str,
    run_name: str,
    model_name_or_path: str,
    dataset_name_or_path: str,
    activations_path: str,
    acts_size: int,
    dict_size: Optional[int] = None,
    expansion_factor: int = 4,
    shard_size: int = 10_000,
    batch_size: int = 2048,
    learning_rate: float = 1e-4,
    num_epochs: int = 10,
    model_layer_index: int = 8,
    device_id: int = None,
    l1_coef: float = 1e-2,
    mode: Literal["relu", "topk"] = "topk",
    k: int = 10,
    log_interval: int = 100,
):
    metadata = {
        "wandb_project": wandb_project,
        "run_name": run_name,
        "model_name_or_path": model_name_or_path,
        "dataset_name_or_path": dataset_name_or_path,
        "activations_path": activations_path,
        "shard_size": shard_size,
        "acts_size": acts_size,
        "dict_size": dict_size,
        "batch_size": batch_size,
        "expansion_factor": expansion_factor,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "model_layer_index": model_layer_index,
        "device_id": device_id,
        "l1_coef": l1_coef,
        "mode": mode,
        "k": k,
        "log_interval": log_interval,
    }
    wandb.init(project=wandb_project, name=run_name, config=metadata)

    use_cuda, device = get_device(device_id)
    if dict_size is None:
        dict_size = acts_size * expansion_factor
    model = SAE(acts_size, dict_size, mode, k, l1_coef)
    model.to(device=device)

    # to automatically log gradients
    wandb.watch(model, log_freq=100)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if activations_path is None or not os.path.exists(activations_path):
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
    train_dataset, eval_dataset = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size, shuffle=False)

    for i in trange(num_epochs, desc="Epochs"):
        model.train()
        for batch_index, (x,) in enumerate(train_loader, start=1):
            x = x.to(device=device)
            output = model(x)
            output["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_index % log_interval == 0:
                wandb.log(
                    {
                        "loss/train": output["loss"].item(),
                        "l1_loss/train": output["l1_loss"].item(),
                        "l0_loss/train": output["l0_loss"].item(),
                    }
                )

        model.eval()
        with torch.no_grad():
            t = {"loss": 0.0, "l1_loss": 0.0, "l0_loss": 0.0}
            for batch_index, (x,) in enumerate(eval_loader, start=1):
                x = x.to(device=device)
                output = model(x)
                t["loss"] += output["loss"].item()
                t["l1_loss"] += output["l1_loss"].item()
                t["l0_loss"] += output["l0_loss"].item()

            wandb.log(
                {
                    "loss/eval": t["loss"] / len(eval_loader),
                    "l1_loss/eval": t["l1_loss"] / len(eval_loader),
                    "l0_loss/eval": t["l0_loss"] / len(eval_loader),
                }
            )
    model.save_pretrained(run_name)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAE model")
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--dataset_name_or_path", type=str)
    parser.add_argument("--activations_path", type=str)
    parser.add_argument("--shard_size", type=int, default=10_000)
    parser.add_argument("--acts_size", type=int)
    parser.add_argument("--dict_size", type=int, required=False)
    parser.add_argument("--expansion_factor", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--model_layer_index", type=int, default=8)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--l1_coef", type=float, default=1e-2)
    parser.add_argument("--mode", type=str, default="topk")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=100)

    args = parser.parse_args()

    main(
        wandb_project="sae",
        run_name=args.run_name,
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
        mode=args.mode,
        k=args.k,
        log_interval=args.log_interval,
    )

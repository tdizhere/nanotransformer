import argparse
import time

import torch

from dataset import get_batch, load_dataset
from model import BigramLanguageModel, GPTLanguageModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("bigram", "gpt"), default="gpt")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--benchmark-steps", type=int, default=1, help="Steps used to estimate total training time")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size or (32 if args.model == "bigram" else 128)
    block_size = args.block_size or (8 if args.model == "bigram" else 256)
    dataset = load_dataset()
    model = BigramLanguageModel(dataset.vocab_size) if args.model == "bigram" else GPTLanguageModel(dataset.vocab_size, block_size=block_size)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2 if args.model == "bigram" else 1e-4)

    start = time.perf_counter()
    for _ in range(max(1, args.benchmark_steps)):
        x, y = get_batch(dataset.train_data, batch_size, block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start
    seconds_per_step = elapsed / max(1, args.benchmark_steps)
    print(f"Benchmark: {seconds_per_step:.4f} seconds/step")
    print(f"Estimated training time for {args.iterations:,} steps: {seconds_per_step * args.iterations / 60:.2f} minutes")

    for step in range(args.benchmark_steps, args.iterations):
        x, y = get_batch(dataset.train_data, batch_size, block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")


if __name__ == "__main__":
    main()

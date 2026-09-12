# Generative Pretrained Transformer Language Model

## 1. Project Description

This project implements two character-level language models trained on the text of Shakespeare plays:

1. A bigram language model.
2. A decoder-only GPT-style transformer language model.

The project downloads the Shakespeare plays dataset through kagglehub, converts the text into integer character tokens, trains a model to predict the next character, and supports autoregressive text generation in the model classes.

The repository preserves the original experimental scripts:

- bigram.py: the original standalone bigram experiment.
- gpt.py: the original standalone GPT-style experiment.

The modular implementation is separated into:

- dataset.py: dataset download, tokenization, encoding, decoding, and batch construction.
- model.py: bigram and GPT-style model definitions.
- train.py: command-line training and training-time estimation.

This is an educational implementation. It demonstrates tokenization, causal self-attention, optimization, and autoregressive generation. It is not intended to match the quality, scale, or production guarantees of a modern large language model.

## 2. Research Objective

The objective is to train a small character-level autoregressive model on Shakespearean text and examine how model capacity affects next-character prediction.

The bigram model is a low-capacity baseline. The GPT-style model adds context, learned positional information, causal multi-head self-attention, feed-forward layers, residual connections, layer normalization, and dropout.

The central prediction task is:

> Given a sequence of characters, predict the next character at every position in the sequence.

## 3. Repository Structure

~~~text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── bigram.py
├── gpt.py
├── dataset.py
├── model.py
└── train.py
~~~

### 3.1 Original scripts

bigram.py and gpt.py are the original standalone experiments. When executed, they download the dataset, construct the model, train immediately, and generate text after training.

These files were intentionally preserved. The modular files provide a cleaner organization without replacing the original experiments.

### 3.2 Modular scripts

| File | Responsibility |
|---|---|
| dataset.py | Downloads the corpus, builds the vocabulary, encodes characters, creates train and validation tensors, and returns random batches. |
| model.py | Defines the bigram model, GPT-style model, causal attention heads, and transformer blocks. |
| train.py | Parses command-line arguments, selects a model, runs optimization, and estimates training time. |

## 4. Software Requirements

### 4.1 Required software

- Python 3.9 or newer.
- PyTorch 2.0 or newer.
- kagglehub 0.3 or newer.
- Internet access during the first dataset download.
- uv recommended for environment management.

Dependencies are declared in:

- requirements.txt, for pip-compatible installation.
- pyproject.toml, for uv sync.

### 4.2 Operating systems

The code uses standard Python and PyTorch APIs and is intended for Windows, Linux, and macOS.

GPU acceleration depends on the PyTorch build installed for the operating system. The same Python code can run on a CPU, but the execution time can be substantially longer for the default GPT configuration.

### 4.3 Network access

The first dataset load calls:

~~~python
kagglehub.dataset_download("kingburrito666/shakespeare-plays")
~~~

The downloaded directory is expected to contain:

~~~text
alllines.txt
~~~

If the dataset is cached locally by kagglehub, later runs may not need to download it again. Cache location and cache behavior are controlled by kagglehub.

## 5. Installation

### 5.1 Recommended installation with uv

From the project directory:

~~~bash
uv sync
~~~

Run the project through the managed environment:

~~~bash
uv run python train.py
~~~

The first uv sync may download PyTorch and its transitive dependencies. Installation time depends on network speed, operating system, Python version, and whether a compatible GPU build is selected.

### 5.2 Installation with pip

Create a virtual environment:

~~~bash
python -m venv .venv
~~~

Activate it on Windows PowerShell:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

Activate it on Linux or macOS:

~~~bash
source .venv/bin/activate
~~~

Install the dependencies:

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

For a CUDA-specific installation, use the official PyTorch installation instructions for the target operating system, Python version, CUDA version, and GPU. The generic dependency file declares PyTorch but does not list every platform-specific CUDA wheel choice.

## 6. GPU and Hardware Support

### 6.1 Device selection

The trainer selects the device using:

~~~python
"cuda" if torch.cuda.is_available() else "cpu"
~~~

If CUDA is unavailable, the code falls back to the CPU. It does not stop with an error merely because no GPU is present.

### 6.2 Verify CUDA

Run:

~~~bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
~~~

A functioning CUDA setup should print True followed by the GPU name.

If the result is False, common causes are:

- A CPU-only PyTorch installation.
- An incompatible NVIDIA driver.
- A PyTorch build that does not match the environment.
- No NVIDIA GPU.
- A CUDA runtime or driver configuration problem.

### 6.3 NVIDIA GPU recommendation

An NVIDIA GPU with a compatible CUDA-enabled PyTorch installation is recommended for the default GPT configuration.

The default GPT configuration uses:

~~~text
batch size:          128
sequence length:     256
embedding dimension: 384
attention heads:     6
transformer blocks:  6
dropout:             0.4
~~~

A GPU with limited memory may not be able to run this configuration.

### 6.4 CPU support

CPU execution is appropriate for:

- Installation checks.
- One-step timing tests.
- Small bigram experiments.
- Reduced GPT smoke tests.
- Debugging.

CPU training of the default GPT configuration can be slow because it performs repeated matrix multiplications and attention operations over sequences of length 256 with a batch size of 128.

### 6.5 Other accelerators

The trainer explicitly selects cuda or cpu. It does not currently contain dedicated device-selection logic for:

- Apple Metal Performance Shaders (mps).
- ROCm-specific device handling.
- TPU execution.
- Other accelerator backends.

Using those backends may require a PyTorch-specific installation and a future change to device selection.

## 7. Memory Requirements and Out-of-Memory Risk

The default GPT model is much more demanding than the bigram baseline.

Important memory consumers include:

1. Model parameters.
2. AdamW optimizer state.
3. Forward activations retained for backpropagation.
4. Attention score matrices.
5. Batch size and sequence length.

Self-attention has quadratic sequence-length behavior. Increasing block size from 128 to 256 increases the size of the attention score relationship substantially.

A CUDA out-of-memory error is the most likely serious runtime failure on a smaller GPU.

Reduce memory use with:

~~~bash
uv run python train.py --model gpt --batch-size 32 --block-size 128
~~~

If memory is still insufficient:

~~~bash
uv run python train.py --model gpt --batch-size 8 --block-size 128
~~~

Further reduction is possible:

~~~bash
uv run python train.py --model gpt --batch-size 4 --block-size 64
~~~

The effects are:

- Lower batch size means fewer sequences are processed simultaneously.
- Lower block size means the model sees a shorter context.
- Lower values alter training behavior and make timing comparisons with the default configuration invalid.

After an out-of-memory error, restart the process so that failed allocations and cached memory are released.

## 8. Dataset Pipeline

The dataset pipeline is implemented in dataset.py.

### 8.1 Dataset download

load_dataset() calls kagglehub with the dataset identifier:

~~~text
kingburrito666/shakespeare-plays
~~~

The code then opens:

~~~text
alllines.txt
~~~

from the downloaded directory.

The dataset is not committed to the repository. It is downloaded at runtime.

### 8.2 Vocabulary construction

The complete text is read into memory. Unique characters are sorted:

~~~python
characters = sorted(set(text))
~~~

Two mappings are created:

- stoi: character to integer token ID.
- itos: integer token ID to character.

Sorting makes the mapping deterministic for the same input text.

### 8.3 Character-level encoding

Every character is converted to an integer token ID. The encoded corpus is stored as a one-dimensional torch.long tensor.

Spaces, punctuation, newline characters, uppercase letters, and other symbols are all treated as separate tokens. There is no word-level or subword-level tokenizer.

### 8.4 Train-validation split

The default split is:

~~~text
training data:   first 90 percent
validation data: final 10 percent
~~~

The validation tensor is created and returned by load_dataset(). The current modular trainer uses the training tensor for optimization but does not currently execute a validation-loss loop.

### 8.5 Batch construction

For each batch, random starting positions are selected. Each input sequence has length block size. The target is shifted by one character.

Example:

~~~text
input:  c1 c2 c3 c4
target: c2 c3 c4 c5
~~~

The target at position t is therefore the next character after the input at position t.

## 9. Model Architecture

### 9.1 BigramLanguageModel

The bigram model uses one embedding table with dimensions:

~~~text
vocabulary size x vocabulary size
~~~

For an input tensor shaped:

~~~text
batch size x sequence length
~~~

the model produces logits shaped:

~~~text
batch size x sequence length x vocabulary size
~~~

When targets are present, logits and targets are flattened across the batch and sequence dimensions. Cross-entropy loss is computed over the vocabulary dimension.

The model does not use attention or a hidden state. It primarily learns local character-transition statistics.

### 9.2 GPTLanguageModel

The GPT-style model contains:

1. Token embeddings.
2. Learned positional embeddings.
3. A sequence of transformer blocks.
4. Final layer normalization.
5. A linear language-model head.

The token and positional embeddings are added before entering the transformer blocks.

The default configuration is:

~~~text
embedding dimension: 384
attention heads:     6
transformer blocks:  6
dropout:             0.4
sequence length:     256
~~~

### 9.3 Causal self-attention

Each attention head computes:

- Keys.
- Queries.
- Values.

The query-key matrix product creates attention scores. The scores are scaled by the square root of the head size.

A lower-triangular mask replaces future positions with negative infinity. After softmax, future positions receive zero probability. This enforces causal prediction.

The attention output is the weighted sum of values. Outputs from all heads are concatenated and projected back to the model embedding dimension.

### 9.4 Transformer block

Each block has two residual pathways:

~~~text
x = x + attention(layer_norm(x))
x = x + feed_forward(layer_norm(x))
~~~

The feed-forward network:

1. Expands the embedding dimension by four.
2. Applies ReLU.
3. Projects back to the original dimension.
4. Applies dropout.

### 9.5 Loss function

Both models use categorical cross-entropy:

~~~text
cross_entropy(predicted_logits, target_token_ids)
~~~

The loss is calculated for every position in every sequence and averaged across the flattened batch.

### 9.6 Text generation

Generation is autoregressive:

1. Start with an initial token tensor.
2. Run the model on the current context.
3. Select logits from the final time step.
4. Apply softmax.
5. Sample one token with torch.multinomial().
6. Append the token.
7. Repeat for the requested number of new tokens.

The GPT model crops the context to the most recent block size tokens before each prediction, preventing the sequence from exceeding the positional-embedding limit.

## 10. Training Procedure

train.py performs the following operations:

1. Parse command-line arguments.
2. Choose CUDA when available; otherwise choose CPU.
3. Select model-specific defaults.
4. Download and encode the dataset.
5. Construct the selected model.
6. Move the model to the selected device.
7. Create an AdamW optimizer.
8. Run benchmark steps.
9. Calculate average seconds per step.
10. Estimate total training time.
11. Run the remaining optimization steps.
12. Print the loss every 100 steps.

Each optimization step performs:

1. Random batch selection.
2. Forward pass.
3. Cross-entropy calculation.
4. Gradient reset.
5. Backward pass.
6. AdamW parameter update.

The benchmark steps are real optimization steps. They update the model before the remaining training loop begins.

## 11. Training-Time Estimation

### 11.1 One-step benchmark

Run one bigram training step:

~~~bash
uv run python train.py --model bigram --iterations 1 --benchmark-steps 1
~~~

Run one GPT training step:

~~~bash
uv run python train.py --model gpt --iterations 1 --benchmark-steps 1
~~~

The command prints information similar to:

~~~text
Benchmark: <seconds per step> seconds/step
Estimated training time for <number of steps>: <minutes> minutes
~~~

### 11.2 Estimate a full run

~~~bash
uv run python train.py --model gpt --iterations 5000 --benchmark-steps 1
~~~

The estimate uses:

~~~text
estimated seconds = seconds per benchmark step x requested iterations
estimated minutes = estimated seconds / 60
~~~

### 11.3 More stable estimate

One step can be noisy because of Python startup, memory allocation, CUDA kernel startup, and other warm-up effects. Use multiple benchmark steps:

~~~bash
uv run python train.py --model gpt --iterations 5000 --benchmark-steps 10
~~~

The average is calculated as total benchmark time divided by the number of benchmark steps.

### 11.4 Timing limitations

The timing estimate is approximate.

It does not include:

- Initial dataset download time.
- Full process startup time.
- All environment initialization overhead.

On CUDA, GPU operations can execute asynchronously. The current measurement uses time.perf_counter() but does not explicitly synchronize CUDA before reading the end time. Consequently, the reported first-step GPU time may be less precise than a synchronized benchmark.

For fair comparisons, keep the following constant:

- Model.
- Device.
- Batch size.
- Block size.
- PyTorch version.
- Benchmark-step count.
- Hardware state.

## 12. Command-Line Reference

### 12.1 Model

~~~bash
--model bigram
--model gpt
~~~

Default:

~~~text
gpt
~~~

### 12.2 Iterations

~~~bash
--iterations 5000
~~~

This is the requested total number of optimization steps, including benchmark steps.

Default:

~~~text
5000
~~~

### 12.3 Benchmark steps

~~~bash
--benchmark-steps 1
~~~

This is the number of real training steps used to estimate average step time.

Default:

~~~text
1
~~~

Use a value between 5 and 20 when a more stable estimate is needed.

### 12.4 Batch size

~~~bash
--batch-size 32
~~~

This overrides the model-specific default.

Defaults:

~~~text
bigram: 32
gpt:    128
~~~

### 12.5 Block size

~~~bash
--block-size 128
~~~

This overrides the model-specific sequence length.

Defaults:

~~~text
bigram: 8
gpt:    256
~~~

For GPT, the block size also controls the learned positional-embedding table and causal attention mask.

## 13. Recommended Run Sequence

### 13.1 Verify installation

~~~bash
uv run python -c "import torch, kagglehub; print(torch.__version__)"
~~~

### 13.2 Verify device selection

~~~bash
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
~~~

### 13.3 Run a CPU-friendly bigram smoke test

~~~bash
uv run python train.py --model bigram --iterations 5 --benchmark-steps 1
~~~

### 13.4 Run a reduced-memory GPT smoke test

~~~bash
uv run python train.py --model gpt --iterations 5 --benchmark-steps 1 --batch-size 8 --block-size 64
~~~

### 13.5 Run default GPT training

~~~bash
uv run python train.py --model gpt --iterations 5000 --benchmark-steps 10
~~~

### 13.6 Run default bigram training

~~~bash
uv run python train.py --model bigram --iterations 2500 --benchmark-steps 10
~~~

## 14. Failure Modes and Troubleshooting

### 14.1 CUDA out-of-memory error

Cause: the model, optimizer, activations, or attention matrices do not fit in available GPU memory.

Try:

~~~bash
uv run python train.py --model gpt --batch-size 8 --block-size 128
~~~

If the failure continues, reduce both values further. Restart the process after the error.

### 14.2 CUDA is not detected

Run:

~~~bash
uv run python -c "import torch; print(torch.cuda.is_available())"
~~~

If the result is False, check the PyTorch build, NVIDIA driver, CUDA compatibility, and physical GPU availability.

The program will use CPU execution when CUDA is unavailable.

### 14.3 Dataset download failure

Possible causes include:

- No internet connection.
- Kaggle service interruption.
- Dataset identifier changes.
- Local cache permissions.
- The downloaded data does not contain allines.txt.

Confirm network access and verify that kagglehub is installed.

### 14.4 Missing allines.txt

The loader expects this exact file:

~~~text
alllines.txt
~~~

If a different dataset layout is downloaded, loading fails before training. The dataset loader would need to be adapted to the new file layout.

### 14.5 Invalid sequence length

The batch function requires the selected data split to be longer than block size.

An excessively large value can cause invalid random-index bounds or incomplete batch construction.

Use a smaller value:

~~~bash
uv run python train.py --model gpt --block-size 128
~~~

### 14.6 Training is unexpectedly slow

Possible causes include:

- CPU fallback.
- Large batch size.
- Large block size.
- CUDA warm-up.
- Limited system memory and swapping.
- Running in a constrained virtual machine.
- Other processes using the GPU.

Check the selected device and reduce batch size or block size for testing.

### 14.7 uv is not installed

Install uv, or use the standard virtual-environment and pip commands in the installation section.

### 14.8 PowerShell activation is blocked

If PowerShell refuses to activate .venv, use uv run directly or follow the local system's approved PowerShell execution-policy procedure.

## 15. Reproducibility

The original scripts set:

~~~python
torch.manual_seed(1337)
~~~

The modular train.py currently does not set a global random seed before model initialization and batch sampling. Therefore, repeated modular runs can produce different results.

For every experiment, record:

- Python version.
- PyTorch version.
- kagglehub version.
- Operating system.
- Device.
- GPU model.
- Driver and CUDA versions, if relevant.
- Model selection.
- Batch size.
- Block size.
- Learning rate.
- Number of iterations.
- Benchmark steps.
- Seconds per step.
- Training loss.
- Validation loss, if implemented.
- Generated sample.

Even with a seed, GPU operations may remain nondeterministic depending on PyTorch kernels and backend settings.

## 16. Current Limitations

The modular implementation is intentionally compact. Its main limitations are:

1. The validation tensor is created but validation loss is not currently calculated during training.
2. Model checkpoints are not saved.
3. Interrupted training cannot be resumed.
4. The modular trainer focuses on training and timing; it does not currently print a generated sample after training.
5. The dataset is downloaded at runtime rather than versioned inside the repository.
6. The generic dependency declaration does not specify every platform-specific CUDA wheel.
7. Mixed precision is not implemented.
8. Gradient accumulation is not implemented.
9. Distributed training is not implemented.
10. Gradient clipping is not implemented.
11. Benchmark steps update model parameters before the main training loop.
12. One-step timing can be noisy.
13. Character-level tokenization produces longer sequences than subword tokenization for the same semantic content.
14. There is no experiment tracking system for metrics, configuration, or generated samples.

These limitations do not prevent the project from demonstrating the core concepts. They should be stated in any formal report or publication.

## 17. FUTURE PLAN TO CONVERT IT INTO LOOP TRANSFORMER LIKE GPT 6 ASTRA
## 18. Interpretation of Results

The bigram model should be treated as a baseline. It can learn local character frequencies and common transitions, but it cannot directly use a long context.

The GPT-style model can use up to block size characters of context. Causal self-attention allows each position to combine information from earlier positions while preventing future-token leakage.

Training loss alone is not sufficient to evaluate language quality. A lower training loss can indicate better fitting of the training corpus without guaranteeing better generalization. When evaluation is added, compare:

- Training loss.
- Validation loss.
- Qualitative generated samples.
- Training time.
- Parameter count.
- Hardware configuration.

Only compare loss or timing values across experiments with compatible datasets, tokenization, model configurations, and optimization procedures.

## 19. Publication and Repository Notes

Before presenting this project as a formal research artifact, consider adding:

- A software license.
- A dataset citation and usage notice.
- A locked dependency file.
- A formal validation-loss evaluation.
- Checkpoint saving.
- Experiment logs.
- Configuration files.
- Fixed random seeds.
- Hardware and runtime details.
- Training curves.
- Generated text samples.
- A clear statement of limitations.

The dataset is downloaded from Kaggle at runtime. Users are responsible for complying with the dataset's access, attribution, and usage requirements.

No project license has been selected in the current repository. Add a LICENSE file before publishing.

## 20. Summary

Recommended first run:

~~~bash
uv sync
uv run python train.py --model bigram --iterations 5 --benchmark-steps 1
~~~

Recommended reduced-memory GPT check:

~~~bash
uv run python train.py --model gpt --iterations 1 --benchmark-steps 1 --batch-size 8 --block-size 64
~~~

Recommended timing experiment:

~~~bash
uv run python train.py --model gpt --iterations 5000 --benchmark-steps 10
~~~

Start with a short run, verify dataset loading and device selection, then run a longer experiment while recording the complete configuration and timing information.

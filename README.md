Simplified CVAE Dialogue Model (Interim Prototype)

This repository contains the interim implementation of a Conditional Variational Autoencoder (CVAE) for dialogue response generation, as a prototype for the final project. This model is inspired by the paper "Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders" (Zhao et al., 2017).

The primary objective of this prototype is to demonstrate the fundamental CVAE architecture and integrate the crucial auxiliary loss function designed to stabilize training and ensure diversity.

🌟 Key Features Implemented

The prototype successfully implements three core technical requirements:

Conditional Variational Autoencoder (CVAE) Architecture:

Uses GRU-based Encoder and Decoder networks.

Models a stochastic latent variable $z$ via the reparameterization trick to capture variation in potential responses.

Discourse-Level Diversity Generation:

Includes a function (generate_diversity) to sample the latent variable $z$ multiple times for a single input context, producing distinct and diverse outputs.

Bag-of-Word (BOW) Auxiliary Loss:

Integrates the $\mathbf{L}_{\mathbf{BOW}}$ loss term into the overall objective. This auxiliary loss is critical for mitigating the "vanishing latent variable" problem by forcing the latent vector $z$ to encode the global word content (bag-of-words) of the target response, leading to more stable CVAE training.

⚙️ Model Objective (Total Loss)

The model is trained to optimize a composite variational lower bound ($\mathcal{L}'$):

$$\mathbf{L}_{\text{Total}} = \mathbf{L}_{\text{Recon}} + \beta_{KL} \cdot \mathbf{L}_{\text{KL}} + \beta_{BOW} \cdot \mathbf{L}_{\text{BOW}}$$

Loss Component

Description

Weight

$\mathbf{L}_{\text{Recon}}$

Sequence Cross-Entropy Loss for reconstruction.

$1.0$

$\mathbf{L}_{\text{KL}}$

KL-Divergence Loss, regularizing $z$ towards $\mathcal{N}(0, I)$.

$\beta_{KL} = 0.01$

$\mathbf{L}_{\text{BOW}}$

Binary Cross-Entropy (BCE) on the $\mathbf{MLP_b}$ output, ensuring $z$ encodes global word presence.

$\beta_{BOW} = 0.1$

🚀 Getting Started

Prerequisites

Python 3.8+

PyTorch (Tested on PyTorch 2.2+)

Running the Code

The implementation is contained within a single Python script (intended for execution in a Jupyter or Google Colab environment).

Clone the Repository (or copy the code):

git clone [Your GitHub Repo URL]
cd [repo-name]




Execution: Run the Python script directly. The script will train the model on a small dummy dataset and then print a demonstration of the diversity generation.

python basic_cvae_chatbot_v2.py




📋 Code Structure Snapshot

The core logic is divided into modular PyTorch classes:

Class/Function

Purpose

Key Feature

Encoder(nn.Module)

Encodes the context into $\mu$ and $\log\sigma^2$.

GRU-based.

Decoder(nn.Module)

Generates the response, initialized by latent vector $z$.

GRU-based, conditioned on $z$.

CVAE(nn.Module)

Orchestrates the process. Includes the mlp_bow for auxiliary loss.

Implements the full VAE forward pass.

get_bow_target

Helper function to create the binary target vector for $\mathbf{L}_{\mathbf{BOW}}$.

Handles token filtering.

generate_diversity

Demonstrates the CVAE's core function by sampling $z$ multiple times.

Stochastic sampling.

💡 Example Diversity Output

After 500 epochs, the model shows distinct responses for the same input, illustrating the diversity injected by $z$:

Input Context: 'what are you doing'
Generating 3 diverse responses by sampling Z:
Response 1: just working <PAD> i
Response 2: i am fine i just
Response 3: just working <PAD> hello

Input Context: 'what is your hobby'
Generating 3 diverse responses by sampling Z:
Response 1: i like play tennis
Response 2: i like play i
Response 3: i like play <PAD> i


⏭️ Future Work

This prototype forms the foundation for the final project, which will include:

Integration of Knowledge-Guided CVAE (kgCVAE) by incorporating linguistic features (e.g., Dialog Acts).

Scaling the model to a large, real-world conversational dataset (e.g., Switchboard).

Implementing quantitative evaluation metrics (BLEU, A-bow, Dialog Act Match) for comparison with published benchmarks.

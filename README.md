


# Simplified CVAE Dialogue Model (Interim Prototype)

This repository contains the interim implementation of a **Conditional Variational Autoencoder (CVAE)** for dialogue response generation, serving as a prototype for the final project. This model is inspired by the paper:

**Zhao et al., 2017 – "Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders".**

The primary goal of this prototype is to demonstrate the fundamental CVAE architecture and integrate the crucial auxiliary loss function designed to stabilize training and ensure response diversity.

---

## 🌟 Key Features Implemented

### 1. Conditional Variational Autoencoder (CVAE) Architecture
- GRU-based Encoder and Decoder networks.
- Stochastic latent variable `z` modeled via the **reparameterization trick** to capture variation in potential responses.

### 2. Discourse-Level Diversity Generation
- `generate_diversity` function samples the latent variable `z` multiple times for a single input context, producing **distinct and diverse outputs**.

### 3. Bag-of-Word (BOW) Auxiliary Loss
- Integrates the **BOW loss** into the training objective.
- Helps mitigate the "vanishing latent variable" problem by forcing the latent vector `z` to encode the global word content of the target response.
- Improves training stability and diversity in generated responses.

---

## ⚙️ Model Objective (Total Loss)

The model optimizes a **composite variational lower bound**:

\[
\mathbf{L}_{\text{Total}} = \mathbf{L}_{\text{Recon}} + \beta_{KL} \cdot \mathbf{L}_{\text{KL}} + \beta_{BOW} \cdot \mathbf{L}_{\text{BOW}}
\]

| Loss Component        | Description                                                                 | Weight       |
|----------------------|-----------------------------------------------------------------------------|-------------|
| **L<sub>Recon</sub>** | Sequence Cross-Entropy Loss for reconstruction                               | 1.0         |
| **L<sub>KL</sub>**    | KL-Divergence Loss regularizing `z` towards N(0, I)                          | β<sub>KL</sub> = 0.01 |
| **L<sub>BOW</sub>**   | Binary Cross-Entropy on the MLP output, ensuring `z` encodes global words    | β<sub>BOW</sub> = 0.1 |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.2+  

### Installation
Clone the repository:

```bash
git clone [Your GitHub Repo URL]
cd [repo-name]
````

### Running the Code

The implementation is contained within a single Python script, intended for Jupyter Notebook or Google Colab:

```bash
python basic_cvae_chatbot_v2.py
```

This will train the model on a small dummy dataset and demonstrate **diverse response generation**.

---

## 📋 Code Structure Snapshot

| Class / Function       | Purpose                                                  | Key Feature                         |
| ---------------------- | -------------------------------------------------------- | ----------------------------------- |
| **Encoder(nn.Module)** | Encodes the context into μ and logσ²                     | GRU-based                           |
| **Decoder(nn.Module)** | Generates the response, initialized by latent vector `z` | GRU-based, conditioned on `z`       |
| **CVAE(nn.Module)**    | Orchestrates full VAE forward pass                       | Includes mlp_bow for auxiliary loss |
| **get_bow_target**     | Creates binary target vector for L<sub>BOW</sub>         | Handles token filtering             |
| **generate_diversity** | Samples `z` multiple times to generate diverse outputs   | Stochastic sampling                 |

---

## 💡 Example Diversity Output

After training on the dummy dataset:

```
Input Context: 'what are you doing'
Generating 3 diverse responses by sampling z:
Response 1: just working <PAD> i
Response 2: i am fine i just
Response 3: just working <PAD> hello

Input Context: 'what is your hobby'
Generating 3 diverse responses by sampling z:
Response 1: i like play tennis
Response 2: i like play i
Response 3: i like play <PAD> i
```

This illustrates how `z` introduces **variability and diversity** in generated responses.

---

## ⏭️ Future Work

* Integration of **Knowledge-Guided CVAE (kgCVAE)** with linguistic features (e.g., Dialog Acts).
* Scaling to large, real-world conversational datasets (e.g., Switchboard).
* Quantitative evaluation using metrics like **BLEU**, **A-bow**, and **Dialog Act Match** for benchmarking.

---

## 📖 References

* Zhao, T., Zhao, R., & Eskenazi, M. (2017). *Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders*. [Paper Link](https://arxiv.org/abs/1703.10960)




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

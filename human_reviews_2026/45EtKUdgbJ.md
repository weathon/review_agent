# Forward-Learned Discrete Diffusion: Learning how to noise to denoise faster

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
Discrete diffusion models are a powerful class of generative models that demonstrate strong performance across many domains. However, for efficiency, discrete diffusion typically parameterizes the generative (reverse) process with factorized distributions, which makes it difficult for the model to learn a target process in a small number of steps and necessitates a long, computationally expensive sampling procedure. To reduce the gap between the target and model distributions and enable few-step generation, we introduce a learnable noising (forward) process for discrete diffusion. Instead of fixing a Markovian forward chain, we adopt a non-Markovian formulation and introduce learnable marginal and posterior distributions. This allows the generative process to remain factorized while matching the target defined by the noising process. We train all parameters end-to-end under the standard variational objective.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper learns the forward noising process in a non-Markovian way so that the training targets match what a factorized reverse model can learn well in just a few steps. It keeps the reverse sampler unchanged. Via learned forward, the paper achieves much better quality–inference latency trade-offs. Across text (ROCStories) and molecular (QM9, ZINC250k) data, the paper with only $T=10$ reverse steps delivers quality close to its $T=100$ counterpart—unlike conventional discrete diffusion.

### Strengths
FLDD’s main idea is both theoretically sound and practically useful: learn the forward noising process in a manner so that the task for the reverse process becomes easier. In addition, the algorithm uses the standard variational objective and does not change the reverse sampler. The posterior is implemented with a cheap Maximum Coupling transport between consecutive marginals, which keeps training tractable while letting each coordinate depend on the whole input. Empirically, the paper shows few-step generation benefits illustrating a better quality–latency trade-off than conventional discrete diffusion.

### Weaknesses
FLDD adds a learned forward network, increasing training compute relative to standard discrete diffusion. The paper does not report quality vs total training and inference budget (e.g., GPU-hours or FLOPs), leaving unclear whether the few-step gains persist under fixed total compute budget.

### Questions
The paper does not compare REINFORCE against other standard gradient estimators for discrete latents. It does not discuss control variates such as Rao–Blackwellization tricks. These would reveal which knobs actually make training computationally stable/cheap. Could you report some of these results?

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes forward-learned discrete diffusion (FLDD). It’s a discrete diffusion with a learnable forward process. The idea is to keep the generative process factorized while matching the target, so it can work well for small numbers of diffusion steps. Experiments show that the performance of FLDD drops a little when the number of diffusion steps goes down from 100 to 10.

### Strengths
- Novel approach for training discrete diffusion models

### Weaknesses
- FLDD with T=100 doesn’t outperform some previous work. The authors argue that FLDD performs better for small Ts. However, there are methods to accelerate sampling of discrete diffusion models, e.g. distillation. For small-T experiments, FLDD needs to be compared with methods focusing on accelerated sampling

### Questions
- What's the batch size used in training and how does it affect the performance?
- What are the additional costs of training, in terms of memory (chips) and duration (time)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Forward-Learned Discrete Diffusion (FLDD), a new framework for discrete diffusion models aimed at improving sampling efficiency and enabling few-step generation. The core idea is to replace the standard fixed, Markovian noising process with a learnable, non-Markovian forward process. By co-training this flexible forward process alongside the reverse (denoising) process, the authors argue that the model can learn an easier-to-invert corruption path. This, in turn, allows a simple, factorized reverse sampler to generate high-quality samples in a significantly reduced number of steps. The method is trained end-to-end using the standard variational objective and is evaluated on synthetic data, binarized MNIST, text generation (ROCStories), and molecular generation (QM9, ZINC250k).

### Strengths
1. The paper does a good job of grounding the proposed method in the established variational inference framework for diffusion models. The formulation for end-to-end training of both the forward and reverse processes while preserving the standard variational objective is principled and clearly presented.
2. The paper is generally well-written and easy to follow.

### Weaknesses
1. The primary qualitative results are presented on a 2D Gaussian mixture and binarized MNIST. These are considered solved or overly simplistic problems in the current deep generative modeling landscape. Demonstrating success on these toy tasks provides very little signal about the method's effectiveness on complex, high-dimensional discrete data that is of interest to the community.
2. A significant weakness of this paper is its failure to demonstrate a clear performance advantage over existing, strong baselines. The paper's own results on the ROCStories dataset (Table 1) show that SEDD (Lou et al., 2023), a recent and relevant discrete diffusion model, achieves a higher MAUVE score (0.598 vs. 0.538) and a better (lower) PPL than the proposed FLDD, even when FLDD uses a large number of steps (T=100). This directly undermines the paper's central claims regarding improved sampling efficiency and achieving a better quality-latency trade-off. Given that a method from 2023 already shows superior performance on the same benchmark, the practical contribution of the proposed, more complex FLDD framework is called into question.
3. The paper fails to cite, discuss, or compare against the seminal work: LLaDA[1], a landmark paper that demonstrated how to successfully apply diffusion models to high-quality text generation and is arguably the most famous work in this specific area.

[1] Nie, Shen, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. "Large language diffusion models." arXiv preprint arXiv:2502.09992 (2025).

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Forward-Learned Discrete Diffusion, a method designed to accelerate the reverse process while supporting end-to-end training.

Unlike previous works, FLDD adopts a non-Markovian formulation with learnable marginal and posterior distributions, which allows the generative process to remain factorized while matching the target distribution defined by the noising process. The proposed method enables a better alignment between the forward and reverse processes even with a small number of steps, thereby improving its generation performances.

### Strengths
- A novel method that integrates insights from SS-DDPM and DPS, achieving excellent performance in few-step generation under specific scenarios.

- The proposed method significantly reduces the requirement for the number of timesteps in the backward process, which is a notable advantage for practical applications.

- This paper is well-structured and logically coherent, with smooth transitions between sections. The examples are appropriately designed, making the core ideas easy to follow.

### Weaknesses
- The definition of "end-to-end" in the context of FLDD remains unclear. It is necessary to clarify whether this refers to training without separate stages or other specific characteristics.

- In Section 4.2, FLDD exhibits poor performance on the ROCStories dataset, but the authors' explanation for this phenomenon lacks sufficient persuasiveness and requires further elaboration.

### Questions
1. In the forward process, what are the key differences between FLDD and Star-shaped Diffusion? Could the authors elaborate on the unique design choices of FLDD in this regard?

2. Compared with continuous diffusion models that adopt similar training strategies on continuous datasets (e.g., MNIST), what specific advantages does FLDD offer? For example, in terms of generation quality, training efficiency, or adaptability to data characteristics.

3. In the experimental results (Table 1), FLDD performs poorly on real-world data, particularly with a PPL of 60, which indicates an inability to generate semantically meaningful sentences. Could the authors provide a more specific analysis of the underlying reasons? For instance, whether it is due to mismatches between the forward process design and real-world data distributions or limitations in the learnable marginal/posterior functions.

4. While FLDD demonstrates strong performance in few-step diffusion, what are the precise time and computational costs compared to existing methods? A smaller number of steps does not always translate to lower overhead, so quantitative comparisons would help verify its efficiency advantages.

### Soundness
3

### Presentation
3

### Contribution
3

# AGDC: Autoregressive Generation of Variable-Length Sequences with Joint Discrete and Continuous Spaces

- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Transformer-based autoregressive models excel in data generation but are inherently constrained by their reliance on discretized tokens, which limits their ability to represent continuous values with high precision. We analyze the scalability limitations of existing discretization-based approaches for generating hybrid discrete-continuous sequences, particularly in high-precision domains such as semiconductor circuit layout designs, where precision loss can lead to functional failure. To address the challenge, we propose **AGDC**, a novel unified framework that *jointly models discrete and continuous values for variable-length sequences*. AGDC employs a hybrid approach that combines categorical prediction for discrete values with diffusion-based modeling for continuous values, incorporating two key technical components: an end-of-sequence (EOS) logit adjustment mechanism that uses an MLP to dynamically adjust EOS token logits based on sequence context, and a length regularization term integrated into the loss function. Additionally, we present **ContLayNet**, a large-scale benchmark comprising 334K high-precision semiconductor layout samples with specialized evaluation metrics that capture functional correctness where precision errors significantly impact performance. Experiments on semiconductor layouts (ContLayNet), graphic layouts, and SVGs demonstrate AGDC's superior performance in generating high-fidelity hybrid vector representations compared to discretization-based and fixed-schema baselines, achieving scalable high-precision generation across diverse domains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes AGDC, a novel autoregressive framework for generating variable-length sequences that contain both discrete identifiers and high-precision continuous values. The core problem it addresses is the precision loss inherent in standard tokenization-based models, which is detrimental in domains like semiconductor design. AGDC tackles this by jointly modeling the two data types: using categorical prediction for discrete values and a conditional diffusion model for continuous vectors, all within a unified autoregressive structure. The authors also introduce ContLayNet, a large-scale benchmark of high-precision semiconductor layouts, and a corresponding set of Design Rule Check (DRC) metrics to evaluate functional correctness.

### Strengths
1. The paper identifies a critical and practical limitation of discretization in generative models. The proposed hybrid approach of combining autoregressive prediction with an inner loop of diffusion sampling is an elegant and novel method to preserve continuous precision.
2. The introduction of the ContLayNet benchmark and its specialized DRC-based evaluation metrics is a major contribution. This provides a much-needed and challenging testbed for a problem space that has been underserved.

### Weaknesses
1. A major concern is the inference speed. The model must run an iterative diffusion sampling process (which is itself multi-step) for every single autoregressive step. This seems computationally prohibitive and likely orders of magnitude slower than discretization-based autoregressive models, potentially limiting its practical utility.
2. It remains unclear what this paper contributes to the machine learning (generative models) community.

### Questions
Could the authors provide a more direct comparison of inference time between AGDC, LT, and DLT for generating, for example, 100 samples? How many sampling steps does the diffusion model use per autoregressive step, and was acceleration (e.g., DDIM) explored?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces AGDC, an autoregressive diffusion model that jointly models discrete and continuous values for variable-length sequences. It then presents ContLayNet, a large-scale benchmark comprising 334K high-precision semiconductor layout samples. Experiments on semiconductor layouts, graphic layouts, and SVGs demonstrate AGDC's superior generation performance.

### Strengths
**S1.** The proposed methodology is conceptually sound and neat.

**S2.** The paper introduces ContLayNet, a real-world large-scale dataset of semiconductor layout samples.

**S3.** Empirical studies (quantitative and qualitative) demonstrate the effectiveness of the proposed approach, with ablation studies provided for various design choices.

**S4.** The paper is overall easy to follow.

### Weaknesses
**W1.** The baselines are limited. Even though some generative models were not designed towards layout generation in particular, they may still be adaptable to this scenario.

**W2.** The paper does not discuss existing papers that combine autoregressive models and diffusion models. E.g.,

- Chen et al. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion. NeurIPS 2024.

- Zhao et al. Pard: Permutation-Invariant Autoregressive Diffusion for Graph Generation. NeurIPS 2024.

- Li et al. LayerDAG: A Layerwise Autoregressive Diffusion Model for Directed Acyclic Graph Generation. ICLR 2025.

**W3.** From Figure 2, different atomic units can have continuous values of different lengths. Meanwhile, equation 7 assumes each atomic unit have exactly a single discrete value and fixed-length continuous values. This seems to suggest a limitation of AGDC in flexibility and generalizability.

### Questions
N.A.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper presents AGDC, a novel autoregressive framework that jointly models discrete and continuous values within variable-length sequences. It predicts discrete values through categorical prediction and continuous values using diffusion-based probabilistic models. The paper presents experiments across semiconductor layouts (with a new large-scale benchmark), graphic layouts, and text-to-SVG synthesis. The results shows that AGDC outperforms existing discretization-based and fixed-schema methods.

### Strengths
* Joint models of discrete and continuous values in autoregressive model is useful for modeling and generating complex data. 
* The paper proposes a new benchmark for circuit layout. 
* The experiments are showing that the encoding and generation work on multiple problems, indicating the generality of the proposed approach. Out of the three parts of the experiment, the circuit layout is the most interesting case study.

### Weaknesses
* The paper is light on theoretical contributions. The technical approach seems as a combination of known models, which translates to explaining “how” the approach works, leaving “why” out. 
* The motivation behind EOS logit adjustment was not clear. 
* The results on the chip layout problem seem  notable. The table shows improvement in several existing metrics in the domain. However, these metrics seem very problem specific and their motivation is not well explained for the general audience. The qualitative notion of “more balanced and well-distributed layers” is hard to justify by representative images only; a quantitative analysis is likely necessary. 
* Additionally, it is difficult to grasp the evaluation setup and metrics for this study without going to the appendix. The evaluation metrics for circuit layout generation should be described in more detail in the main body of the paper, by moving some text from the appendix. 
* The applications on other two case studies in the evaluation (graphics layout and text-to-svg) seem less impactful.

### Questions
No questions. My overall suggestion is that the authors revise the paper to emphasize the merits of their algorithmic contribution and provide more background for the readers who are not circuit design experts.

### Soundness
2

### Presentation
1

### Contribution
2

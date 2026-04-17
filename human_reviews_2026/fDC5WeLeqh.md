# AWM: Accurate Weight-Matrix Fingerprint for Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Protecting the intellectual property of large language models (LLMs) is crucial, given the substantial resources required for their training. Consequently, there is an urgent need for both model owners and third parties to determine whether a suspect LLM is trained from scratch or derived from an existing base model. However, the intensive post-training processes that models typically undergo—such as supervised fine-tuning, extensive continued pretraining, reinforcement learning, multi-modal extension, pruning, and upcycling—pose significant challenges to reliable identification. In this work, we propose a training-free fingerprinting method based on weight matrices. We leverage the Linear Assignment Problem (LAP) and an unbiased Centered Kernel Alignment (CKA) similarity to neutralize the effects of parameter manipulations, yielding a highly robust and high-fidelity similarity metric. On a comprehensive testbed of 60 positive and 90 negative model pairs, our method demonstrates exceptional robustness against all six aforementioned post-training categories while exhibiting a near-zero risk of false positives. By achieving perfect scores on all classification metrics, our approach establishes a strong basis for reliable model lineage verification. Moreover, the entire computation completes within 30s on an NVIDIA 3090 GPU.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose AWM, a method based on centered kernel alignment to fingerprint and compare LLMs. It is more accurate than competing methods such as HuRef and REEF on a large set of different LLMs of different families. However, it requires knowledge of all the weights of the LLMs to be compared.

### Strengths
- Empirically the new proposed algorithm is quite strong, improving upon previous methods such as HuRef and REEF in distinguishing between related and unrelated LLMs on a fairly large set of LLMs tested.  

- The algorithm works directly with weight matrices and only uses centered kernel alignment computation and the Hungarian algorithm for finding permutations, and is therefore very efficient. 

- The authors give a detailed list of potential weight manipulations on different parts of the transformer architecture by an attacker in Section 4 and explain how this guides the design of their algorithm.

### Weaknesses
- This fingerprinting method requires access to the weights of the models to be compared, which isn't always possible unless both models are open-sourced. This limits the applicability of the method compared to other fingerprinting approaches. 

- In Section 4 the authors discuss how an attacker might manipulate the weights to evade detection. However, these manipulations are never evaluated against the algorithm in the experiments. This creates a gap in the argument of the paper.

### Questions
- Have the authors consider how to apply their method to other forms of manipulations/infringement such as model distillation?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes AWM, a training-free, white-box fingerprint for LLMs that combines LAP-based layer matching with an unbiased CKA similarity to measure model lineage directly from weights. The experimental suite spans six common post-training transformations (SFT, continued pretraining, RL-based alignment, multimodal extension, pruning, and MoE upcycling) over 60 positive and 90 negative pairs. The method is computationally efficient (single-GPU, sub-minute per pair) and requires no instrumenting or watermarking of the target model.

### Strengths
1.The method is minimalist and targeted, focusing on embeddings and Q/K, which makes LAP+linear CKA implementation straightforward and efficient (single-GPU, sub-minute).
2.Dynamic layer assignment via LAP addresses the weakness of fixed layer-to-layer comparisons and tolerates moderate architectural edits.
3.It requires no data collection, retraining, or watermark insertion, so it does not degrade model quality and is easy to operationalize.
4.Empirically, the approach cleanly separates derived from independent models (AUC=1.0, TPR@1%FPR=100%), implying near-zero false-positive risk.

### Weaknesses
1.The pipeline relies on overlaps in token embeddings and Q/K weights; if the suspect model re-trains the tokenizer or heavily replaces early blocks, the LAP matching can become unstable and |Z| may drop.
2.In cases of knowledge distillation, AWM will likely fail because cross-model weights exhibit near-zero statistical correlation despite functional similarity. Notably, recent alleged infringement incidents (e.g., API-based distillation claims) fall into this category.
3.Although LAP provides some structural elasticity, it presupposes roughly comparable sets of alignable matrices. If the suspect model adds/removes many layers or alters layer roles, matching can break or produce erroneous alignments. While permutation invariance should not affect CKA in principle, aggressive layer reordering can still mislead the LAP stage and degrade end-to-end performance.
4.Since the method is simple and single, the attacker can try to perform some lossless transformation on the AWM model to reduce the CKA similarity but maintain the model output. This is not difficult to achieve.
5.The approach evaluates one-to-one similarity and does not address models fused from multiple sources (e.g., model 𝐶 partly derived from 𝐴 and 𝐵).
6.The only comparison methods are HuRef and REEF. Why are there no comparisons with baselines used in REEF, such as PCS and ICS?
7.The related work section recommends adding some black box-based methods. Currently, piracy through APIs is also very common.

### Questions
1.The first step of the method relies on a shared vocabulary. How should we handle cases where the suspect model retrains or replaces its tokenizer/vocabulary?
2.Does the author have any insights on piracy cases involving knowledge distillation? Could lightweight black-box probes be combined with AWM to first flag potential distillation cases?
3.When the architecture undergoes significant depth/width changes, cross-layer sharing, or reordering of blocks, can you constrain the LAP with structural priors to prevent misalignment?
4.Since the method is very simple, it becomes easier to attack. How can this issue be addressed?
5.Have you considered approaches for mixed-origin cases, for example, solving multiple partial LAPs and reporting a mixed (or mixture) score?
6.Release an evaluation sheet (model names/versions, checkpoints, tokenizer specs, training corpus tags), and precise CKA/LAP implementation details (module selection, sampling of parameter blocks, normalization, kernel choices).
7.It is recommended to add some comparison methods and, if possible, add explicit ablation experiments

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
This paper presents a method for LLM intellectual property (IP) protection, aiming to overcome the vulnerabilities of current fingerprinting schemes, which are often not robust against intensive post-training and malicious weight manipulations. The authors introduce AWM (Accurate Weight-Matrix Fingerprint), a novel, training-free, intrinsic weight-based fingerprinting scheme designed to be highly robust to such modifications. The method's core mechanism leverages the Linear Assignment Problem (LAP) and an unbiased Centered Kernel Alignment (CKA) similarity metric. This process first extracts permutation and signature transformations from the word embedding matrices via LAP , and then uses the CKA-based metric to robustly compare the Q and K matrices across layers, a design intended to neutralize the effects of parameter manipulations like orthogonal transformations. Experimental results demonstrate that AWM achieves perfect classification scores and shows exceptional robustness.

### Strengths
- The idea of this paper is novel.
- This paper presents promising results.
- This paper provides a comprehensive theoretical analysis.

### Weaknesses
- Dependency on Word Embeddings: The method relies heavily on the word embedding matrix. An attacker could potentially defeat the detection by freezing all other layers and only replacing or retraining the embedding layer, which may cause the method to fail.

- Limited Detection Scope: The method's scope is restricted to $W_Q$ and $W_K$ matrices. An attacker could steal other components (e.g., implanting stolen FFN blocks into an MoE model). This form of partial theft would likely go undetected.

- Other suggestions for improving the writing:
  - The theoretical analysis in Section 4 is somewhat confusing.  It may be better for the authors to first introduce the basic ideas and conclusions at the beginning of the section to improve readability.

  - The Related Work section on fingerprinting is limited, focusing mostly on traditional classification models and only two LLM fingerprinting papers. It may be better for the authors to conduct a more exhaustive survey to enrich this section.

### Questions
Please refer to the Weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a mechanism to "fingerprint" an LLM. This is to find out if an LLM is trained from scratch or is a finetuned/derived version of a different LLM. The paper evaluates their methods on multiple models, different post training techniques and they achieve perfect scores in all classification metrics.

### Strengths
The computation completes within 30 seconds on a single NVIDIA 3090 GPU.

The method does not require any additional training and does not impact the LLM's performance (as other watermarking approaches)

The evaluation is very comprehensive across multiple llama models and multiple offspring variations where models are post trained with sft, continued pretraining, RL. 

They also present a false positive evaluation with 90 known to be unrelated pairs, where they show other approach like REEF shows significant false positives mostly for models that use same training data sources.

### Weaknesses
Maybe lack of experiments with larger models and other model architectures like MoE.

### Questions
Any thoughts on how this will work with MoE models?

Any thoughts on how this will work if the post training is done with LoRA methods and or adding new layers of randomly initialized parameters.

### Soundness
3

### Presentation
4

### Contribution
3

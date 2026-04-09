# Highly Relevant Papers for Fine-Tuning Research

Based on systematic search through 368 human review files for papers addressing fine-tuning, OOD generalization, memorization, mechanistic insights, and related topics, the following papers have been identified as highly relevant.

## 1. Supervised Fine-Tuning & Fine-Tuning Dynamics

### Review File: 204sPiwBbB.md
**Paper:** Training with Annotations (TWA) - Machine Translation fine-tuning with MQM error annotations

**Summary:** Proposes fine-tuning machine translation systems using span-level error annotations from quality estimation. The method uses unlikelihood loss to discourage generating tokens in error spans. Compared against SFT and DPO baselines.

**Key Weaknesses from Reviews:**
- Limited novelty: "The main novelty of this work is utilizing additional annotations to improve translation systems, which is not surprising. Otherwise, the proposed unlikelihood training is straightforward."
- Weak experimental setup: Only evaluated on two translation directions on one dataset; raises questions about generalizability
- Motivation unclear: Why focus on machine translation when the technique could apply to other tasks?
- Fine-tuning gaps underexplored: No clear comparison of when SFT improves results vs. other factors

---

### Review File: v4MTnPiYXY.md
**Paper:** Q-SFT - Integrating Q-learning with SFT for LLMs in multi-turn RL settings

**Summary:** Proposes using token logits from pretraining as value functions in Q-learning instead of learning new value heads. Provides theoretical analysis and experiments on text games and robotic tasks.

**Key Weaknesses from Reviews:**
- Unclear problem formulation: Paper conflates single-token actions with multi-turn dialogue, making the "multi-turn" aspect confusing
- Simplified experimental setup: Tasks use small vocabularies and well-defined reward functions not translating to practical problems
- Limited theoretical guarantees: If π_β is small, resulting policy can be overly conservative
- Infrastructure overhead: Requires double forward passes (original + fine-tuned model) at inference time

---

### Review File: koza5fePTs.md
**Paper:** Benchmark study of LLM planning with fine-tuning on optimal solutions

**Summary:** Benchmarks planning capabilities of LLMs on PDDL and travel planning domains, analyzes impact of fine-tuning on optimal planning solutions with few-shot examples.

**Key Weaknesses from Reviews:**
- Overfitting issues not addressed: Near-perfect accuracy in SFT only when train/eval data are similar - "does not lead to any conclusions regarding generalization abilities of LLMs"
- Evaluation limitations: Only tested on limited tasks and single dataset (RedPajama)
- OOD generalization unclear: How does performance degrade when training distribution differs from test?
- Limited novelty: Mostly combines existing datasets without generating substantially new data

---

## 2. Out-of-Distribution Generalization & Domain Shift

### Review File: EXnDAXyVxw.md
**Paper:** Quantization-aware training (QT-DoG) for domain generalization

**Summary:** Uses quantization noise during training as implicit regularizer to guide optimization toward flatter minima. Ensemble of quantized models combined into "model soup" for improved OOD generalization.

**Key Weaknesses from Reviews:**
- Limited novelty: "Their main idea is to replace standard training in ERM with the existing quantized technique... hardly a new practice"
- Weak theoretical justification: Lacks formal definition of "flatter minima"; no clear theoretical evidence that quantization leads to flatness
- Inconsistent results: Different quantization methods have varying effects; some worse than baseline
- Limited experimental scope: Only image classification datasets; WILDS and larger-scale evaluations missing
- Unclear experimental setup: How are hyperparameters like step size determined? Is quantization only applied at model save?

---

### Review File: cu2CT2VAvs.md
**Paper:** State Collapse in Mamba-2 - Length generalization failures in RNN models

**Summary:** Identifies "state collapse" phenomenon in RNN-based models on long contexts, proposes training-free and training-based mitigation methods.

**Key Weaknesses from Reviews:**
- Vague problem definition: "State Collapse" lacks formal definition; conflated with poor length generalization performance
- Limited evaluation: Only tested on single model (Mamba-2) and single dataset (RedPajama); generalization unclear
- Training-free methods have limitations: Methods 1 & 3 induce forgetting; Method 2 degrades performance significantly and breaks parallelization
- Evaluation limited to language modeling: No evaluation on long-context tasks like passkey retrieval or ∞-BENCH
- Claims disconnect from evidence: Analysis of causes and proposed solutions appear disconnected from observed phenomenon

---

## 3. Memorization in Neural Networks

### Review File: OZVTqoli2N.md
**Paper:** Memorization and forgetting in continual learning with adaptive networks

**Summary:** Studies catastrophic forgetting in continual learning through analysis of model capacity and state dynamics. Proposes that memorization requires overparameterization.

**Key Weaknesses from Reviews:**
- Lack of clear definitions: "State collapse" and related phenomena not formally defined
- Limited scope: Experiments only on MNIST and Fashion-MNIST with 2-3 tasks
- Poor baseline comparisons: Only compared to EWC and MAS; missing comparisons to recent methods
- Scalability concerns: Method doesn't scale to realistic datasets (CIFAR, ImageNet)
- Task diversity missing: Only grayscale 28x28 images; unclear how methods perform on complex RGB data

---

## 4. Module-Level & Mechanistic Analysis

### Review File: SI6zocV2SS.md
**Paper:** Incremental Task Alignment (ITA) - Model merging for incremental learning

**Summary:** Proposes theoretical analysis of multitask loss through second-order Taylor expansion. Shows that if all individual models perform well on all tasks, merged model performs well. Introduces ITA method with EWC regularization.

**Key Weaknesses from Reviews:**
- Weak theoretical justification: Regularizing task vectors toward base model as proxy for good performance on out-of-distribution data is "somewhat of a big leap"
- Limited scope: "Only tested on image classification setting; unclear how methods generalize to language modeling or other domains"
- Missing baselines: No comparison with other model merging techniques like ColD Fusion, TIES, or recent MoE merging methods
- Experimental limitations: All datasets (IN-R, C-100, CUB) are ID relative to ImageNet pretraining; simple fine-tuning might suffice
- Unclear decentralized training: Claims decentralized execution but sequential FIM updates seem problematic

---

## 5. Parameter-Efficient & Selective Fine-Tuning

### Review File: 5s1qpjrNvZ.md
**Paper:** Shallow Safety Alignment in LLMs - Defense against fine-tuning attacks

**Summary:** Identifies that safety alignment in LLMs only affects first few output tokens; proposes data augmentation (partial refusals) and constrained-SFT methods to deepen alignment.

**Key Weaknesses from Reviews:**
- Weak baseline comparisons: Missing comparisons to existing defenses like circuit breaking or prompt-level defenses
- Unclear threat model: Vulnerabilities primarily relevant to open-weight models; unclear how this applies to black-box APIs
- Limited analysis: True reasons why data augmentation works need deeper discussion; unclear if defense handles all attack variants
- Non-zero residual vulnerability: After mitigation, ASR still substantial; paper doesn't explain remaining vulnerability sources
- Incomplete evaluation: Methods not compared against other selective/parameter-efficient alignment approaches

---

## 6. Evaluation on Limited Benchmarks/Tasks

### Review File: 1Qpt43cqhg.md (Q-SFT, see above)
- Only 2 text-based tasks and robotic manipulation task
- Small vocabulary/action spaces not representative of real problems
- Missing comparisons to baseline SFT on same setup

### Review File: ZHTYtXijEn.md
**Paper:** Adaptive network growth with Hebbian learning for continual learning (DIRAD/PREVAL)

**Summary:** Proposes dynamic network expansion triggered by saturated gradients; uses Hebbian learning for task detection in continual learning settings.

**Key Weaknesses from Reviews:**
- Only MNIST and Fashion-MNIST evaluation: "The experimental setup is too simple to evaluate the algorithm properly"
- Poor metric selection: Reports only average accuracy; missing backward transfer, forward transfer metrics that characterize forgetting
- Scalability not demonstrated: Method becomes computationally prohibitive on realistic datasets
- Missing comparisons: No comparison to recent continual learning methods; only EWC baseline
- Unclear mechanism: Why does Hebbian learning specifically identify task-relevant neurons? Motivation unclear

---

## 7. Mechanistic Insights & Understanding Why Methods Work

### Review File: 6Mxhg9PtDE.md
**Paper:** Shallow Safety Alignment (see above - 5s1qpjrNvZ.md)

**Additional Insights:**
- Authors try to explain depth of alignment through KL divergence analysis but evidence is limited
- Data augmentation method's success mechanism unclear: Is it learning to give refusals after harmful content, or something else?
- Proposed solutions appear partially effective but underlying reasons for residual vulnerabilities not explored

---

### Review File: gcQAQfXv6.md (Not available - file doesn't exist)

### Review File: 1H90Gb9rJ9.md
**Paper:** Boolean Network Optimization via Neural Networks

**Summary:** Optimizes 2-layer NN representations of Boolean functions through monomial reduction and NPN classification. Uses convex relaxation for lossless compression.

**Key Weaknesses from Reviews:**
- Limited scope: Only 2-layer NN optimization, too local for broader insights
- Scalability concerns: NP-hard problem; relaxation to L1 may be suboptimal; unclear how method scales
- Limited applicability: Specific to Boolean networks; generalization to other neural network types unclear
- Narrow experimental scope: Only Boolean circuits evaluated; no other domains tested
- Accessibility: Highly specialized; limited relevance to broader ML community

---

## Summary of Key Patterns

### Common Weaknesses Across Fine-Tuning Papers:

1. **Limited Experimental Scope**
   - Most papers evaluated on 1-2 datasets only
   - Limited diversity in tasks/domains
   - OOD evaluation often missing

2. **Unclear Mechanisms**
   - Why methods work not well explained
   - Gap between theory (if provided) and practice
   - Hyperparameter choices arbitrary or unexplained

3. **Overfitting Issues**
   - Near-perfect accuracy only when train ≈ test
   - Limited analysis of what causes improvements vs. data quality
   - Generalization to new task distributions unexplored

4. **Baseline Comparisons**
   - Missing comparisons to strong recent baselines
   - Often only compared to simple methods (EWC, SFT)
   - Parameter-efficient fine-tuning methods underexplored

5. **Evaluation Limitations**
   - Limited number of tasks/benchmarks used
   - Missing important metrics (backward/forward transfer, memorization measures)
   - Single dataset evaluation common

### Opportunities for Fine-Tuning Research:

- **Mechanistic understanding**: Deeper analysis of why selective fine-tuning works better
- **OOD evaluation**: Systematic evaluation across multiple OOD datasets with clear distribution shifts
- **Limited benchmark handling**: Understanding how fine-tuning generalizes when trained on subset of data
- **Module-level analysis**: Systematic analysis of which modules/layers change during fine-tuning
- **Memorization analysis**: Understanding when models memorize vs. generalize during fine-tuning

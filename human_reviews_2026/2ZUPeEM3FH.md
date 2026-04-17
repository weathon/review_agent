# Learn-to-Distance: Distance Learning for Detecting LLM-Generated Text

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Modern large language models (LLMs) such as GPT, Claude, and Gemini have transformed the way we learn, work, and communicate. Yet, their ability to produce highly human-like text raises serious concerns about misinformation and academic integrity, making it an urgent need for reliable algorithms to detect LLM-generated content. In this paper, we start by presenting a geometric approach to demystify rewrite-based detection algorithms, revealing their underlying rationale and demonstrating their generalization ability. Building on this insight, we introduce a novel rewrite-based detection algorithm that adaptively learns the distance between the original and rewritten text. Theoretically, we demonstrate that employing an adaptively learned distance function is more effective for detection than using a fixed distance. Empirically, we conduct extensive experiments with over 100 settings, and find that our approach demonstrates superior performance over baseline algorithms in the majority of scenarios. In particular, it achieves relative improvements from 54.3% to 75.4% over the strongest baseline across different target LLMs (e.g., GPT, Claude, and Gemini). A python implementation of our proposal is publicly available at https://github.com/Mamba413/L2D.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new rewrite-based detection algorithm for the detection problem of LLM generated text. The author first analyzes the principle and robustness of rewriting detection algorithms from a geometric perspective, and then proposes a new method for adaptive learning text rewriting distance.

### Strengths
1. Clear theoretical insight:
The geometric analysis (Propositions 1 & 2) provides an explanation for why rewrite-based approaches remain robust in unseen prompt scenarios.

2. Extensive experimentation:
The experiments are comprehensive — 27 settings across 3 datasets, 3 prompt types, and multiple target models (Claude-3.5, GPT-4o, Gemini-2.5) — showing consistent improvements.

3. Adversarial robustness:
The evaluation under paraphrasing and decoherence attacks demonstrates the proposed method’s strong robustness.

### Weaknesses
1. Questionable novelty of “Prompt-robust detection”:
The notion of prompt-robustness largely overlaps with prior work on generalization detection, where detectors are expected to perform under unknown prompts or semantic variations. Many existing zero-shot methods (e.g., DetectGPT, Fast-DetectGPT) are already prompt-agnostic, so it’s unclear whether the proposed notion is truly new.

2. Fairness of comparison (Figure 2):
The proposed method requires training, whereas Fast-DetectGPT is a zero-shot detector. Comparing them directly may be unfair since the proposed approach has access to additional supervision and compute.

3. Lack of implementation details:
The paper is vague about key training aspects, such as the architecture of the distance module, learning rate, optimization steps, dataset scale, or fine-tuning hyperparameters. This limits reproducibility.

4. High computational cost:
The method involves multiple LLM queries — one for rewriting and another for distance computation — which can be prohibitively expensive in large-scale or real-time scenarios.

5. Misleading improvement reporting:
The “relative gain” metric (e.g., “average improvements of 45.3%–62.5%”) can exaggerate the results. For example, when AUROC improves from 0.951 to 0.987, the absolute gain is minor (~0.036), yet the table reports a 72.9% “relative gain.” Reporting absolute AUC improvements would be more transparent and standard.

### Questions
1. What are the detailed training settings (loss function, optimizer, learning rate, batch size, training steps)?

2. Does Figure 2 involve an unfair comparison between a trained detector and zero-shot baselines?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a prompt-robust detection framework for LLM-generated text based on a geometric understanding of rewrite-based methods. The authors show that human-written texts exhibit larger reconstruction errors than LLM-generated ones, and that this difference remains stable even under unseen prompts. Building on this insight, they introduce a method that learns an adaptive distance function between an input text and its rewritten version using a fine-tuned language model, rather than relying on fixed distances such as BLEU or BERTScore. Experiments across many datasets and several LLMs demonstrate improvement over baselines.

### Strengths
1. The paper is well organized and easy to follow
2. Important and timely topic
3. prompt robustness is very important for these trained llm-text classifiers.

### Weaknesses
1. My major concern lies in the geometric assumptions underpinning the theory, which are elegant but often unrealistic in practice. The framework assumes that LLM-generated text is a linear projection of human-written text onto an “LLM subspace” (Assumption 2) and that rewriting behaves equivalently on human and LLM-like inputs (Assumption 3). However, real-world text generation is highly nonlinear and context-dependent, and rewriting can amplify stylistic or semantic differences depending on prompts, temperature, and decoding randomness. The additive noise model  R(x) = \Pi_M(x) + e  also oversimplifies rewriting dynamics, as  e may not be small or confined to the same subspace. Consequently, the theoretical claims (Propositions 1–2) serve more as conceptual heuristics than as formal guarantees of robustness.

2. From a practical standpoint, the proposed approach faces several deployment challenges. First, it requires direct access to the target LLM for rewriting, which may not always be feasible in real-world . This dependency makes the method less suitable for general-purpose detection across unknown or evolving models. Second, the approach involves fine-tuning a distance model to adapt to each target LLM’s text distribution. This fine-tuning step also risks overfitting when the available training data are limited in domain diversity or prompt variety, potentially reducing generalization to unseen contexts. Finally, the paper offers little analysis on scalability and runtime efficiency, especially for long documents or large-scale batch detection scenarios where repeated rewriting and distance computation could become prohibitively expensive. These constraints limit the method’s immediate practicality despite its strong empirical performance.

3. In my understanding, your method requires fine-tuning the detector for each target LLM. Given this, is it fair to directly compare your approach with previous rewrite-based or other detection methods that operate in a zero-shot or non–fine-tuned setting? How do you ensure that the comparison across methods remains fair and consistent?

In general, I would consider this a borderline paper.

### Questions
1. The theoretical framework relies on strong geometric assumptions (e.g., LLM text as a projection of human text). Have you empirically verified or tested how well these assumptions hold in practice?

2. How well does the learned distance function transfer to new or unseen LLMs without re–fine-tuning?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper addresses the prompt-robust detection of LLM-generated text, a realistic and challenging setting where the prompts used to produce the text are unobserved. The authors propose a geometric interpretation of rewrite-based detection methods, proving why reconstruction errors differ between human and machine text (Proposition 1) and remain robust to prompt-induced distribution shifts (Proposition 2). Building on this, they introduce a machine-learning-based rewrite detector that learns a distance function via fine-tuning a language model, rather than relying on fixed metrics like BERTScore or Levenshtein distance. Extensive experiments across 24 datasets, 7 LLMs, and 3 prompt types show relative AUC improvements of 45–62% over the strongest baselines, with strong resistance to adversarial paraphrasing and decoherence attacks.

### Strengths
- Introduces a clear geometric interpretation of rewrite-based detectors, offering theoretical insight that previous empirical works lacked.
- The learned-distance formulation bridges theory and implementation elegantly, and the optimization is compatible with LoRA-style fine-tuning for scalability.
- Evaluated on > 100 settings (24 datasets, 7 LLMs, 3 prompt types, 2 attacks) with consistent superiority over 11 baselines.
- Addresses the realistic "unseen-prompt" condition that undermines most prior detectors, and demonstrates resilience under paraphrasing/decoherence.

### Weaknesses
- The Hilbert-space and projection assumptions (Assumptions 1–3) are strong, but empirical verification of these geometric hypotheses is limited.
- The authors mention small declines on certain datasets but do not analyze why the learned distance struggles there.
- Because the detector is trained using LLM-generated corpora, it may implicitly learn stylistic or semantic regularities specific to those generation distributions. Discussing how well the method generalizes beyond the seven tested generators (e.g., to unseen LLM families) would strengthen the paper.
- Fine-tuning a surrogate model to learn the distance is non-trivial; runtime and parameter-efficiency trade-offs are not quantified.

### Questions
Although this is a well-motivated paper that advances in prompt-robust LLM detection, I still have a few questions that I would truly appreciate if the authors could kindly address during the rebuttal:
- The experiments convincingly demonstrate robustness to paraphrasing and decoherence attacks. Could the authors kindly comment on how the method might behave under stronger semantic-preserving perturbations (e.g., back-translation, style transfer), or whether they foresee any limitations under such conditions?
- While the average AUC gains are impressive, the tables do not include variance or confidence intervals. Would it be possible for the authors to share whether these results were averaged over multiple random seeds, and how stable the improvements are across runs?
- Finally, since the approach requires multiple rewritings and fine-tuning, could the authors give a sense of the computational cost? For example, how much GPU time is typically needed?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a fine-tuning approach for pre-trained LLMs that aims to maximize the divergence in reconstruction errors between human-rewritten and LLM-rewritten text. The method is grounded in the assumption that LLM-generated texts form a subset of human-written texts, and that this relationship persists in their semantic projections—thus justifying the adopted distance-learning strategy. Evaluated through extensive experiments, the method is shown to outperform existing baselines. The primary contribution of this work is a novel perspective on modeling the divergence between LLM-generated and human-written text, coupled with comprehensive empirical validation of the method's effectiveness.

### Strengths
1. The paper presents a well-structured categorization of existing methods for detecting human-written and LLM-generated text, help building a clear roadmap and contextualizing its own research direction.

2. It introduces a novel perspective centered on maximizing the discrepancy between human-written and machine-generated text, which is supported by well-motivated theoretical assumptions.

3. The method's effectiveness is validated through extensive experiments, which include comparisons against a wide range of baselines across multiple datasets.

### Weaknesses
1. The paper mentions that "adaptively learn a distance function" to enhance detection performance, but the implementation of this adaptive learning process is not clearly described. 

2. The pre-trained LLMs used in the experiments are not explicitly identified, making it difficult to assess the experimental setup with confidence. Given the importance of model selection in evaluating the contribution of the work, such implementation details should be unambiguously stated.

3. There is an apparent discrepancy between the reported "relative gain" (e.g., claims of 100% improvement) and the modest absolute improvements observed in the AUC scores. Although the calculation method is provided in the appendix, the presentation in the main text can be misleading. Highlighting large relative gains derived from small absolute baselines may overstate the practical advancement and requires clearer contextualization to avoid confusion.

### Questions
My questions and comments have been stated in the detailed comments.

### Soundness
3

### Presentation
3

### Contribution
2

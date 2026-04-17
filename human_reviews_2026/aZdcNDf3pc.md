# Perturbations Matter: Sensitivity-Guided Hallucination Detection in LLMs

- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
Hallucination detection is essential for ensuring the reliability of large language models. Internal representation–based methods have emerged as the prevailing direction for detecting hallucinations, yet the internal representations often fail to yield clear separability between truthful and hallucinatory content. To address this challenge, we study the separability of  the sensitivity to prompt-induced perturbations in the internal representations. A theory is established to show that, with non-negligible probability, each sample admits a prompt under which factual samples exhibit greater sensitivity to prompt-induced perturbations than hallucinatory samples. When the theory is applied to the representative datasets, the probability reaches nearly 99%, suggesting that sensitivity to perturbations provides a discriminative indicator. Building on this insight, we propose a theory-informed method Sample-Specific Prompting (SSP), which adaptively selects prompts to perturb the model’s internal states and measures the resulting sensitivity as a detection indicator. Extensive experiments across multiple benchmarks demonstrate that SSP consistently outperforms existing hallucination detection methods, validating the practical effectiveness of our method SSP in hallucination detection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel approach to detect hallucinations in LLMs by analyzing the sensitivity of internal representations to prompt-induced perturbations. The key finding is that truthful responses exhibit greater sensitivity to perturbations than hallucinatory ones, with theoretical guarantees for high separability. The proposed method, Sample-Specific Prompting (SSP), dynamically constructs prompts per sample, measures representation shifts via cosine distance, and trains lightweight encoders using contrastive loss. Experiments on QA datasets such as TruthfulQA demonstrate superior AUROC over baselines, along with strong efficiency and generalization.

### Strengths
**Novel Perspective on Hallucination Detection**: This work is the first to explicitly leverage perturbation sensitivity as a detection signal, shifting the focus from static embeddings to dynamic representation shifts.

**Solid Theoretical Foundation**: Theorems 1 and 2 provide probabilistic guarantees on separability, supported by Cantelli's inequality and empirical validation on real datasets. This grounds the method in theory rather than relying solely on heuristics.

**Comprehensive Experiments**: Evaluations across multiple models, datasets demonstrate the robustness of the approach, which outperforms baselines on these benchmarks.

### Weaknesses
The comparison in Figure 1 may be problematic. K-means is an unsupervised method, while the proposed Perturbation Sensitivity is supervised. The poor performance of K-means might not indicate weak inherent separation but could result from its inability to find the optimal separation hyperplane, making the comparison potentially unfair.

Therefore, the claim that "internal representations of LLMs frequently fail to provide a clear separation between truthful and hallucinatory content" requires further discussion. Several recent studies [1,2] have demonstrated good separation using internal representations directly, raising questions about the validity of this separation bottleneck.

Although this method beats the traditional methods like Linear Probe or  SAPLMA, the method introduces complexity by requiring prompt optimization for the training data, which may raise concerns about its robustness and practicality.

[1] Bürger et al. Truth is Universal: Robust Detection of Lies in LLMs. NeurIPS 2024

[2] Liu et al. On the Universal Truthfulness Hyperplane Inside LLMs. EMNLP 2024

### Questions
The main questions concern the limitation described in the introduction and Figure 1.

Beyond the concerns mentioned in the Cons, the paper uses three labeling criteria: ROUGE-L (R), BLEURT (B), and DeepSeek-V3 (D). Given that using ROUGE-L to measure correctness can be inaccurate [3], I suggest that only the scores for DeepSeek-V3 should be displayed, as there are too many scores in Table 1.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies hallucination detection in large language models by measuring the perturbation sensitivity of intermediate embeddings under sample-specific prompts. The method is motivated by a theoretical oracle construction suggesting that prompts can be designed to maximize sensitivity for truthful outputs while minimizing it for hallucinations. Accordingly, the authors propose to generate a noise prompt is generated to probe the model’s internal representations' sensitivity.  The paper evaluates the approach across multiple benchmarks, demonstrating improved detection performance over several baselines and showing robustness across different datasets.

### Strengths
The paper includes thorough evaluations, including multiple datasets and models, and perform valuable ablation studies. Particularly, the method demonstrates promising generalization performance, indicating practical applicability.

### Weaknesses
1. The paper should better situate itself among prior perturbation-based hallucination detection methods (e.g., SPUQ[1]), which often find hallucinated outputs to be more sensitive. While the current work focuses on a different aspect—intermediate embeddings rather than outputs—it nonetheless creates an apparent tension with prior work that would benefit from clarification.

[1] Gao, Xiang, Jiaxin Zhang, Lalla Mouatadid, and Kamalika Das. "SPUQ: Perturbation-Based Uncertainty Quantification for Large Language Models." In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2336-2346. 2024.

2. It would be beneficial to decode and display the perturbed prompts  and report model performance under perturbation.If performance degrades significantly, the method requires two inference passes per sample -- one for generation, one for decoding. In that case, comparisons to simpler baselines, such as linear probes, may no longer be fair given the added computational cost.

3. The proof in Section 3 appears to formalize the intuition rather than justify it. It assumes the existence of a prompt per sample that maximizes sensitivity for non-hallucinated outputs and minimizes it for hallucinated outputs. However, a symmetric argument could be made in the reverse direction if the prompt is assumed to minimize sensitivity for hallucinated outputs. This does not resolve the tension noted above and can make the theoretical claim more confusing.

4. The motivation for defining a separate “separation” metric is unclear. It seems closely related to AUROC and does not appear in prior literature; the added value should be clarified.

5. It would be valuable to see the model's performance on smaller and newer llama models, such as Llama 3.2 1B.

### Questions
see weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a sensitivity-based approach for hallucination detection in large language models (LLMs), arguing that internal representations alone fail to separate truthful and hallucinatory responses effectively. The authors introduce the idea of prompt-induced perturbation sensitivity—measuring how internal representations change when a prompt is slightly altered—and develop a theoretical claim that truthful responses are more sensitive to such perturbations. Building on this observation, they propose a new method, Sample-Specific Prompting (SSP), which dynamically generates prompts to estimate this sensitivity. Experiments across several datasets and LLM architectures show that SSP outperforms existing baselines on AUROC-based metrics.

### Strengths
The paper addresses a relevant and timely problem—detecting hallucinations in LLMs—where most prior approaches focus either on self-assessment or static internal representations. The authors attempt to provide a fresh angle by investigating sensitivity rather than absolute embedding values. The theoretical framework, although somewhat abstract, is clearly presented, and the authors support their arguments with multiple datasets and models. The inclusion of both empirical and analytical sections demonstrates thoroughness, and the work’s emphasis on sample-specific perturbations aligns with growing interest in adaptive and interpretable LLM evaluation methods.

### Weaknesses
Despite its ambitious scope, the paper’s contribution remains limited and somewhat overstated. The central claim—that prompt-induced sensitivity can differentiate truthful from hallucinatory outputs—appears largely empirical and circular, since prompts are tuned per sample in a way that trivially enforces separability - if I am not missing something. The “theoretical” results are more notational restatements of empirical intuition rather than proofs grounded in meaningful probabilistic assumptions. The dependence on oracle-style prompt optimization raises questions about scalability and real-world feasibility—how such per-sample prompt adjustments can be achieved during inference is unclear.

Moreover, the methodology is excessively complex for what amounts to a perturbation-based scoring heuristic. Theoretical bounds presented (Theorems 1 and 2) are neither practically verifiable nor rigorously linked to model architecture or data distributions. The authors claim near-perfect separability (~99%) but later concede that this is merely an “oracle lower bound,” which significantly weakens the practical impact. In the experiments, improvements in AUROC are modest (a few percentage points) once fair baselines and realistic labeling conditions are considered. There is also no statistical significance testing or error analysis to justify the claimed robustness.

Conceptually, the idea of “sensitivity-guided detection” overlaps with well-known approaches in adversarial robustness and influence functions, but these connections are not acknowledged. The paper’s novelty claim therefore feels overstated. The writing also tends toward heavy mathematical formalism without clear intuition, and the overall contribution risks being perceived as incremental.

### Questions
How can the proposed Sample-Specific Prompting (SSP) be efficiently implemented at inference time, given that per-sample optimization is computationally expensive?

What are the actual computational costs (training and inference) compared to strong baselines such as HaloScope or EGH?

Can the authors provide an ablation showing how much improvement comes from sensitivity computation versus prompt tuning?

How sensitive are the reported results to the choice of similarity metric (cosine vs. Euclidean) or embedding layer?

What guarantees exist that perturbation sensitivity reflects factual correctness rather than syntactic or stylistic variance?

The paper claims “theoretical separability.” Could the authors clarify what assumptions make the theorems valid and whether these hold in practical LLMs?

Why are real human-labeled factuality datasets not used for calibration, given the reliance on DeepSeek-V3 pseudo-labels?

Is there any evidence that the proposed metric generalizes to open-ended generation tasks, rather than structured QA datasets?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a hallucination detection technique that uses expertly chosen prompts (adapted for each question) to perturb the distribution of true and hallucinated outputs, creating a bigger separation between them than without these additional prompts. The aim is to use these perturbations to increase the separability between true outputs and hallucinations. The paper first proposes a theoretical discussion on how a specific prompt can be crafted for each question that would almost perfectly separate the hallucinated and true outputs. The paper then supports its claim by comparing its hallucination detection technique with other SOTA techniques, showing definite improvements.

### Strengths
1. The paper provides both a theoretical analysis as well as the empirical results to support their hypothesis. To my understanding, they both appear to be on solid ground.
2. A large plethora of techniques are compared, across several different datasets, and two different models (a third one in the appendix). Overall, the experiments are robust enough to support the final claim.
3. The paper is well written and easy to read. I enjoyed reading this work.

### Weaknesses
I don't have any weaknesses in the soundness or contributions of this paper. I really enjoyed reading this work.

I do, however, have a big objection to papers that move the Related Work section to the appendix. The appendix is to provide additional results/analysis for readers who might be interested in learning more about the paper. It is NOT simply an extension of the main paper, and in my opinion, a lack of related works discussion in the main paper really hurts the readability of the work. I understand problems of limited space, but related work should not be the section that gets axed because of it.

I don't like it when reviewers suggest adding new parts to the paper without also suggesting what should be removed. Just a suggestion, I believe details about Theorem 1 and the results of Figure 2 can be compressed, with the rest moved to the appendix, to make space for related work discussion in the main paper (if the authors prefer, they can even have a 'shorter' related work section in the main paper to situate their work, and a 'longer' related work section in the appendix). Feel free not to take this suggestion, using your own way of finding space, but I strongly recommend having a brief discussion of related work in the main paper.

### Questions
My assumption is that the final 'training data' actually used to train the detector contains a set of 100 questions, one answer per question, and a label for whether that answer is a hallucination or the truth. This is what I believe is used in other detection methods, and so I assume the same is done here.

If the above is correct, the detector never really has access to the 'hallucinated output' and 'true output' pairs for the same question. While the objective is to push all hallucinated answers to low sensitivity and true answers to high sensitivity, which automatically creates a separation, I wonder if having access to actual pairs would help with the separability even more? Curious to hear if the authors think a small set of carefully labeled data with actual pairs could be beneficial, or if just pushing the answers to two extremes can achieve that implicitly?

### Soundness
3

### Presentation
3

### Contribution
3

# Meta-Target DPO: Learning Adaptive Confidence Targets via Meta-Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 10, 4

## Abstract
Direct Preference Optimization (DPO) offers an effective paradigm for aligning Large Language Models (LLMs), yet its performance can be compromised by noisy or ambiguous preference data common in real-world scenarios. Standard DPO formulations often lack mechanisms to adapt to varying levels of reliability across training instances. This paper introduces Meta-Target DPO (MT-DPO), a novel framework that achieves robust preference alignment by dynamically learning adaptive confidence targets for each preference pair. MT-DPO employs a meta-learning approach where an auxiliary confidence module predicts a sample-specific target probability, representing the degree of belief in the observed preference. This module is informed by intrinsic signals, notably perplexity differentials derived from an anchored reference model, indicative of label consistency. Guided by a small, trusted meta-dataset, the confidence module is trained to generate targets that optimally steer the main policy optimization. MT-DPO optimizes the LLM policy using a cross-entropy objective, effectively minimizing the divergence between the policy's implied preference probability and the dynamically learned confidence target for each pair. This allows the learning process to naturally down-weight uncertain instances and potentially rectify contributions from mislabeled data by adapting the target across the full confidence spectrum. Comprehensive experiments on standard alignment benchmarks demonstrate that MT-DPO significantly outperforms vanilla DPO and other robust alignment strategies on both clean and synthetically noisy datasets, showcasing its superior adaptability and effectiveness in handling preference uncertainty through learned target modulation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Meta-Target DPO (MT-DPO), which is an extension of Direct Preference Optimization (DPO) in the direction of meta learning. Specifically, this paper dynamically adapts the confidence targets for each preference pair to manage noisy or ambiguous preference data. MT-DPO improves the robustness by introducing a confidence prediction module which is trained via a small, trusted meta-dataset. Experiments show consistent gains over naive DPO and other robust alignment methods across various datasets and syntheric noise settings.

### Strengths
- The motivation is clear to use the meta learning approach.
- Comprehensive experiments and ablation studies. Showing results of clean (meaning no syntheric noise) also increases reliability.
- Consistent performance gain.

### Weaknesses
- Citations are not in the parentheses: Fix \cite to \citep.
- The method is not novel. It is just applying the conventional meta learning to DPO
- The usage of PPLDiff as confidence is also nothing but an adaptation of the previous research, and it is not proved whether it is an adequate metric for measuring the confidence of a sample. 
- One of the weaknesses of meta learning is its computational difficulty. It is even more serious issue for LLM.
- Experiments with real noise (e.g. HH, UltraFeedback) will improve the credibility of the performances. It is well known that symmetric noise is not enough to represent real world noise.
- It may not be easy to simply plug in to exisiting baselines.

### Questions
- How the authors can be sure the small dataset is high quality and trusted? Specifically, deciding the preference label flip may be okay, but how each pair's exact reward difference can be expressed only even with clean label?
- How the meaning of the confidence module can be interpreted? What is the meaning of the weight?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the problem of noisy and ambiguous preference data. The authors propose Meta-Target DPO (MT-DPO). This new objective trains the policy to match its implied preference probability with an adaptive, sample-specific confidence target ($\hat{p}_{n}$).

The authors introduce a small auxiliary confidence module (a small MLP) that predicts the target $\hat{p}_{n}$ for each training sample. The primary input to this module is the perplexity difference (PPLDiff) between the winner and loser responses, computed using an anchored SFT model.

This confidence module is trained by evaluating the performance of a virtual policy update (trained on the main data with $\hat{p}_{n}$) on a small and clean datased. The primary input to this confidence module is the perplexity differential (PPLDiff) between the two responses, calculated using an anchored reference model.

### Strengths
- The core idea of learning an adaptive, instance-specific confidence target $\hat{p}_{n}$ is a reasonable way to handle noise (particularly compared to the standard of using global noise rates like in cDPO)

- The paper includes a good set of experimental comparisons. It evaluates MT-DPO against several robust baselines, including rDPO, CDPO, and PerpCorrect-DPO. The results in Tables 1 and 2 show consistent gains over these methods.

- Through ablations, the authors demonstrate that the meta-learning process is data-efficient, requiring only a small meta-dataset (e.g., 50-150 samples) to be effective. 

- The ablation in Table 4 confirms that PPLDiff is a critical input feature, and the analysis in Figure 2 gives a good qualitative intuition for how the learned targets $\hat{p}_{n}$ correlate with noise and PPLDiff values.

### Weaknesses
- The primary weakness is that the proposed method seems too complex for the problem it solves. The three-phase meta-learning loop (virtual update, meta-update, actual update) introduces significant implementation complexity. The paper's own ablations (Table 4) and analysis (Figure 2) show that the PPLDiff is the dominant, if not the only, input feature and that the learned target $p_n$ is strongly correlated with it. This suggests the complex meta-learning framework is essentially just learning a simple function $f(\text{PPLDiff}) \rightarrow p_{n}$. The paper is missing a **very important** baseline: using PPLDiff directly as the confidence target, maybe via a simple, non-learned heuristic like $\hat{p}_{n} = \sigma(-k \cdot \text{PPLDiff})$ (where $k$ is a tuned hyperparameter). Without this comparison, the added complexity of the meta-learning method is not justified.

- The paper claims to handle ambiguous preference data, but the solution is to learn a target $p_n \approx 0.5$. This is equivalent to down-weighting or ignoring the sample, not really dealing with preference pluralism (e.g., by learning a multimodal preference distribution). Like many robustness-focused methods, this approach still implicitly assumes a single "collective" preference and treats subjective disagreements as noise to be discarded. 

- The meta-learning process relies on two key components: (1) a "virtual policy update" $\theta_{virtual}^{(t+1)}$, which depends on the current policy $\pi_{\theta}^{(t)}$, and (2) the PPLDiff signal, which depends on the anchor model $\pi_{anchor}$. All experiments initialize the policy $\pi_{\theta}$ from a well-tuned SFT model. If the policy starts in a "colder" (less-aligned) state, its virtual updates may be noisy or uninformative, which could destabilize the meta-gradient for the confidence module. The paper does not investigate this potential instability.

- How does performance change if the new updated model is used vs an achored model to calculate the PPLDiff?

- The paper is ambiguous about the exact inputs to the confidence module $V(\cdot;\phi)$. Section 3.3 states it "primarily" uses PPLDiff. However, the hyperparameter table (Table 5) lists only "Standardized PPLDiff" as the input, and the ablation in Table 4 also seems to show single-feature inputs. This should be clarified. If it is only PPLDiff, this strengthens Weakness #1.

- The cDPO result in Table 1 seems to weak. How was the noise rate tuned?

- In Table 3, why is 96.58 highlighted and not 96.70? Also, confidence intervals should be included for the win rates. Also, the standard is to compare win rates against GPT-4, not the SFT model, can those win rates be reported? Alongside the Length Controlled win rates as well

### Questions
- Given the method's complexity and the strong signal from PPLDiff, could the authors please provide a comparison against a simpler heuristic baseline that uses PPLDiff directly to set the target $p_n$ (e.g., $p_n = \sigma(-k \cdot \text{PPLDiff})$), without any meta-learning? This seems like the most critical missing comparison to justify the framework

- Could the authors clarify whether only PPLDiff was used as the input to the confidence module for the main experiments reported in Tables 1 and 2? How does the MT-DPO framework perform if the initial policy $\pi_{\theta}$ is initialized from a base pre-trained model (e.g., base Llama-2) rather than the SFT model? Does the meta-learning process remain stable when the initial virtual policy updates are based on a poorly-aligned model?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
It has previously been shown that the overfitting of DPO to noisy preferences can be ameliorated by using binary cross entropy with target probabilities that estimate the noise rate of preferences.  This paper proposes estimating the target probabilities using an auxiliary confidence network, with features chosen to be informative about the quality of a preference pair, and trained using a meta-learning procedure on a small meta-learning dataset with high-accuracy preferences.

### Strengths
The proposed meta-learning procedure is an effective and elegant solution to the problem of estimating the confidence of preference scores given the very small size of preference datasets.

Paper is well written, and appendices were highly informative.

Experiments demonstrate superiority over strong baselines.  Statistical significance of the improvement is demonstrated.

### Weaknesses
None noted.

### Questions
p. 6 says that PPLDiff is the "primary input" of the confidence module in experiments.  Appendix D seems to indicate that it's not just the "primary input," it's actually the only input - is that correct?  Have you done any studies considering other possible inputs?

The base models are somewhat old  (Llama-2-7-B and Phi-2).  Why? Is that necessary to make the meta-learning gradient computable, or is that so that the "gold standard" GPT-4 can be considered 100% correct by comparison?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Meta-Target DPO (MT-DPO), a robust framework designed to improve preference alignment of Large Language Models (LLMs) under noisy or ambiguous supervision. The method augments standard DPO by learning adaptive confidence targets for each preference pair via a meta-learning module, trained using a small, trusted meta-dataset. The confidence module leverages intrinsic signals such as perplexity differentials from a reference model to infer label reliability, and these learned targets guide the main policy optimization through a cross-entropy objective. Empirical results demonstrate superior robustness and adaptability over baseline DPO variants, particularly under synthetic noise and real-world uncertainty conditions.

### Strengths
1 The paper is well written and easy to follow.

2. The problem of adaptive weights with meta-learning is interesting.

3. The authors conduct some experiments to verify the effectiveness of their method.

### Weaknesses
1. The improvement compared with baselines like PerpCorrect-DPO is limited but the training pipeline is complex.

2. It's better for authors to provide detailed time complexity analysis and training time comparison to further verify their method except from Appendix D.3.

3. The authors conduct experiments on old models like Llama2 or Phi-2 which limit the conclusion from their experiments. Moreover, it's better to adopt their methods to different variants of DPO algorithms.

### Questions
Please refer to weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3

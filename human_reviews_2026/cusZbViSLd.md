# Bounds of Chain-of-Thought Robustness: Reasoning Steps, Embed Norms, and Beyond

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Existing research indicates that the output of **Chain-of-Thought (CoT)** is significantly affected by input perturbations.  
Although many methods aim to mitigate such impact by optimizing prompts, a theoretical explanation of how these perturbations influence CoT outputs remains an open area of research.  
This gap limits our in-depth understanding of how input perturbations propagate during the reasoning process and hinders further improvements in prompt optimization methods.  

Therefore, in this paper, we theoretically analyze the effect of input perturbations on the fluctuation of CoT outputs.  
We first derive an upper bound for input perturbations under the condition that the output fluctuation is within an acceptable range, and we prove that:  

- *i)* This upper bound is **positively correlated** with the number of reasoning steps in the CoT;  
- *ii)* Even an infinitely long reasoning process **cannot eliminate** the impact of input perturbations.  

We then apply these conclusions to the **Linear Self-Attention (LSA)** model, which can be viewed as a simplified version of Transformer.  
For the LSA model, we prove that the upper bound for input perturbation is **negatively correlated** with the norms of the input embedding and hidden state vectors.  

To validate this theoretical analysis, we conduct experiments on **three mainstream datasets** and **four mainstream models**.  
The experimental results align with our theoretical analysis, empirically demonstrating the correctness of our findings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper provides a theoretical investigation into the robustness of Chain-of-Thought (CoT) against input perturbations, supported by empirical tests on a range of open-source models to validate their arguments. The authors model the CoT process as a bivariate Lipschitz function and leverage classical stability results of Lipschitz functions to derive stability relationships for the Linear Self-Attention (LSA) model. Through theoretical analysis and experimental validation, the authors find that: (1) model performance is positively correlated with robustness; (2) longer reasoning chains reduce the impact of input perturbations, but cannot eliminate it entirely; and (3) the norms of the embedding and hidden state vectors are negatively correlated with model robustness. The paper also proposes a novel prompt optimization method aimed at enhancing model robustness.

### Strengths
1. This paper rigorously derives bounds on output fluctuation under input perturbation, building from a Lipschitz continuity framework (Eq. 1, Thm. 1) to a specialized analysis of Linear Self-Attention (LSA). This progression yields novel insights, crucially linking perturbation robustness to the norms of input and hidden state embeddings (Thm. 2).
2. The experimental evaluation is comprehensive, conducted across multiple datasets and state-of-the-art LLMs. The results, notably in Table 1 and Figures 1-3, provide compelling quantitative evidence for the paper's central claims by directly visualizing the relationships between input perturbation, reasoning steps, and output fluctuation.

### Weaknesses
1. The paper models the LLM's reasoning process directly as a bivariate function. However, in reality, the input to an LLM during reasoning clearly consists of more than two tokens. The authors need to more clearly demonstrate why this model holds.
2. The authors primarily provide a theoretical analysis for the Linear Self-Attention (LSA) model in the main text, while the discussion of other modules is relegated to the appendix and remains rather superficial. A more in-depth discussion on the impact of these other components on the theoretical results is necessary, or alternatively, the authors should justify why the LSA findings are sufficient for analyzing the entire model.
3. In the experimental section, the authors need to clearly describe their procedure for perturbing the input prompts and explain how they modify the norms of the embedding and hidden state vectors.
4. In the results shown in Figures 2 and 3, the model performance declines as the length of the Chain-of-Thought increases. This suggests that, under the experimental setup of this paper, extending the number of CoT steps may no longer be beneficial. Consequently, the validity of the conclusions related to Output Fluctuation is called into concern. The authors should identify more suitable tasks where increasing CoT steps demonstrably improves model performance, and then examine the behavior of OF under such conditions.
5. The results presented in Figures 4 and 5 do not provide compelling support for the authors' claims. Particularly in Figure 4, aside from the notable jump between 60-70, the remaining data points lack a clear monotonic increasing trend.
6. The paper identifies that larger embedding and hidden state norms reduce CoT robustness, a finding that appears counterintuitive from a practical training perspective. Standard techniques that effectively constrain these norms (e.g., aggressive weight decay, smaller initialization) are well-known to potentially harm training stability. Furthermore, if the absolute magnitude of the perturbation is held constant, a smaller vector norm would intuitively represent a larger *relative* perturbation, which one might expect to be more detrimental to robustness. The authors should provide a more intuitive explanation and discussion to reconcile this theoretical finding with practical intuition. For instance, is the analysis sensitive to the assumption that the perturbation is absolute rather than relative? A discussion on this apparent discrepancy is crucial for the audience to properly interpret and trust the theoretical results.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a theoretical analysis of how input perturbations propagate through CoT reasoning in LLMs. It derives upper bounds on output fluctuations under Lipschitz continuity, showing that: (i) increasing reasoning steps mitigate, but never remove, the impact of input perturbations; and (ii) robustness is negatively correlated with the norms of input embeddings and hidden state vectors. The analysis is instantiated on a Linear Self-Attention model, treated as a simplified Transformer, and validated through experiments on four models and three benchmarks. Empirical results align with theory, confirming that longer reasoning chains dampen but do not eliminate perturbation sensitivity. The authors also propose a simple prompt-pick strategy that maximises input-perturbation tolerance, yielding modest improvements.

### Strengths
- Theoretical grounding: The paper offers one of the first formal derivations of CoT robustness bounds, establishing links between perturbations, reasoning steps, and embedding norms.


- Clear empirical validation: Experiments on diverse datasets and models convincingly support theoretical claims.

- Insightful analysis of LSA: Using Linear Self-Attention provides tractable yet informative insights into Transformer dynamics.

### Weaknesses
- Simplified modelling assumptions: The reliance on LSA as a proxy for real Transformers oversimplifies key non-linear dynamics (softmax attention, layer normalisation).

- Lack of deeper interpretability: While the derivations are sound, the paper does not explore the mechanistic implications for CoT reasoning or noise propagation in practical LLMs.

- Marginal methodological contribution: The proposed prompt optimisation method adds limited value and feels auxiliary to the theoretical core.

- Lack of transparency: The authors could have shared the code in order to improve reproducibility and be transparent

### Questions
- Theoretically, why should the negative correlation between robustness and embedding norms hold across non-linear, attention-based architectures?

- Beyond quantifying bounds, how might your framework inform practical interventions to improve CoT robustness during training or inference?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper provides an analysis on the robustness of CoT reasoning w.r.t. input perturbations. The authors derive output upper bounds on how input perturbations propagate through CoT reasoning, claiming that increasing the number of reasoning steps reduces the fluctuation, but it can't be fully eliminated. Besides they show that robustness is negatively correlated with input and hidden state norms.

### Strengths
Providing a formal mathematical model of CoT perturbation propagation and finding an upper bound on robustness.

### Weaknesses
From what i understand the Lipschitz continuity assumption is an over simplifying assumption

### Questions
I am not sure about this but the authors considered multiple assumption to justify their modeling. Is there any prove that these assumption doesn't make the results to diverge from real cases?

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
This paper studies the robustness of CoT reasoning and theoretically proves that while increasing reasoning steps can partially mitigate the effect of input perturbations, even infinitely long chains cannot completely eliminate them. Further analysis on the simplified Linear Self-Attention model shows that larger input embedding and hidden state norms reduce robustness. Experiments on several mainstream large models and reasoning datasets validate the theory, and a new prompt selection method based on norm minimization is proposed to effectively enhance robustness.

### Strengths
This paper provides a formal theoretical derivation of the robustness of CoT reasoning and establishes an upper bound on output fluctuations under input perturbations.

### Weaknesses
1. This paper primarily adopts OF as the robustness metric, but does not account for semantic-level robustness. Could this limitation lead to an underestimation of the model’s actual robustness?

2. This paper does not specify the exact prompt templates used in the experiments. Could the authors clarify the prompt design choices?

3. The analysis suggests that larger input embedding norms reduce robustness. However, given that different models vary in embedding initialization and pretraining corpora, how generalizable is this finding across architectures?

### Questions
see weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

# How Do Language Models Speak Languages? A Case Study on Unintended Code-Switching

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Unintended code-switching, which refers to the phenomenon where LLM unexpectedly switch languages, poses a fundamental challenge in the multilingual capabilities in LLMs.
However, the fundamental properties of their underlying circuits, such as what they consist of, where they emerge in the network, and how to mitigate their effects, remain unexplored.
Existing works on the mechanistic interpretability depend on additional training (e.g., sparse autoencoders) or manual annotation, both of which pose limitations in real-world scenarios.
In this work, we introduce a scalable circuit discovery framework that causally localizes multilingual neurons, describes their functional patterns, and groups neurons into circuits.
We find that the circuits for multilingual generation fall into two different regimes: a language regime which acts as a lingual key to detect language patterns, and a semantic regime which functions as a contextual value to retrieving language-agnostic semantics.
These two regimes, in normal cases, converge smoothly to make final predictions, but in code-switching scenarios, semantics dominate the circuit, overriding typical language pathways and destabilizing outputs.
Furthermore, we fine-tune the identified language sub-circuit ($\sim0.019$\% of all neurons), reducing the code-switching rate by $20.8$\% with minimal parameter updates, validating the effectiveness of the discovered circuits for practical scalability. Our work serves as a preliminary exploration of multilingual generation circuits, offering actionable insights for neuron-based mechanistic interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the neural circuits behind unintended code-switching in multilingual LLMs. Using a proposed causal discovery framework, it identifies two key regimes: a language circuit that detects linguistic patterns and a semantic circuit that retrieves language-agnostic meaning. The study shows that imbalances between these regimes trigger code-switching and that fine-tuning the language sub-circuit significantly reduces this behavior with minimal updates.

### Strengths
- This study conducts analysis on the cause of language-switching; The research theme is important for safety and intriguing.
- The proposed method is quite interesting.
- The authors conducted a comprehensive experiments using two models, multiple languages, multiple benchmarks, ensuring the findings and effectiveness.

### Weaknesses
- Although this study is quite interesting, there are significant issues for the readability of this paper. Please look at "Questions" for the detail.
- Equation (3): I think this equation assumes d(a_l) / d(a_e) = 1, but is this assumption reasonable? Any comment?
- 4.3: This chapter is important because this experiment result is a direct ground for the claim about "the language confusion is the result of competition between language-specific circuits and semantic circuits". So, in order to supplement the claim, I think we need an extensive experiments, for example how about strengthen the circuits of language neurons instead of ablating semantic circuits?

### Questions
- L166: Atteibution Patching: This is an explanation of the existing work, so I think this paragraph should move to outside Chapter 3 (Method). For example, how about creating preliminary chapter?
- Equation(1): This equation is important as a fundamental information to understand the proposed method (Hierarchical Attribution Patching). I recommend that the authors write more detailed derivation of this equation.
- Equation(3): Likewise, I recommend that the authors write more detailed derivation of this equation.
- L190: What is the definition of "late neuron" and "early neuron"?
- L167: does neuron "n \in R^d" refer to "a column of the MLP down-projection" as described at the footnote in page4?
- L220: What is a definition of "e"?
- L222: What is a definition of activation()?
- L261: does metric m refer to equation(2)? or something else?
- L264: What is a detailed definition of m(\phi) and m(M)? I could not clearly understand the exact process based on the current description.
- L350: What is a definition of "Averaged attribution ratios"? Please explain the details.

### Soundness
3

### Presentation
1

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
This paper investigates how multilingual language models internally represent languages and why they sometimes code-switch unexpectedly. The authors develop a hierarchical circuit-discovery framework that traces causal pathways from neurons to model outputs. They use attribution patching and neuron-level interventions to isolate circuits responsible for language versus semantic processing. Through controlled experiments, they find that disabling or amplifying a very small subset of neurons (around 0.02% of total) significantly alters the rate of code-switching, supporting the view that unintended switching results from competition between language and semantic circuits. The method also enables lightweight mitigation through targeted fine-tuning, which reduces code-switching by more than 20% without harming general performance.

### Strengths
The study provides a clear and rigorous causal framework for analyzing multilingual behavior in large models, going beyond correlational neuron activation studies. Its conceptual contribution lies in formalizing code-switching as a competition between distinct subcircuits and providing empirical validation through targeted interventions. The approach is carefully designed: attribution patching, hierarchical tracing, and neuron-level ablations are combined coherently. The experiments are extensive and include both diagnostic and corrective analyses, showing consistent causal effects across two model architectures and multiple languages.

The paper is also methodologically elegant and transparent. It integrates interpretability techniques into a reproducible pipeline, avoiding heuristic prompt-based evaluation. The figures and tables clearly illustrate how the identified neurons affect behavior, and the mitigation results demonstrate a practically relevant application. Overall, the paper balances mechanistic insight with empirical robustness, offering one of the most convincing analyses to date of language competition inside multilingual LLMs.

### Weaknesses
While the framing of language and semantic circuits is intellectually appealing, the novelty is moderate because the main components—attribution patching, neuron labeling via LLMs, and targeted ablation—are adaptations of existing interpretability tools. The paper’s originality therefore lies in its integration and application rather than in new methodology. A deeper theoretical justification of the “competition” model or mathematical analysis of how these circuits interact would strengthen the contribution.

The evaluation scope is also somewhat limited. Only two model families are studied, both decoder-only transformers with similar architectures. Attention circuits are not analyzed in depth, leaving open whether the phenomenon is primarily MLP-based or distributed across attention heads. The neuron description and grouping process depends on LLM judgments, which, although checked for precision, could introduce bias. Finally, the mitigation results focus on controlled test sets rather than naturally occurring code-switched corpora, so the real-world generality of the fix is uncertain.

### Questions
1. How sensitive are the results to model architecture and scale? Have you tested whether the same type of language-semantic circuit separation appears in models such as Mistral or Gemma?

2. The hierarchical attribution assumes approximate linearity in the residual stream. How does nonlinearity between layers affect the faithfulness of the discovered circuits?

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
# 🧩 Summary
This paper explores the mechanistic basis of **unintended code-switching** in multilingual large language models (LLMs).  
The authors propose a **causal circuit discovery framework** that identifies neuron-level structures involved in multilingual text generation.  
The framework integrates:
1. **Circuit Localization** – a hierarchical extension of *Attribution Patching* (Nanda, 2022), tracing neuron-to-neuron causal connections;  
2. **Neuron Description** – using an auxiliary LLM to generate textual explanations from token projections and activation samples;  
3. **Neuron Grouping** – clustering neurons into interpretable “super-neurons” based on these descriptions.  

Through experiments on **Qwen2.5-7B** and **LLaMA3.1-8B**, the authors show that multilingual generation involves two interacting subsystems:  
a **Language sub-circuit** maintaining linguistic context, and a **Semantic sub-circuit** capturing language-agnostic meaning.  
They argue that unintended code-switching arises when semantic circuits dominate language circuits, and that fine-tuning only ~0.019% of neurons mitigates this by **20.8%**.

### Strengths
- **Novel application of mechanistic interpretability.**  
  Code-switching is an interesting and underexplored multilingual behavior, and analyzing it through causal circuit tracing is conceptually fresh.  

- **Detailed technical description.**  
  The paper provides pseudo-code (Algorithm 1), datasets, and hyperparameters, making the procedure relatively transparent.

- **Empirical support.**  
  Ablation and fine-tuning experiments show that identified neurons have tangible, directionally causal influence on multilingual output.

- **Clear organization.**  
  The presentation of results—particularly Figures 1–3 and Tables 1–5—is systematic and accessible.

### Weaknesses
### 1. Limited generality despite broad framing
The paper describes its framework as “universal,” but this appears to refer to universality *within multilingual generation*.  
All methodological components—the contrastive data construction, logit-difference metric, and grouping prompts—are defined specifically for language variation (e.g., English vs. French tokens).  
As such, its applicability to **non-linguistic circuits** (e.g., reasoning, arithmetic, syntax) remains **untested**, not necessarily impossible.  
To claim methodological generality, experiments on other capabilities would be required.

### 2. Ambiguity in the significance of the phenomenon
While the introduction states that code-switching undermines the multilingual reliability of LLMs, the **broader motivation and impact** of identifying its neural cause remain underdeveloped.  
The work does not clearly articulate whether the findings  
(a) inform multilingual training design,  
(b) improve robustness or safety, or  
(c) offer theoretical insight into cognitive code-switching.  
Thus, the study risks appearing descriptive rather than explanatory.  
Clarifying the *scientific or practical value* of uncovering these circuits would substantially strengthen the paper.

### 3. Causal rigor and robustness
The causal evidence relies on attribution ratios and targeted suppression, which demonstrate directional sensitivity but not sufficiency.  
The hierarchical patching assumes local linearity (SiLU removal) and uses fixed hyperparameters (ϵ = 0.001, L = 5) without sensitivity analysis.  
These limitations are acknowledged (Appendix A.1.5) but not explored experimentally.

### 4. Neuron description methodology
The LLM-based “Neuron Description” step increases readability but adds interpretive variability.  
Table 9 reports only moderate alignment (≈ 0.7 precision) between explanations and activations.  
While the authors argue that textual descriptions capture superposed neurons better than static token lists, this claim is qualitative.  
Quantitative comparisons or human evaluations would help establish whether such descriptions meaningfully improve interpretability.

### 5. Evaluation and reproducibility
The behavioral metrics (LPR/LCR) rely on heuristic language identification and LLM-assisted annotation (Appendix A.4.3), which may misclassify loanwords or named entities.  
A small-scale human evaluation would increase confidence.  
The reproducibility statement mentions partial code in supplementary materials and a plan for full release upon acceptance—helpful, but not yet public.  
Computation cost (GPU-hours) is also unreported.  
Minor typos remain (e.g., *none-code-switched → non-code-switched*).

### Questions
1. Could you clarify how identifying the cause of code-switching advances LLM robustness or linguistic understanding?  
2. How generalizable is the proposed framework to **non-linguistic circuits**, and what modifications would be required?  
3. How stable are the LLM-based neuron descriptions across random seeds or different explainer models?  
4. Have you tested sensitivity to key hyperparameters (ϵ, L) in hierarchical patching?  
5. Can you provide quantitative comparisons with Sparse Autoencoder or ATP\* frameworks?  
6. Has any human validation been performed for the LPR/LCR metrics?  
7. What is the approximate runtime (GPU-hours) for one full circuit-discovery pipeline?

### Soundness
3

### Presentation
3

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
The paper investigates unintended code-switching in multilingual LLMs, a phenomenon where models unexpectedly switch between languages during generation. The authors propose a causal circuit discovery framework that identifies, interprets, and groups multilingual neurons into circuits governing linguistic behavior. Using attribution patching, they reveal two main circuit regimes: (1) language circuits that detect language patterns, and (2) semantic circuits that represent language-agnostic meaning. They argue that code-switching arises from competition between these circuits when semantic activations override language-specific ones. Experiments on Qwen2.5-7B and Llama3.1-8B show that fine-tuning a small set of language-specific neurons (≈0.02%) reduces code-switching by ~20% without hurting task performance.

### Strengths
- Novel mechanistic approach: The paper introduces a scalable, neuron-level causal tracing method that extends attribution patching, addressing a gap in multilingual interpretability research.

- Insightful hypothesis: It identifies competition between semantic and linguistic circuits as the mechanistic source of code-switching, a plausible and interpretable explanation.

- Strong empirical results: The selective neuron fine-tuning experiments are efficient and demonstrate measurable improvement (20.8% reduction) in code-switching.

- Method generality: The framework is model-agnostic and tested across two different LLMs and several languages, showing robustness.

- Clear connection to interpretability: The combination of automatic neuron description and grouping makes a step toward more interpretable circuit-level analysis.

### Weaknesses
- Overstated causal claims: The method’s causal validity is not convincingly demonstrated—gradient-based attribution and patching only provide correlational evidence under strong linearity assumptions.

- Limited theoretical grounding: The “competition between semantic and language circuits” hypothesis lacks rigorous formalization and is supported mainly by qualitative visualizations.

- Reproducibility concerns: Although implementation details are mentioned, many core components (e.g., neuron labeling prompts, clustering procedure, dataset construction) depend on opaque LLM judgments.

### Questions
as suggestion you might also want to use a token-level or sequence-level code-switching detection approaches.
For instance, for code-switch language identification: https://aclanthology.org/2024.acl-short.43/ which uses models like https://aclanthology.org/2023.findings-emnlp.410/ to automatically detect and classify code-switched content.

### Soundness
3

### Presentation
3

### Contribution
3

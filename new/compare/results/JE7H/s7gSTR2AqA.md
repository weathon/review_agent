---
job_id: 364e251e-3d05-4236-b3d0-20733f0fa74b
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: s7gSTR2AqA.pdf
paper: Evolution and Compression in LLMs: On the Emergence of Human-Aligned Categorization
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work studies semantic category formation and inductive biases in large language models using information-theoretic tools and iterated learning, which clearly falls within representation learning, interpretability, and applications to cognitive science, all squarely in ICLR’s scope.

## Minimum Quality
Pass ✅.  
The paper has all core components (Abstract, Introduction, Background/Related Work, Methodology/Experimental setup, Results, Discussion). The work is technically non‑trivial, empirically substantial, and the exposition is clear enough to evaluate. I see no fatal methodological error or misuse of data strong enough to warrant desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no hidden prompts, instructions targeting automated reviewers, or other manipulative content in the main paper.

---

# Expected Review Outcome:

## Summary

The paper investigates whether large language models (LLMs) develop human‑aligned, information‑theoretically efficient semantic categories, focusing on color as a testbed.  

First, the authors run an English color‑naming experiment with 39 models, analyze the resulting category systems in the Information Bottleneck (IB) complexity–accuracy plane, and compare them to human English and World Color Survey (WCS) languages.  

Second, they introduce Iterated in‑Context Language Learning (IICLL), an LLM analogue of human iterated language learning, to study how pseudo color‑naming systems evolve over generations and whether they converge toward IB‑efficient and human‑like solutions; they further show preliminary results on “Shepard circles” as a non‑color domain.

---

## Strengths

1. **Clear, theory‑driven connection between IB and LLM behavior.**  
   The paper leverages the IB framework of Zaslavsky et al. (2018) to place LLM‑induced color naming systems in the information plane. Using the pre‑existing IB color model and Eq. (1) / Eq. (4) for the objective, they can directly compare LLM systems to the theoretical IB bound and to human languages. This is conceptually clean and gives a principled way to quantify “efficiency” rather than relying on ad‑hoc metrics.

2. **Broad and systematic empirical study across many models.**  
   The English naming experiment is run on 39 models spanning several families, sizes, and training regimes (Table 1, Appendix D). Figure 2(c) summarizes English‑alignment (1 − NID) and IB complexity for all these models, clearly showing systematic trends: larger, instruction‑tuned models tend to have higher complexity and better alignment to English, while many base or smaller models either collapse to very simple systems or produce noisy categories. This breadth of evaluation is valuable for the community and goes well beyond typical single‑model cognitive comparisons.

3. **Strong use of visual analyses that make the claims tangible.**  
   - **Figure 2(a)** places instruction‑tuned models on the complexity–accuracy plane next to the IB bound and the English point. It makes very clear that only a few models (e.g. Gemini‑2.0, Gemma‑3‑27B‑it) approach English and that many lie well below both English and the IB curve, supporting the claim that most LLMs do not naturally recover the English color system.  
   - **Figure 2(b)** shows mode maps for English and several top‑performing models. The Gemini and Gemma maps visually resemble the English partition, whereas Llama and Qwen show different partitioning patterns. This figure is crucial for demonstrating that “alignment” is not just a scalar but reflects real category structure.  
   - **Figure 3** shows IICLL trajectories overlayed with WCS languages and human IL endpoints in the information plane. Gemini’s trajectories visibly span a comparable complexity range to WCS data and human IL systems, whereas Gemma, Llama, and Qwen mostly end up in the low‑complexity region. This strongly supports the core narrative about model‑dependent inductive biases.

4. **Carefully designed iterated learning setup with informative dynamics.**  
   The IICLL procedure (Section 4.2 and Appendix G, Figure 1(c)) closely parallels human iterated language learning: each “generation” is trained in‑context on a small sample of color–label pairs, then asked to label all 330 WCS chips, and the resulting system is sampled to train the next generation. The dynamics in **Figure 4** are particularly compelling: efficiency loss drops rapidly over generations (4a), IB‑alignment improves (4b), and alignment with WCS languages increases (4c), with convergence in roughly 4–5 generations, mirroring human IL studies. This is one of the most convincing pieces of evidence that the models’ behavior cannot be reduced to simple memorization of initial random labels.

5. **Non‑trivial baselines and control analyses.**  
   The authors do not stop at descriptive plots. They provide:
   - A **rotation analysis** (Figure 11) testing whether the final LLM systems are special relative to hue‑cyclic permutations, showing that for Gemini, rotations significantly reduce efficiency and alignment, ruling out trivial contiguity artifacts.  
   - An **NN baseline** for IICLL (Appendix M, Figure 17) that applies nearest‑neighbor classification in sRGB space under the same iterated‑sampling pipeline. The fact that Gemini’s $k=14$ trajectories in Figure 17 achieve higher efficiency and alignment than this NN baseline supports the claim that the emergent structure is not just “smooth clustering” in sRGB.  
   - A **history window ablation** (Appendix K, Figures 14–15) that shows how omitting conversation history (window size 0) leads to more degenerate systems, which both validates the need for history and documents its impact.

6. **Insightful findings on training stages and input modalities.**  
   - The Olmo‑2‑32B checkpoint analysis (Appendix F, **Figure 10**) shows that English‑alignment only slightly increases over stage‑1 pretraining but jumps after early stage‑2 instruction‑tuning, directly linking alignment to instruction‑tuning rather than just scale.  
   - The comparison between sRGB and CIELAB inputs in Appendix C (**Figure 6**) and the multimodal minimal‑pair results in **Figure 8** highlight an interesting and non‑obvious result: feeding models CIELAB coordinates (closer to human perception) or even images does not systematically improve performance and can hurt high‑performing models, while sometimes helping smaller ones. This suggests a mismatch between how LLMs internally represent color and human perceptual spaces.

7. **Preliminary evidence for cross‑domain generality.**  
   The Shepard circles experiment (Section 4.3, **Figure 5**) is admittedly small‑scale but shows that Gemini, under IICLL, starts from random pseudo labels and, over generations, produces increasingly compact and interpretable partitions in a 2D conceptual space that reflect both radius and angle. Even though IB metrics are not computed here, the qualitative trajectories are an interesting first step toward generalizing the account beyond color.

8. **Relevance for interpretability and cognitive alignment of LLMs.**  
   The paper offers a concrete, testable notion of “semantic efficiency” and “human‑alignment” grounded in information theory and rich cognitive data, not loose intuitions. For the interpretability and cognitive modeling community around LLMs, this is a useful methodological blueprint for probing representations via IB‑style evaluation and iterated in‑context learning.

---

## Weaknesses

1. **Methodological and mathematical details of the IB computations are under‑specified.**  
   The central empirical claims hinge on efficiency loss $\varepsilon$, mutual informations $I_q(M;W)$ and $I_q(W;U)$, and the IB bound $\mathcal{F}_\beta^*$ from Eq. (1) and Eq. (4). However, the main paper does not spell out how these quantities are estimated from empirical LLM naming data, nor how optimal IB systems and $\mathcal{F}_\beta^*$ are numerically obtained. For instance:
   - There is no description of the optimization algorithm used to obtain the IB curve (presumably some Blahut–Arimoto‑type procedure) nor of any regularization or smoothing applied to $q(w|m)$, which is estimated from finite sample naming data.  
   - The paper defines $\varepsilon=\min_{\beta}\{\frac{1}{\beta}(\mathcal{F}_{\beta}[q]-\mathcal{F}_{\beta}^{*})\}$, but does not explain the range or grid of $\beta$ used, how $\mathcal{F}_\beta^*$ is computed for each $\beta$, or how sensitive $\varepsilon$ is to these choices.  
   - Appendix A gives formal definitions in Eqs. (2) and (3), but still omits numerical details that are essential to trust the exact positions of points and the magnitude of efficiency loss in Figures 2, 3, and 4.  
   Since the conclusions depend crucially on “near‑optimality” claims (e.g., that Gemini’s trajectories in Figure 3 lie near the IB bound), a more transparent mathematical and algorithmic description is needed.

2. **Some definitional and notational inconsistencies around $\beta$ and the IB objective.**  
   In the main text, Eq. (1) defines the IB objective $\mathcal{F}_\beta[q]=I_q(M;W)-\beta I_q(W;U)$ with $\beta\ge 0$, whereas Appendix Eq. (4) uses $\beta\ge 1$. This is a small but non‑trivial inconsistency, especially given that efficiency loss is computed by minimizing over $\beta$. The paper never clarifies whether the optimization is over $\beta\in[0,\infty)$ or $\beta\in[1,\infty)$, nor how this choice relates to the balance between complexity and accuracy. This does not invalidate the general pattern but weakens the mathematical rigor of the evaluation.

3. **Over‑interpretation of IICLL results as evidence for an intrinsic IB‑efficiency bias.**  
   The authors conclude that “Gemini truly exhibits an emergent inductive learning bias toward IB‑efficiency” and that LLMs more broadly may share the same IB‑driven principle as humans (e.g., Section 4.2 and Discussion). While the IICLL trajectories in **Figure 3** and the metrics in **Figure 4** are suggestive, alternative explanations are not fully ruled out:
   - The NN‑rgb baseline in Appendix M is only evaluated for the $k=14$ condition, and only in an aggregate way (Figure 17). It is plausible that a relatively simple algorithm that enforces contiguity and balanced use of labels across generations could also yield high IB‑alignment in the lower‑$k$ cases.  
   - The rotation analysis (Appendix H, Figure 11) shows strong non‑triviality for Gemini but results are “less conclusive” for Gemma, Llama, and Qwen. This suggests that for some models, proximity to the IB curve may partly reflect generic contiguity in color space rather than an explicit drive toward IB optimality.  
   - The iterated chains are relatively short (13 generations max), and there is no systematic analysis of stationary distributions as in Griffiths & Kalish (2007).  
   Overall, the data show that some LLMs can move toward IB‑efficient, human‑like systems under IICLL, but the inference that they “share the same fundamental principle that underlies semantic efficiency in humans” feels stronger than what is warranted by the present evidence.

4. **Prompting and decoding choices may significantly affect results but are not systematically explored.**  
   For open‑weight models, the authors implement constrained choice by scoring allowed terms by log‑probability under a single continuation and sampling with default HF settings (temperature 0.6, top‑p 0.9). This is a non‑standard hybrid between classification and generative sampling, and yet:
   - There is no ablation on temperature/top‑p, nor a comparison to deterministic argmax over allowed terms. Both could materially alter noise levels in category systems, especially for smaller and less instruction‑tuned models.  
   - For Gemini, the API supports hard constrained decoding, while for the others the authors approximate constraints using logprob scoring. This difference is not clearly controlled for, and could partly explain Gemini’s cleaner, more stable systems in Figures 2 and 3.  
   - In the IICLL setting, the sliding window of 10 previous interactions (Appendix K) is motivated by preliminary experiments, but beyond counting degenerate chains in Figures 14–15, there is no detailed analysis of how window size affects complexity, alignment, or convergence speed.  
   Given that small shifts in prompt or decoding strategy can radically change LLM behavior, the strong model‑comparison claims would be more convincing with a richer set of ablations on these choices.

5. **English color naming experiment uses an arguably unnatural input representation without thorough justification.**  
   In the main English naming experiment, colors are presented as numeric sRGB triples in text, even for large text‑only models. This is unlike human tasks where participants see colored patches. The authors show that using images does not improve performance for already strong models (Figure 8) and that CIELAB inputs actually degrade performance (Appendix C, Figure 6).  
   However, this raises several conceptual questions:
   - For instruction‑tuned models, there is likely uneven exposure to raw numeric RGB triples during training; many may instead learn color terms via naturalistic textual contexts (“red apple”, “sky is blue”, etc.), not numerical color codes.  
   - The failure on CIELAB suggests the models are not invariant to reparametrizations of color space, which complicates the cognitive claim that they “possess human‑like color categories”.  
   - For multimodal models, the finding in **Figure 8** that images sometimes hurt performance is intriguing, but the paper does not dig into *why* this might be and whether it is due to prompt design (e.g., mixing image and text) or to the visual encoders.  
   Overall, the experimental design is reasonable as a first step, but for a paper whose motivation is alignment with human color cognition, it would be good to more directly leverage the visual channel and discuss the representational mismatch more thoroughly.

6. **Limited quantitative analysis of uncertainty and variability.**  
   Much of the main‑text evidence for the English naming task is presented without uncertainty estimates. **Figure 2(a)** and **Figure 2(c)** plot single points per model, but:
   - There are no multiple runs per model, no random‑seed or decoding‑noise variance, and no error bars. For smaller or base models, the difference between “unordered mess” and a coarse but regular partition could be within the variance induced by sampling with temperature 0.6 and top‑p 0.9.  
   - In the IICLL experiments, **Figure 3** shows trajectories for multiple chains, which gives some sense of variability, and **Figure 4** includes 95% confidence intervals across chains. However, the English naming plot lacks any comparable quantification.  
   This makes it difficult to assess how robust the ranking of models in Figure 2(c) truly is, especially if small metric differences determine interpretive claims like “model X is closer to English than model Y”.

7. **Shepard circles experiment is too preliminary for the strength of the claims made.**  
   Section 4.3 and **Figure 5** show only a handful of IICLL chains for Gemini on a single $k=4$ condition, without IB analysis, human comparison, or systematic metrics. Yet the Discussion uses this as evidence that LLMs “potentially have a domain‑general bias to organize features into non‑arbitrary, and increasingly regular, semantic categories”.  
   As presented, this is closer to an anecdotal case study than a robust result. At minimum, some quantitative measure (e.g., mutual information between latent variables and labels, cluster compactness) and/or comparison to human data from Carr et al. (2020) would be needed to support domain‑general claims.

8. **Related work on LLMs, IB, and semantic categorization is incomplete.**  
   The paper does a good job covering IB + color + human IL literature, but several directly relevant recent works on LLMs and semantic efficiency or concept structure are not cited or discussed, suggesting that the positioning of the contributions could be improved (see “Potentially Missing Related Work” below).

9. **Scope of claims relative to analyzed domains and languages.**  
   All rigorous IB analyses are restricted to color naming, and mostly to comparison with English and WCS languages. While color is a uniquely data‑rich domain, conclusions like “IB‑efficiency may emerge to support intelligent behavior” feel very broad given that only one semantic domain and one language’s IL dynamics (English) are used. The authors acknowledge this in the Discussion, but still phrase the main takeaway quite strongly. Some tempering of the general claims would improve the scientific balance.

---

## Potentially Missing Related Work

Below are directly related works that appear not to be cited in the paper and should be discussed:

1. **Mukherjee et al., “Large Language Models Estimate Fine‑Grained Human Color‑Concept Associations”, 2024.**  
   This work studies how LLMs capture human color–concept associations in a fine‑grained way, highly relevant to Section 2.1 and the English color naming experiment. It should be cited in the background on color and LLMs, near Abdou et al. (2021), Patel & Pavlick (2022), and Marjieh et al. (2024), and contrasted with the present work’s focus on IB efficiency and cultural evolution rather than association strength.

2. **Sun et al., “Concept Bottleneck Large Language Models”, 2025.**  
   This paper introduces concept bottleneck architectures for LLMs, connecting explicit concept representations and downstream behavior. Given that this submission also discusses semantic bottlenecks and IB‑style compression of meaning to words, it is highly relevant conceptually. It should be mentioned in Section 2.2 or the Discussion when talking about interpretable or bottlenecked representations in LLMs.

3. **Huang et al., “Traceable and Explainable Multimodal Large Language Models: An Information‑Theoretic View”, 2025.**  
   Presents an information‑theoretic analysis of multimodal LLMs. Since this paper also employs information‑theoretic tools (IB) and includes multimodal models (Gemini, Qwen‑VL), it would be natural to connect to Huang et al. in Section 2.2 or the multimodal analysis in Section 4.1.

4. **Tan et al., “Vision LLMs Are Bad at Hierarchical Visual Understanding, and LLMs Are the Bottleneck”, 2025.**  
   Studies limitations of vision‑LLMs in forming structured visual representations. This is relevant to the surprising finding that image input does not help and sometimes hurts color naming (Figure 8) and to the Shepard circle results. It should be discussed in Section 4.1 or 4.3 when interpreting multimodal model behavior.

5. **Bhatt, “Predictive Coding and Information Bottleneck for Hallucination Detection in Large Language Models”, 2026.**  
   Applies IB‑style reasoning directly to LLM behavior in a different context (hallucination detection). This is relevant to the Discussion’s broader claim about IB‑efficiency emerging in LLMs. A brief comparison in Section 2.2 or 5 would help situate this work within the growing literature on IB and LLMs.

6. **Chen et al., “Discreteness and Systematicity Emerge to Facilitate Communication in a Continuous Signal‑Meaning Space”, 2024.**  
   Investigates the emergence of discrete categories in communication systems. It is conceptually aligned with the cultural evolution and emergent category structure studied here via IICLL. It should be connected to the IL/NIL literature in Section 2.3 or the discussion of emergent communication (currently citing Chaabouni et al., Tucker et al., Gualdoni et al.).

7. **Peng et al., “Human‑Guided Complexity‑Controlled Abstractions”, 2023.**  
   Explores how abstractions with controlled complexity arise with human guidance. This is relevant to the paper’s focus on complexity–accuracy tradeoffs and near‑optimal abstractions. It could be discussed in the IB background or Discussion as related evidence that human‑preferred abstractions often lie near IB‑style tradeoffs.

8. **Maldonado et al., “Evidence for a Language‑Independent Conceptual Representation of Pronominal Referents”, 2023.**  
   While Zaslavsky et al. (2021) on person systems is cited, this follow‑up work provides more evidence on language‑independent conceptual representations and efficient coding in pronouns. It should be added alongside the pronoun work in Section 2.2 as another example of IB‑efficient semantic systems.

9. **Eisape et al., “Toward Human‑Like Object Naming in Artificial Neural Systems”, 2023.**  
   Directly examines object naming behavior in neural models. It parallels this paper’s concern with human‑like naming behavior, though in a different modality and domain. It should be mentioned when discussing prior work on emergent naming systems in artificial agents in Section 2.3 or in the Introduction.

---

## Questions

1. **Details of IB computation and efficiency loss.**  
   - How exactly are $I_q(M;W)$ and $I_q(W;U)$ estimated from the empirical $q(w|m)$ obtained from LLM naming outputs? Are you using plug‑in estimators, smoothing, or something else?  
   - What is the numerical procedure (and parameter settings) for computing the IB bound $\mathcal{F}_\beta^*$ and optimal encoders, and what is the grid or range of $\beta$ used when computing $\varepsilon$?  
   A more explicit description (possibly with pseudocode) would significantly increase confidence in the quantitative claims.

2. **Clarification on $\beta$ range and consistency between main text and appendix.**  
   The main text uses $\beta\ge 0$ in Eq. (1) while Appendix Eq. (4) uses $\beta\ge 1$. Which constraint is actually used in computations, and does changing the lower bound affect any of the reported efficiency losses or positions in Figures 2–4?

3. **Robustness of English naming results to decoding parameters and prompting.**  
   Have you tried alternative decoding setups for open‑weight models, such as deterministic argmax over allowed labels (temperature 0, top‑p=1) or different temperatures? If so, do the rankings in Figure 2(c) and the qualitative mode maps in Figure 9 remain similar? If not, could you comment on how sensitive your findings are likely to be to these choices?

4. **Role of the sliding history window in IICLL.**  
   Beyond preventing degeneracy as shown in Figures 14–15, does varying the window size (e.g., from 0 to 50) change the typical *complexity* of the final systems or their alignment to WCS/IB? Any quantitative characterization here would clarify whether the history window is merely stabilizing or also shaping the emergent inductive bias.

5. **Interpreting the failure of CIELAB inputs.**  
   You show in Figure 6 that all models struggle when prompted with CIELAB triples. Do you interpret this primarily as a limitation of the models’ exposure to these coordinates in training, or as evidence that the internal color representations are not aligned to perceptual metrics? Have you considered prompting with human‑readable approximations (“light bluish‑green”) derived from CIELAB to separate these possibilities?

6. **Shepard circles: can you add any quantitative measure?**  
   Even simple metrics such as mutual information between each latent dimension (radius/angle) and cluster labels, or cluster compactness, could strengthen the Shepard circles result in Figure 5. Is any such analysis available, and if not, could it be added?

7. **Extent to which initial random mapping biases the IICLL outcome.**  
   For the $k=14$ chains, you mention that most models quickly converge to low‑complexity solutions while Gemini can sometimes sustain more complex systems. How sensitive are these outcomes to the particular random initialization of $L_0$? Do some initializations systematically yield higher final complexity than others, or does the process wash out the initialization quickly?

Clear answers or additional analyses on these points could significantly increase my confidence in both the quantitative IB claims and the interpretation of IICLL as evidence of an intrinsic efficiency bias.

---

## Flag For Ethics Review

No ethics review needed.

---

## Details Of Ethics Concerns

N/A. The work uses publicly available datasets and commercial/open models via APIs, and does not raise obvious concerns about privacy, discrimination, or harmful deployment.

---

## Soundness Rating

3: good.  
The experimental design is generally solid, and the qualitative patterns (e.g., Figures 2–4) convincingly support the main claims, but some important methodological details about IB computation, decoding choices, and robustness are missing or under‑specified.

---

## Presentation Rating

3: good.  
The paper is well‑written, with clear figures (especially Figures 2–5) and a strong narrative connecting IB theory, human data, and LLM behavior. However, some key technical details are relegated to prior work or omitted, and the related work on LLMs + IB could be more complete.

---

## Contribution Rating

3: good.  
The combination of IB analysis with a large‑scale LLM color‑naming study and the introduction of IICLL to mimic human iterated learning is a meaningful and relevant contribution. The work advances our understanding of LLM inductive biases and their relation to human semantic systems, even if some claims are a bit stronger than the data warrant.

---

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The paper offers a thoughtful, theory‑driven study that bridges cognitive science and LLM analysis, with substantial empirical work and several genuinely interesting findings (notably Gemini’s near‑optimal IB behavior under IICLL). At the same time, some methodological details and robustness checks are missing, the generality claims are slightly overstated, and the Shepard circles extension is thin. On balance, I see this as a solid and valuable contribution that deserves to be part of the discussion at ICLR, but it is not without weaknesses.

---

## Reviewer Confidence

4: confident.  
I am familiar with the IB literature and iterated learning, and I have carefully examined the equations, figures, and experimental protocols. Some of the claims depend on implementation details not fully specified in the main text, which prevents a “5”, but my overall assessment is unlikely to change drastically.
# Internal Planning in Language Models: Characterizing Horizon and Branch Awareness

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
The extent to which decoder-only language models (LMs) engage in planning, that is, organizing intermediate computations to support coherent long-range generation, remains an important question, with implications for interpretability, reliability, and principled model design. Planning involves structuring computations over long horizons, and considering multiple possible continuations, but how far transformer-based LMs exhibit them without external scaffolds, e.g., chain-of-thought prompting, is unclear. We address these questions by analyzing the hidden states at the core of transformer computations, which capture intermediate results and act as carriers of information. Since these hidden representations are redundant and encumbered with fine-grained details, we develop a pipeline based on vector-quantized variational autoencoders that compresses them into compact summary codes. These codes enable measuring mutual information and analyzing the computational structure of the underlying model behavior. Using this framework, we study planning in LMs across synthetic grammar, path-finding tasks, and natural language datasets, focusing on two planning properties: (i) the planning horizon of pre-output computations, and (ii) the extent to which the model considers alternative valid continuations. As a separate downstream use of the same pipeline, we also analyze how decision-relevant information is distributed across layers and earlier prefix blocks when producing next-token predictions. Together, these analyses advance our understanding of planning in LMs and provide a general-purpose pipeline for inspecting internal model dynamics. Our results reveal that the effective planning horizon is task-dependent, that models implicitly preserve information about unused correct continuations, and that predictions draw most on recent computations, though earlier blocks remain informative.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates whether decoder-only language models (LMs) engage in internal planning, defined along three axes: forward-looking (planning beyond the next token), branch-aware (considering multiple valid continuations), and stateful (reusing earlier computations). The authors propose an information-theoretic framework that compresses high-dimensional hidden states into discrete codes using a modified Vector-Quantized Variational Autoencoder (VQ-VAE), enabling mutual information (MI) estimation between different parts of the model’s internal computations. They apply this pipeline across synthetic (CFG, path-finding) and natural language (OpenWebText) tasks, finding that planning behavior is task-dependent. Stronger evidence of non-myopic computation is observed in structured reasoning tasks than in syntactic or natural language settings.

### Strengths
- The paper is well-motivated. Previous studies on the internal mechanisms of LMs often rely on linear probing, which is susceptible to confounders and may lead to unreliable conclusions. The proposed method avoids these confounders, making the analysis of internal planning in LMs more reliable.
- The use of mutual information over compressed discrete codes provides a principled and scalable approach to analyzing high-dimensional activations.
- The experimental design covers a thoughtful range of tasks—from highly structured symbolic problems to natural language—allowing for nuanced insights into when and how planning emerges.
- The validation experiment in Appendix A.3 reasonably demonstrates that the VQ-VAE compression preserves meaningful statistical dependencies.

### Weaknesses
- Overall, the empirical findings are somewhat incremental. For instance, the observation that LMs plan more in path-finding tasks and exhibit myopic behavior in CFG tasks is somewhat expected. Additionally, the proposed framework does not offer a mechanistic explanation of how such planning is implemented, which limits the significance of the findings.
- In my view, using VAE-based methods also has drawbacks compared to linear probing:
  - Interpretability is relatively poor. Since the latent space of VAEs is inherently difficult to interpret, even if changes in MI are detected, it remains challenging to explain the underlying mechanisms behind these changes.
  - The VQ-VAE introduces its own set of hyperparameters and design choices (e.g., codebook size, cosine penalty), and the sensitivity of the results to these choices is not thoroughly explored.
- The reliance on normalized MI makes it difficult to assess the absolute strength of planning signals. Moreover, the paper does not convincingly demonstrate that the observed MI reflects causal planning rather than passive statistical correlation.

### Questions
- I am somewhat confused about the authors’ definitions of planning properties: "Stateful" involves reusing earlier computations, but isn’t this also a manifestation of being "forward-looking"? Could the authors provide more intuitive examples to clarify the distinction between these two concepts?
- Regarding the use of VAE for compressing information to obtain distributions:
  - Could the authors briefly explain why this method is unlikely to introduce external information? Is it because the two positions being compared are compressed by two independent VAEs? Is it possible that the learned encoder and codebook inadvertently encode external information, leading to misleading conclusions?
  - Could the same MI patterns arise in a model that does not "plan" in any meaningful sense?
- Based on the observed patterns of LM planning across different tasks, could the authors propose any improvements to the LM architecture or algorithms, such as a blueprint for enhancement?
- If earlier blocks retain "nontrivial information," as claimed, why doesn’t ablating or perturbing them significantly degrade performance? Such a test could potentially strengthen the argument for statefulness.

### Soundness
3

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
For interpretability purposes, it is useful to be able to estimate the mutual information (MI) between two distinct blocks of activations in a transformer model. The authors propose to do this by quantizing these activations using a specially trained Vector-Quantized Variational Autoencoder (VQ-VAE), then measuring the MI between the resulting discrete codes. This novel interpretability technique is then used to measure the presence of planning in transformer models in both toy settings and natural language.

### Strengths
- As far as I'm aware, the use of VQ-VAE to estimate the mutual information between high-dimensional continuous activations is novel.
- Several distinct aspects of planning are investigated in both toy settings and natural language.
- The authors acknowledge that their absolute MI estimates may be misleading, and focus on relative comparisons.
- The authors provide a basic sanity check to validate their MI via VQ-VAE technique (A.3).

### Weaknesses
1. I am not fully convinced by the authors' philosophical argument that MI via VQ-VAE is more appropriate than existing linear probe techniques. 
	- To fix notation, say that we care about two model activations $h_1,h_2\in\mathbb{R}^d$. We can either train two VQ-VAE encoders $E_1,E_2$ and estimate the mutual information $I(E_1(h_1);E_2(h_2))$, or we can train a probe $\phi\colon\mathbb{R}^d\to\mathbb{R}^d$ and measure the $\ell_2$ loss $||\phi(h_1)-h_2||_2^2$. 
	- My understanding is that *both* of these, up to some constant factor, are valid lower bounds for the quantity of interest $I(h_1; h_2)$. The former is a lower bound by the data processing inequality, while the latter is by [1, Prop 1.5].
	- The extent to which these bounds are loose is due in the former case to the information discarded by $E_1,E_2$, and in the latter case to the inability of the probe architecture to represent the true conditional probability $P(h_2\mid h_1)$. It's not clear to me why the former approach is better or more principled than the latter.
	- I'm not sure I understand what is meant by the "confounding effect" of learned probes mentioned in the introduction. Both the VQ-VAE approach and the probing approach require training separate auxiliary models, and the quality of the lower bound depends on the representational capacity of the auxiliary model in either case.
	- I would argue that in the neural network setting, *both* approaches are somewhat philosophically dubious. It is often the case that $h_2$ is a deterministic function of $h_1$ (for example, in the setting of Section 3.1, where $h_1=h_{1:T}^{1:L-1},h_2=h_{T+\tau}^L$, this is true if sampling temperature is zero). In this case $I(h_1;h_2)=H(h_2)$, telling you nothing about the relationship between $h_1$ and $h_2$.
2. Experimental results have no comparisons with baselines such as linear probes.
	- Since the theoretical/philosophical discussion is somewhat unclear, an empirical comparison would give much stronger evidence for the suitability of MI via VQ-VAE vs probes for interpretability.
	- As it stands, the paper applies an unproven method to novel toy settings without established baselines. Without proper baselines, it is hard to be confident that these results are correct or meaningful.
3. MI between past and future model activations is a fairly weak information-theoretic condition, and is e.g. always present in the zero-temperature autoregressive setting. In my opinion, "planning" is a somewhat misleading term for this, as it implies intentionality in the model which may not really be present.

 [1] Xu et al. "A theory of usable information under computational constraints". ICLR 2020.

### Questions
- How canonical are the three aspects of planning you study here (forward-looking, branch-aware, stateful)? Are you claiming that possessing all three properties is either necessary or sufficient for being a good planner?
- One advantage of linear probes is that they are cheap and simple to implement. How expensive is the VQ-VAE to train? Does it scale to the kinds of LLMs that may be used in practice (billions of parameters)?
- What temperature was used for sampling in the experiments?
- I strongly suggest adding a comparison to linear probe baselines. I.e., wherever $I(E_1(h_1), E_2(h_2))$ is measured, instead fit a linear probe from $h_1$ to $h_2$ and report $R^2$.
- In the introduction, I recommend fleshing out the discussion of MI vs probes and citing [1].
- In "training objectives: next-token vs multi-token", I recommend citing [2], which shows that cross-token gradients induce forward planning even with a next-token objective.
- I like the sanity check performed in A.3, but having a discrete domain with support only 0-1 OOMs larger than codebook size is maybe too artificial. Are there continuous distributions for which the theoretical MI can be computed and compared with the estimated MI via VQ-VAE? Alternatively, what if the support is exponentially larger than codebook size (which is what happens in practice, because there are exponentially many possible input sequences)?

[1] Xu et al. "A theory of usable information under computational constraints". ICLR 2020.

[2] Wu et al. "Do language models plan ahead for future tokens?" COLM 2024.

### Soundness
2

### Presentation
2

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
The paper examines whether decoder-only language models exhibit planning behavior. It formalizes planning through three abilities: forward-looking horizon (how far future information is encoded), branch awareness (whether multiple plausible continuations are internally represented), and statefulness (how much earlier computations influence later predictions). Using an information-theoretic framework, the authors compress hidden states with a VQ-VAE and compute normalized mutual information between prefix and future representations to quantify these properties. Experiments across grammar, path-finding, and natural text tasks show that planning behavior increases with task complexity, while models remain mostly short-sighted and multi-token prediction provides only minor gains.

### Strengths
Please find the strengths below:
1. The paper provides a novel perspective by examining whether decoder-only language models possess planning capabilities from an information-theoretic viewpoint, making the study a fresh and meaningful contribution to the understanding of planning in LMs.
2. The use of VQ-VAE to discretize hidden states and compute normalized mutual information across layers offers a new theoretical approach for analyzing information flow, which could potentially be applied to other research problems.
3. The paper presents intuitive figures and maintains a clear logical flow, helping readers grasp the key ideas and results effectively.

### Weaknesses
Please find the weaknesses below:
1. Intuitively, mutual information captures only statistical correlations rather than causal relationships, and the paper does not provide theoretical justification that higher MI truly reflects stronger planning or reasoning ability; this assumption may thus be overly simplified.
2. Although the experiments cover diverse tasks, they mainly rely on small-scale models and synthetic settings, without evaluation on more recent large language models. In addition, the VQ-VAE discretization process may introduce information loss or noise, yet the paper lacks a rigorous analysis or quantification of its impact on measurement accuracy.
3. The multi-token prediction objective yields only marginal improvements, suggesting that the proposed framework, while informative, offers limited practical gains in enhancing model performance.

### Questions
The questions are related to the weaknesses:
1. Can the authors provide a more rigorous theoretical analysis linking mutual information to actual planning or reasoning ability, rather than relying on intuitive correlation?
2. Could future experiments include larger-scale and more capable reasoning models to test whether the observed phenomena generalize beyond small synthetic settings?
3. Can the proposed framework offer more concrete guidance or insights on how to enhance models’ reasoning or planning capabilities?
4. Could the authors further analyze how the framework applies to reasoning behaviors under Chain-of-Thought (CoT) or Tree-of-Thought (ToT) prompting settings?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies planning capabilities of LLMs through an information-theoretic lens. The authors develop a framework using Vector-Quantized Variational Autoencoders (VQ-VAE) to compress the hidden states of the LLM into discrete codes. This allows them to compute mutual information (MI) estimates between different "computational blocks"/hidden states. 
Using this approach, they analyze three key aspects of planning: (i) the planning horizon (how far ahead models plan before generating tokens), (ii) branch awareness (whether models internally represent alternative valid continuations), and (iii) computational history dependence (which earlier computations inform current predictions). 
The framework is evaluated across synthetic grammar tasks, path-finding problems, and natural language (OpenWebText), comparing next-token prediction (NTP) versus multi-token prediction (MTP) training objectives.

### Strengths
- I think the idea behind the paper is quite novel: thee use of information-theoretic measures via VQ-VAE compression to study planning is a creative methodological tool

- Also, the specific focus on quantifying planning through three different characteristics is exciting. Namely, via horizon, branching, and history dimensions. This provides a more structured framework for understanding LM internal computations. 

- The experimental analysis is comprehensive, including controlled synthetic tasks (CFG, path-finding) to natural language. 

- The path-finding task with disjoint correct and decoy paths is particularly well-designed for testing branch awareness. I really liked the idea. 

- This work addresses an important open question about whether and how LMs engage in planning-like computations. The finding that planning behavior is task-contingent has implications not only for model design, but also for training strategies. 

- I believe the framework itself could also be valuable for broader interpretability research, not just planning.

### Weaknesses
While I really enjoyed this paper, I think it also has room for improvement:

- The main text simply says "we train a VQ-VAE", but since this is a key aspect of the method, I would expect more information about it in the main text. However, crucial aspects are only in the Appendix. 

- Given the details in the appendix, the number of hidden states to store is huge, which suggests that the framework is non-trivial since the VQVAE is difficult to train. 

- The experiments were carried using only GPT-3 Small models (12M-202M parameters). Findings may not generalize to larger models where emergent capabilities differ. It also becomes more difficult to apply the framework for larger models.

- The core pipeline (freeze LM -> sample block -> train VQ-VAE -> quantize -> estimate MI -> ...) is convoluted. A figure summarizing the pipeline/steps of the full framework would help a lot. Right now the reader keeps jumping between sections in the main text and appendix to figure out what is the full framework.

### Questions
- For each experiment, do you train one VQ-VAE per dataset/task (e.g. one for CFG prefix, one for PF prefix, one for PF paths, one for OpenWebText blocks), or per model checkpoint?

- Roughly how many sequences and optimization steps were needed to train your VAEs?

- What are the overall conclusions for NTP versus MTP?

### Soundness
3

### Presentation
2

### Contribution
3

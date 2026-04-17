---
job_id: 873b2c1d-c5cb-406b-9325-83271239c51b
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: USyGD0eUod.pdf
paper: Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper studies sparse autoencoders, interpretability metrics, and random vs trained transformers, which fits squarely under “visualization or interpretation of learned representations” and “sparse coding / representation learning” for ICLR.

## Minimum Quality
Pass ✅.  
The paper has all key sections (Abstract, Introduction, Related Work, Results/Experiments, Toy Models/Methodology, Limitations, Conclusion) plus extensive appendices. The work is technically correct as far as can be checked, uses standard methodology, and presents substantial empirical evidence across multiple model sizes. Weaknesses exist but are not at the “fatal error” level.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not detect any instructions or content attempting to manipulate automated reviewing; the text reads like a normal research paper.

---

# Expected Review Outcome:

## Summary

The paper evaluates sparse autoencoders (SAEs) trained on residual stream activations of Pythia transformers under several conditions: fully trained models, various randomization schemes (including re-randomized weights with and without embeddings, and step-0 initialization), and a strong “control” where token embeddings are replaced with fresh Gaussian noise at inference.  

Using automatic neuron-description pipelines (mainly the “fuzzing” AUROC metric of Paulo et al. 2024) and standard SAE metrics (explained variance, cosine reconstruction similarity, L1 norms, cross-entropy loss recovery), the authors find that SAEs trained on randomized transformers often achieve aggregate scores similar to those on trained transformers, and much better than the Gaussian-control. They further propose a simple token-distribution entropy metric and toy models of superposition to argue that existing aggregate metrics fail to capture feature “abstractness,” and that random networks and/or embeddings may already exhibit substantial sparse structure.

## Strengths

1. **Important sanity-check question, directly targeted at current practice.**  
   The core question, “Do standard SAE metrics and auto-interpretability scores actually distinguish trained from random transformers?”, is sharply posed, highly relevant to the mechanistic interpretability community, and rarely tested with serious null models. The results highlight a potentially uncomfortable fact: widely reported auto-interpretability scores alone are not strong evidence of learning meaningful computational structure.

2. **Systematic experimental design across multiple randomization schemes and model scales.**  
   The paper goes beyond the trivial “trained vs random weights” comparison by introducing several nuanced baselines (Section 3): re-randomized including embeddings, re-randomized excluding embeddings, step-0 initialization, and a harsh Gaussian-embedding control. Evaluating all of these across Pythia models from 70M to 6.9B parameters, with per-layer SAEs, gives a much richer picture.  
   - **Figure 2** is particularly compelling: across five model sizes and eight metrics (explained variance, cosine similarity, L1 norm, AUROCs for fuzzing and detection, CE loss score, token-entropy), trained and randomized variants track each other closely on most metrics, while the control cleanly separates. This figure alone delivers a very strong empirical argument that common metrics are blind to whether the underlying transformer was actually trained.

3. **Careful use of automatic explanation pipelines and controls.**  
   The auto-interpretability setup (Page 4–5) is faithful to current best practice (Bills et al., Paulo et al., Choi et al.), using fuzzing and detection AUROCs from Meta-Llama-3.1-70B as explainer/simulator and sampling 100 features per layer.  
   - **Figure 1** (ROC curves across layers for Pythia-6.9B) clearly shows near-identical ROC curves for trained and randomized variants, with only the control near chance. This is a clean sanity check: if the entire pipeline were nonsense, all curves would sit near chance; instead, the pipeline clearly reacts to the Gaussian-noise control, but not to training vs randomization.

4. **Thoughtful, if preliminary, attempt to quantify feature “abstractness”.**  
   The token-distribution entropy metric (Pages 5–6) is simple but insightful: it measures how concentrated each latent’s activations are on specific token IDs.  
   - In **Figure 2**, the entropy row shows trained models’ entropy rising with layer index, while randomized variants stay low and the control stays high.  
   - **Figure 20** (Appendix H) sharpens this: for randomized models there is a clear negative correlation between entropy and fuzz AUROC (only simple, single-token features are well-explained), while the trained model uniquely exhibits high-entropy, high-AUROC latents. This is strong evidence that trained models contain genuinely more abstract, multi-token concepts that aggregate metrics simply average away.

5. **Nontrivial toy-model analysis of superposition and random networks.**  
   Section 4 is more than hand-waving: it reuses the Sharkey et al. toy superposition setup, verifies that SAEs can recover ground-truth features (Figure 4), and then uses Pareto frontiers over explained variance vs sparsity (L1 penalty) to compare SAEs on superposed vs Gaussian datasets, before and after passing through a random MLP.  
   - **Figure 5a–b** suggest that random MLPs “sparsify” inputs in a way that narrows the gap between superposed and Gaussian controls. This supports the authors’ speculation that a lot of the sparse structure exploited by SAEs may be induced or amplified simply by random linear / ReLU transformations.

6. **Good use of robustness and ablation checks.**  
   The appendices include several robustness checks that increase trust in the empirical story:
   - Training SAEs with 1B vs 100M vs 1M tokens (Figures 15–16) to show that the similarity between trained and randomized variants is not a data-scarcity artifact.
   - Varying SAE hyperparameters (expansion factor and k) for multiple models (Figures 18–19) shows that auto-interpretability AUROCs remain similar even when reconstruction metrics distinguish underpowered SAEs.
   - Multiple seeds at 70M (Figure 17) give uncertainty bands, showing the main patterns are not due to single runs.

7. **Clarity and transparency of experimental details.**  
   The methodology is described sufficiently to reproduce the setup: Pythia models and step-0 revisions, RedPajama data, activation buffers, TopK SAEs with specific R and k, use of sparsify/delphi toolkits, and compute budget in **Table 1**. The inclusion of many concrete feature examples (Section J, Section L) substantively supports the qualitative claims about token-level vs abstract features.

## Weaknesses

1. **Limited scope of claims vs breadth of experiments: focus almost entirely on one SAE family and one model/dataset family.**  
   The conclusions are framed quite broadly (“automated interpretability metrics do not distinguish trained and random transformers”), but the experiments are relatively narrow:
   - All main results are on Pythia language models and RedPajama text; there is no exploration of other architectures (e.g., GPT-2, decoder-only non-Pythia, multi-modal transformers, or non-LM settings such as protein or vision models).  
   - All main SAEs are TopK (k-sparse) autoencoders with the same training objective; other popular variants (Gated SAEs, JumpReLU, Jacobian SAEs, switch SAEs) are only referenced, not tested.  
   This mismatch matters because their central conclusion is about the *metrics*, not about TopK SAEs specifically. It remains an open question whether e.g. Jacobian SAEs or concept-bottleneck SAEs evaluated with more causal metrics would behave similarly under randomization. At minimum, the paper should be more explicit that the empirical claim is about *TopK decoder-based SAEs on Pythia LMs*, not SAEs in general.

2. **Causal significance of features is not probed; evaluation remains correlational.**  
   The argument that SAEs on random transformers “do not capture learned computation” leans on model intuitions, but is never tested via interventions on the original models. All interpretability evaluation is based on:
   - Reconstruction metrics (explained variance, cosine similarity) and  
   - External LLM-based classifiers/simulators (fuzzing / detection AUROCs).  
   Crucially, no experiments use causal metrics that are standard in mechanistic interpretability, such as activation patching, feature steering, or targeted concept erasure on the underlying model. For example, it would be compelling to show that features with similar fuzz AUROCs in trained vs random transformers differ sharply in their ability to steer model outputs or to affect loss when ablated.  
   As it stands, the paper demonstrates that *correlation-based* metrics do not distinguish trained and random nets, but leaves the door open that more causal metrics would behave differently; this weakens the strongest versions of the paper’s claims.

3. **Toy models in Section 4 are only weakly tied to the main transformer results.**  
   While the toy analysis is thoughtful, the link to real SAEs on Pythia is mostly qualitative. Several issues stand out:
   - In Section 4.2, the use of **explained variance vs sparsity Pareto fronts** as an implicit superposition measure is reasonable, but the precise choice of sparsity metric, $L^{1}/\sqrt{L^{2}}$, is somewhat ad hoc and not justified beyond “following Sharkey et al.”. Other sparsity metrics are relegated to the appendix; no single scalar or statistical test is used to compare input vs output distributions in Figure 5.  
   - In Section 4.1, the statement that matrix multiplication “preserves superposition” relies on the generative model $x \sim \mathcal{N}(Dz, \Sigma), \ x' = Wx \Rightarrow x' \sim \mathcal{N}(WDz, W\Sigma W^\top)$, but this is only algebraic closure of a Gaussian mixture; it does not quantify *sparsity* or *identifiability* of the latent $z$. In particular, the heavy-tailed Lomax prior on $z$ is not used in the derivation, and the paper gives no theoretical condition under which SAEs trained on $x'$ can still recover features aligned with $z$.  
   - The toy networks are shallow two-layer MLPs; transformers with residual connections, attention, and layernorm could behave differently. The paper stops short of demonstrating that the degree of “sparsification” observed in Figure 5 is sufficient to explain the near-identical AUROCs in Figures 1–2.
   Overall, Section 4 would be stronger if it either (a) provided more direct quantitative predictions that could be checked on Pythia activations (e.g., specific expected sparsity statistics), or (b) was clearly positioned as speculative and decoupled from the main empirical claim.

4. **Statistical analysis of auto-interpretability metrics is relatively thin.**  
   For each SAE, the authors sample 100 latents for auto-interpretability and report layer-wise mean AUROCs, often using a single run. Uncertainty is plotted only for Pythia-70M (Figure 17); other models show only mean lines. This raises several issues:
   - We do not see distributions of AUROCs across latents per condition; in particular, it would be useful to know whether trained models have a longer right tail (few very well-explained features) that might be hidden by averaging.  
   - There is no formal statistical test comparing trained vs randomized variants; the plots suggest similarity, but e.g. a small but systematic gap in later layers could still be meaningful.  
   - The choice of 100 sampled features might under-sample rare high-quality latents in trained models, especially in larger SAEs (e.g., Pythia-6.9B with R=64).  
   Without more rigorous uncertainty quantification or distributional analysis, it is difficult to tell whether “similar” curves in Figures 1–2 are truly indistinguishable or just overlapping enough to look visually similar.

5. **Interpretation of the control baseline and its implications.**  
   The Gaussian-embedding control is crucial: it nicely demonstrates that the auto-interpretability pipeline is not trivially fooled by any input. However, its extremity also complicates interpretation:
   - In this control, each token instance is mapped to an independent Gaussian vector, so there is *no* consistent embedding for a token across occurrences. It is unsurprising that SAEs trained on such activations struggle to reconstruct or be interpreted (Figure 2 rows 1–3 and the entropy row).  
   - This makes the gap between control and randomized variants very large, potentially overshadowing smaller, but still meaningful, differences between trained and randomized transformers.  
   A more nuanced control, such as permuting token embeddings across the vocabulary or random rotations of the embedding matrix, might have provided a more informative baseline that “breaks semantics” without completely erasing token consistency.

6. **Token-distribution entropy is only partially validated as a measure of “abstractness”.**  
   While I like the idea, several concerns remain:
   - The metric is inherently tied to tokenization and dataset frequency; a feature that fires on rare but semantically coherent phrases might have similar entropy to one firing on frequent function words.  
   - There is no direct human-annotation validation that higher entropy corresponds to more abstract or multi-token concepts; the paper mostly relies on qualitative examples and heuristic interpretation of **Figure 20**.  
   - Entropy is computed only on the top activating examples used for auto-interpretability prompts, not on the full dataset, which may bias the estimate towards whatever tokens were selected by the SAE training dynamics.  
   The metric is a promising direction, but the paper somewhat overstates its success (“proof-of-concept that successfully revealed differences”) given the limited validation.

7. **Some missing or under-discussed related work, especially on alternatives to SAEs and on sparsity vs interpretability.**  
   The Related Work section is solid on SAEs and auto-interpretability, but omits some recent, closely aligned work:
   - Approaches that compare SAE features to transformer key-value memories or other internal structures, or highlight that sparsity does not guarantee interpretability, are not discussed (see detailed list below).  
   - There is limited discussion of concept-bottleneck SAEs and causal interpretability methods that purposefully enforce semantically labeled latents.  
   This omission marginally weakens the positioning of the paper’s claims about “SAEs and automated metrics” in the broader interpretability landscape.

8. **Minor mathematical and clarity issues.**  
   - In Section 4.1, the notation $x \sim \mathcal{N}(x; Dz, \Sigma)$ and $x' \sim \mathcal{N}(z; WDz, W\Sigma W^\top)$ is slightly confusing: the Gaussian’s argument is usually the random variable, not repeated; writing $x \sim \mathcal{N}(Dz, \Sigma)$ and $x' \sim \mathcal{N}(WDz, W\Sigma W^\top)$ would be clearer. Also, as noted above, the Lomax prior on $z$ is mentioned but not used in the derivation; more explicit conditions under which superposition is preserved would help.  
   - In Appendix I.1, the data-generation algorithm is mathematically dense and only partially motivated. For example, the mapping $\alpha_i \to \alpha_i^{\lambda i}$ and normalization $\alpha_i \to m \alpha_i / (n_s \sum_j \alpha_j)$ could be briefly justified in probability terms; currently it reads like code without a clear mathematical rationale.  
   These are not fatal, but they do make it harder to fully grok the toy model setup.

## Potentially Missing Related Work

1. **Ye & Suzuki & Inaba, “Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders” (2025).**  
   This paper directly compares the interpretability of transformer key–value memories to SAEs, arguing that much structure can already be read off without dictionary learning. It is highly relevant to the claim that SAEs *on random networks* might be exploiting architectural or input-induced structure. It should be discussed in Section 2 (Mechanistic interpretability) and contrasted to this work’s finding that SAEs on random weights still look “interpretable” under aggregate metrics.

2. **Simon & Zou, “InterPLM: Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders” (2025).**  
   This applies SAEs to a *non-text* domain (protein LMs) and studies interpretability there. It is relevant as an example of SAEs in a different modality; the current paper’s conclusions may or may not transfer. It would fit well at the end of Section 2 (“Mechanistic interpretability”) and in the Conclusion when discussing generality beyond language models.

3. **Zhang, “Sparse but not Simpler: A Multi-Level Interpretability Analysis of Vision Transformers” (2026).**  
   This work examines the relationship between sparsity and interpretability in vision transformers and argues that sparsity alone does not guarantee simpler or more interpretable representations. It should be mentioned in Section 2 under “Mechanistic interpretability” or “Polysemanticity”, and could strengthen the argument in Section 6 that high sparsity / SAE scores are not sufficient evidence of meaningful learned computation.

4. **Kulkarni, Weng & Narayanaswamy, “Interpretable and Steerable Concept Bottleneck Sparse Autoencoders” (2025).**  
   This introduces concept-bottleneck SAEs that explicitly enforce interpretable and steerable latents. It is closely related to the discussion in Section 3 and Section 6 about needing better metrics for “computationally significant” features. It should be cited when discussing alternatives and potential future directions for evaluation beyond aggregate auto-interpretability.

## Questions

1. **Distribution-level differences between trained and randomized models.**  
   Figures 1 and 2 show similar mean AUROCs. Could you provide histograms or CDFs of fuzzing AUROCs for individual latents (perhaps in a subset of layers) to check whether trained models have a heavier right tail, even if means are close? If such tails exist, how would that affect your conclusions?

2. **Causal relevance of features.**  
   Have you tried any simple causal intervention experiments (e.g., activation patching with SAE reconstructions, or toggling individual high-AUROC features) to test whether features with similar auto-interpretability in trained vs random models differ in their ability to affect model loss or outputs? Even a small-scale study on one model/layer would significantly strengthen the central claim.

3. **Explainer model dependence.**  
   All auto-interpretability results use Meta-Llama-3.1-70B. Do you have evidence that similar patterns hold when using smaller explainers (e.g., 8B) or non-LLama models, especially given that Choi et al. fine-tune explainer models? Might some of the similarity between trained and random transformers be due to biases in the explainer rather than the SAE features?

4. **Sensitivity to SAE architecture.**  
   Section 3 focuses on TopK SAEs; have you run even limited experiments with a different SAE objective (e.g., L1 penalty without TopK, Gated SAEs, or Jacobian SAEs) on a subset of layers to check if random vs trained distinguishability changes? If such results exist, even in the appendix, it would help clarify whether the phenomenon is tied to the k-sparse objective.

5. **Clarification on the role of the Lomax prior in Section 4.1.**  
   The superposition argument uses a Lomax distribution for $z$ but the formal derivation seems to only exploit Gaussian closure under linear maps. Could you clarify whether the heavy tails of the Lomax prior play any role in guaranteeing “superposition preservation,” or whether they matter only empirically for the toy examples?

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The empirical methodology is careful and well-documented, with extensive baselines and robustness checks. The main claims are appropriately hedged. Some analyses (distributional statistics, causal relevance of features, and tighter linking of toy models to transformers) are missing or only qualitative, so I would not rate the soundness as “excellent,” but the current evidence solidly supports the main narrative.

## Presentation Rating

3: good.  
The paper is generally well-written and easy to follow despite a large volume of experiments. Figures 1–2 and 5 are especially informative, and the many qualitative examples in the appendices are helpful. A few mathematical derivations (Section 4, Appendix I) could be clarified, and the scope of claims vs experiment coverage could be more crisply stated.

## Contribution Rating

3: good.  
The work makes a meaningful contribution to the interpretability community by stress-testing widely used SAE evaluation metrics against strong random baselines, and by proposing a simple metric that captures an aspect of feature “abstractness” currently ignored by aggregate scores. It is not a new method, nor a deep theoretical analysis, but as a critical empirical study it is valuable and likely to influence how future SAE work is evaluated.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The paper delivers a well-executed and quite timely empirical critique of current SAE evaluation practice, with convincing figures (especially Figure 2 and Figure 20) showing that aggregate auto-interpretability scores cannot by themselves distinguish trained from random transformers. The work is carefully done and relevant to many ongoing projects. Its limitations are mainly in scope (one SAE family, one model/dataset family) and in the lack of causal tests of feature relevance, which keep it from a higher score but do not, in my view, undermine its core message.

## Reviewer Confidence

4: confident.  
I am familiar with the SAE and mechanistic interpretability literature and checked the math and experimental design in detail. Some aspects (e.g., specific explainer-model behaviors) are inherently hard to verify without rerunning, but they do not affect my main assessment.
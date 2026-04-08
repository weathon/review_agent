## Human Reviewer 1

### Summary
This paper proposes a new mechanistic interpretation of in-context learning: rather than copying label tokens via induction heads, large language models perform a task-oriented information removal that compresses query hidden states toward a low-rank, task-verbalization subspace (TVS). The authors demonstrate (1) that injecting a trained low-rank filter into a layer’s residual stream converts near-zero zero-shot performance into strong task-specific outputs; (2) that few-shot demonstrations induce geometric changes in hidden states — increased eccentricity and covariance flux into the learned TVS — consistent with implicit information removal; and (3) that a subset of attention heads  causally contribute to this aligning operation: ablating them reduces covariance flux, eccentricity, and downstream accuracy. The paper thus argues DHs complement induction heads and together explain a broader range of ICL phenomena.

### Strengths
The paper’s framing of the “information removal → task-verbalization subspace” is conceptually novel and shifts the focus from mere copying to selective suppression

The paper is well-written and easy to follow. The low-rank injection experiment is simple, interpretable, and effectively demonstrates that a small learned linear filter can steer outputs.


The identification of Denoising Heads (DHs) and the analysis of their interaction with induction heads provide a richer mechanistic understanding of in-context learning.

### Weaknesses
Ablating a head is a valid intervention, but zeroing a head output changes the residual stream and thus all downstream activations; the observed drops in Covariance Flux / Eccentricity / Accuracy may partly reflect these propagated network dynamics rather than a head performing a localized denoising operation.

The method appears to identify DHs only on a subset of layers. If many DHs remain unidentified, ablation results could be underestimated or misattributed. 


The DH identification uses fixed relative-change thresholds (e.g., −3.5% / −5%). It is unclear how sensitive results are to threshold choice and whether effects are statistically significant across random seeds / datasets / prompts.

### Questions
Can you show that amplifying DH outputs (scale >1) increases Covariance Flux/Eccentricity and improves accuracy in settings where the model is weak (e.g., few-shot with noisy labels)?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 2

### Summary
The paper proposes a novel interpretation of the ICL mechanism by viewing it as a process to remove redundant information in the query’s hidden states for the targeted tasks. Specifically, the paper applies a low-rank filter on the hidden states for informational removal and finds it boost the performance on the designed tasks significantly. On top of that, the paper conducts further analysis to prove that a few-shot ICL implicitly performs a similar information removal process. Finally, the paper identifies attention heads that are responsible for such removal behaviors and shows that they are crucial for the success of ICL.

### Strengths
- The paper includes well-designed illustrations that well support the claims.
- The paper provides a perspective for demystifying ICL that is novel yet aligns with many previous observations.
- To support the claim that ICL performs information removal, the paper provides decently comprehensive analyses that dives deep into the activities of the hidden states of the model using self-crafted analysis methods.
- The paper includes experiments and discussions to compare their newly discovered Denoysing Heads against the previous induction heads.

### Weaknesses
The paper provides many insights on the mechanism of ICL, but the potential future directions built on these findings are less clear.

### Questions
What are possible sources of the accuracy improvement from instruction given that instructions do not seem to contribute to the information removal process?

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
10

### Confidence
2

---

## Human Reviewer 3

### Summary
This paper investigates the internal mechanism of In-Context Learning (ICL) in large language models (LLMs) from a new perspective — task-oriented information removal. Instead of viewing ICL as pattern imitation or label copying via Induction Heads, the authors argue that few-shot demonstrations act as filters that suppress task-irrelevant information in the hidden representations.

To support this view, they introduce two novel probing metrics: Covariance Flux through the Task-Verbalization Subspace (TVS) and Eccentricity, which quantify how hidden-state variance aligns with task-relevant directions and how anisotropic (filtered) the representations become. Using these tools, they demonstrate that (1) LLMs spontaneously recover the TVS during ICL, (2) this process can be measured as an information-removal trajectory across layers, and (3) a new class of attention heads, Denoising Heads, drive this mechanism, overcoming the well-known limitation of Induction Heads when facing unseen labels.

### Strengths
1. **Novel probing metrics with clear interpretability.**
The introduction of Covariance Flux through TVS and Eccentricity provides a fresh, quantitative lens for studying internal representation dynamics of LLMs. These metrics go beyond uncovering new ICL mechanisms and identifying denoising heads, enabling a measurable understanding of how information filtering evolves within transformer layers.

2. **New mechanism of ICL: task-oriented information removal.**
This paper shifts the focus of ICL mechanism from token-copying to information filtering, which deepens our conceptual understanding of ICL.

3. **Identification of Denoising Heads addressing the Induction-Head limitation.**
The discovery of Denoising Heads is an insightful contribution. These heads operate locally and semantically, filtering query representations rather than copying lexical patterns, thereby explaining why LLMs can generalize to unseen labels -- something Induction Heads cannot do.

4. **Strong empirical grounding.**
The experiments are extensive and solid.

### Weaknesses
**Overly dense and somewhat disorganized presentation.**
While the findings are rich, the paper feels overly packed. The main sections attempt to cover TVS analysis, metric design, mechanistic ablations, and head identification all at once. The narrative flow occasionally sacrifices clarity for compactness, making it harder for readers to see the conceptual through-line.

### Questions
Suggestion:

 **Center the paper on the probing methodology.** 

If the authors had focused the paper around the two new metrics (Covariance Flux and Eccentricity) as a general probing framework, the contribution would appear sharper and of broader relevance. The ICL mechanism and Denoising Head discovery could then be presented as compelling applications of this framework, rather than parallel findings competing for attention. 

**A more layered structure** -- with detailed exposition of the probing metrics in the main text and the some extended empirical findings moved to the appendix -- would make the paper both clearer and more impactful. You can organize the paper similar to [1].

Minors:

1. In Figure 3, I understand that the marker represents the lower bound of information removal with a value between 0 and 1. However, please include at least one reference marker (for example, the marker size corresponding to no information removal) so that readers can intuitively gauge the approximate lower bound portion of information removed.

[1] Ren, Yi, and Danica J. Sutherland. "Learning dynamics of llm finetuning." arXiv preprint arXiv:2407.10490 (2024).

### Soundness
4

### Presentation
2

### Contribution
4

### Rating
10

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper argues that few-shot in-context learning (ICL) works by selectively removing task-irrelevant information from query representations, steering the model toward the intended task. In zero-shot settings, injecting a low-rank "task-verbalization" filter that projects onto a Task-Verbalization Subspace (TVS) dramatically improves accuracy even while preserving only ~0.7% of dimensions, while in few-shot ICL the model spontaneously moves hidden states toward this TVS, as measured by geometric metrics that rise in middle-to-late layers. The paper identifies "Denoising Heads" (DHs)—attention heads whose ablation disrupts this removal operation—which are largely independent of induction heads and show local re-encoding patterns over query tokens; ablating DHs significantly reduces ICL accuracy and nearly collapses performance in unseen-label settings, demonstrating the causal importance of this mechanism. The authors note this mechanism applies to clustering-style classification tasks but not bijective tasks like translation, with experiments conducted across multiple text classification datasets and models including LLaMA and Qwen variants.

### Strengths
1. Task‑oriented information removal operationalized via a low‑rank subspace and two geometric metrics that correlate with accuracy and layer depth. 

2. identification and causal ablations of Denoising Heads, with independence from induction heads and a sensible local re‑encoding attention pattern. 

3. Explicit failure case on bijective tasks (translation/fact recall) and an explanation of why low‑rank removal should not help there. 

4. Broad, careful experiments across six datasets and three models, with appendix replications and unseen‑label analyses that stress‑test induction‑only stories.

### Weaknesses
1. Theory is largely heuristic. The link between covariance and “information removal” is argued but not formally proved; metric choices and DH thresholds (±3.5%) are somewhat ad hoc. 

2. Narrow task family. The focus is single‑label classification; reasoning, chain-of‑thought, or generation settings are not studied, and the paper itself notes the mechanism likely does not apply to bijective mappings. 

3. Scale and generality. Only up to 8B models are used. It remains unclear whether DH distributions and metric trends stabilize for larger base or instruction‑tuned LMs.

### Questions
1. How sensitive are results to scanning all layers/heads? Could you provide one full‑layer run on a smaller model to confirm no hidden DH clusters?

2. Do your metrics or DH patterns say anything in multi‑class generative outputs (e.g., natural language rationales)? If not, what would be needed?

3. Any preliminary results on larger models (e.g., 13B/34B) or instruction‑tuned variants to test whether DHs persist or shift?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
4
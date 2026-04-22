# Negative Pre-activations Differentiate Syntax

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 8

## Abstract
Modern large language models increasingly use smooth activation functions such as GELU or SiLU, allowing negative pre-activations to carry both signal and gradient. Nevertheless, many neuron-level interpretability analyses have historically focused on large positive activations, often implicitly treating the negative region as less informative, a carryover from the ReLU-era. We challenge this assumption and ask whether and how negative pre-activations are leveraged by models. We address this question by studying a sparse subpopulation of Wasserstein neurons whose output distributions deviate strongly from a Gaussian baseline and that functionally differentiate similar inputs. We show that this negative region plays an active role rather than reflecting a mere gradient optimization side effect. A minimal, sign-specific intervention that zeroes only the negative pre-activations of a small set of Wasserstein neurons substantially increases perplexity and sharply degrades grammatical performance on BLiMP and TSE, whereas both random and perplexity-matched ablations of many more non-Wasserstein neurons in their negative pre-activations leave grammatical performance largely intact. Conversely, on a suite of non-grammatical benchmarks, the perplexity-matched control ablation is more damaging than the Wasserstein neuron ablation, yielding a double dissociation between syntax and other capabilities. Part-of-speech analysis localizes the excess surprisal to syntactic scaffolding tokens, layer-specific interventions show that small local degradations accumulate across depth, and training-dynamics analysis reveals that the same sign-specific ablation becomes more harmful as Wasserstein neurons emerge and stabilize. Together, these results identify negative pre-activations in a sparse subpopulation of Wasserstein neurons as an actively used substrate for syntax in smooth-activation language models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper examines the role of so-called 'Wasserstein neurons' in transformer networks, emphasising their use of negative activations to encode syntactic information. Building upon prior work on Wasserstein neurons, the authors analyse how clamping these negative activations to zero affects model performance.
Experiments demonstrate that eliminating negative activations in Wasserstein neurons significantly degrades model performance, whereas the same intervention has minimal impact on non-Wasserstein neurons.

### Strengths
The concept of Wassestein neurons is interesting, and this work tries to explain why they are important.

Important plots have error bars.

The importance of Wasserstein neurons for grammar in LLMs is rigorously shown.

### Weaknesses
1) While the findings are interesting, the work largely extends ideas already presented in “Wasserstein Distances, Neuronal Entanglement, and Sparsity.”
I understand the main experimental results as follows: Suppressing activations in neurons identified as important (from previous work) harms model performance more than suppressing other neurons, which is intuitive and expected from the original discussion. 

2) The part of specifically attributing grammar to these neurons is not convincing.  
Is it possible that grammar is simply the first thing that is lost when a model degenerates? 
How does performance decrease on non-grammar benchmarks when disturbing Wasserstein-Neurons? Perplexity indicates that changes could be large, too, which would contradict a specific attribution.

**Minor remarks:**

Figure 3 states: "(a,b) share the same legend." However, they have different x and y labels and different numbers on the axis.

The text has an abundance of -- [some text] -- mid-sentences, which disturbs the flow of reading.

### Questions
Specifically, the "CONCLUSION AND DISCUSSION" part reads to me like all it could be logically derived from the statement:
"Wasserstein neurons are important and modifying them disrupts model performance."
Could the authors elaborate on key findings that are rigorously analyzed with e.g., ablation studies that are not in very close relation to this?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper shows the importance of wasserstein neurons within LLMs for syntactic processing. Precisely, wasserstein neurons are shown to map similar input vectors to different negative output values in LLMs equipped with smooth activation functions. Such neurons emerge very early during retraining and are more frequent in larger models. Ablating wasserstein neurons results in degraded syntactic capabilities, as reflected by the BLIMP and TSE benchmark scores. Furthermore, wasserstein neurons are shown to affect most the LLM’s performance on ellipsis and subject agreements, most crucially at early layers. Lastly, negative differenntiation is shown to be a mechanism for syntactic processing, forming negative-negative pairs.

### Strengths
* The paper is creative and shows a large set of results.
* The paper links elegantly syntax and single neurons, which is a unique and well executed finding.
* The writing and the figures are clear and precise.

### Weaknesses
* The correlation between the pretraining emergence of Wasserstein neurons and TSE is not convincing alone, there could be a lot of possible confounds during pretraining.

* There is no link with the mechanistic understanding of syntactic processing. For example, understanding *how* these neurons combine to  parse syntactic trees.

* There is no automated method for knowing what syntactic features drive each neuron, this would allow to list all wasserstein neurons and their associated syntactic pattern.

* The proposed mechanism is specific to LLMs with smooth activation functions (GELU/SiLU), but, is this condition necessary to acquire syntax? However, ReLU-based LLMs such as OPT also exhibit clear syntactic competence, suggesting that smoothness in the negative regime is not necessary for syntax acquisition. This raises the possibility that negative differentiation is one implementation path rather than a universal requirement for syntactic encoding.

### Questions
* When computing WD with respect to the unit Gaussian, where is this gaussian centered? Does this not modify the WD?
* Figure 2d, what is “syntactical scaffolding experience”? Why are determiners, punctuation … more important for syntax?
* It is quite surprising that syntactic processing happens at early layers within the MLP blocks and not in middle layers in attention layers (as previously suggested by probing works)? Is there any explanation for this?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors study "Wasserstein" neurons, which exhibit strongly non-Gaussian distributions of preactivations. They show that for models with non-ReLU activation functions, negative preactivations are functionally important through sign-specific ablations, developmental analysis across checkpoints, and other experiments. The authors provide additional evidence for these neurons being especially implicated in predicting grammatical techniques by targeting their analysis to grammatical benchmarks.

### Strengths
- **Well-motivated**: The introduction is clearly written. The discussion on the need to reexamine the distribution of negative pre-activations for non-ReLU is strong.
- **Diverse set of models**: I appreciate that the authors have included a diversity of LLMs. 
- **Self-explanatory figures**: The figure captions are clearly written and self-explanatory, making this easy to follow. 

Full disclosure: I reviewed an earlier workshop version. From memory, this revision is a substantial improvement in cohesion.

### Weaknesses
A. WD may act as a proxy for skew/kurtosis rather than "entanglement."  

In many places, the authors treat "high WD from a Gaussian baseline" ("WD score") as equivalent to "highly multimodal" or "strongly entangling." This is implicitly justified via the observation that WD and mapping difficulty (MD)  are highly correlated: i.e., neurons with high WD score are likely to map similar input vectors to widely separated preactivations. However, "high WD score" is also compatible with "non-negligible higher moments." This is not discussed, which is a source of confusion and leads to implicitly overstating the strength of the Wasserstein neurons as "entangled neurons" hypothesis.

Minimum fix: Some additional comment clarifying the alternative hypothesis (high skew/kurtosis/higher moments) would be enough to resolve this concern. The authors could mention as a limitation that the methodology cannot pick up the distinction between heavy tails and multimodality and that this weakens the "entanglement" interpretation.

Maximum fix: Run an additional comparison between WD and skewness/kurtosis. 

Point of clarification: Section 2.1 defines MD by input‑difference / output‑difference averages. If that is indeed the implemented metric, high MD means _low_ output separation for similar inputs. This is a bit confusing because the text uses (lines 126–127) "MD of a neuron to summarize how far apart neurons map similar inputs."

B. The argument for WNs affecting grammar over other kinds of structure is weak(ly presented). 

The grammar hypothesis is introduced in section 2 by demonstrating that WD correlates with TSE (a specific grammatical benchmark) performance over training. In this section, the authors never discuss any alternative explanation (e.g., WNs are helpful for general-purpose natural language processing). Towards the end of the section the authors ask (lines 195–198): "is [this] structure merely a byproduct, or is it causally necessary for grammatical behavior?" This is a sudden and unexpected jump.

Evidence for the grammar hypothesis is presented in section 3 and Fig 3c & d and section 3. I find this mostly convincing, though I would like to see an additional comparison like Figure 3c against a non-grammatical control benchmark (from my current understanding, the authors have not yet ruled out the possibility of WNs being important for grammar *as well as* being important to other roles – though figure 4d does point in this direction). 

Line 275–278 "Cumulative ablations raise error across all syntactic classes, consistent with a role for Wasserstein neurons as more general-purpose carriers of grammatical information rather than specifically detecting individual phenomena" There is no discussion of any control assessing the effects across non-syntactic categories. Perhaps this evidence is implicitly presented in figure 4, but it is not clear to me where to look for this. I would appreciate if the authors can comment on this. 

Minimum fix: The paper would be much easier to understand if the grammar hypothesis is introduced more naturally. Another option would be to simply foreshadow the grammar hypothesis in section 2 and promise that it will be justified later. 

Maximum fix: The paper would benefit from repeating 4c with additional non-grammatical benchmarks, and providing further detailed examples illustrating the effects of ablations on individual samples in Fig 4d. This could be limited to an appendix. 

Small additional note: when comparing to perplexity‑matched controls, please state clearly whether the layer distribution and number of perturbed units per layer are matched. If not, a sentence explaining that the current matching is still incomplete would pre‑empt reader confusion.

C. Treatment of negative-negative separation is relatively weak. 

There's an obvious confounder in the analysis of NN vs PN vs PP differentiation. Zero-clamping negative preactivations collapses the distinction between two samples that both have negative preactivations that are only distinguished in magnitude. The same is not true for two samples that differ in preactivation sign or where both inputs have negative preactivations. As such, it is very unclear to me whether the observed higher correlations between NN ablations and damaged performance tell us anything about whether NN-differentiated samples are more important for grammar. The correct baseline would also involve ablating the particular positive activations for the PN or PP pairs. 

Minimum fix: Mention the confounder explicitly and point to it as a limitation of the current design. 

Maximum fix: Run additional experiments. For example, you could run (i) a matching positive‑clamp for the PN/PP baselines in the same layers, or (ii) sign‑flip/abs interventions to test whether sign vs. magnitude drives the effect.

Suggestion: I find the NN/PN/PP differentiation analysis to be relatively weaker than the rest of the text. I would recommend moving section 7 (and 6?) to the appendix (along with a clearer mention of confounders), and using the remaining space to strengthen the case for the preceding sections (esp. the grammar hypothesis). 

If the authors implement all of my minimum fixes and one or two of the maximum fixes, I will raise my score to an accept. I think including all of the maximum fixes is unlikely to further raise my score to a strong accept, but I would still recommend the authors consider these additional changes.

### Questions
- Besides the fact that WD is more neuron-focused than MD (lines 127–129), is the main reason to favor WD over MD that WD is more computationally efficient? It seems like it might otherwise be better to compute aggregate MD scores for each neuron since this would get closer to the question of entanglement. 

Minor improvements:

- (This is related to weakness A) The definition of Wasserstein neurons and "non-Gaussian" structure could be improved in Sec 2.1 and Fig 1. Phrasing such as (lines 76–78) "the non-Gaussian structure of Wasserstein neurons appears in the positive region ... [for GELU] the non-Gaussian structure is concentrated in the negative pre-activation region"  is confusing because the distribution as a whole is clearly non-Gaussian. I believe the authors mean to say something like "the bulk of preactivations occupies a large Gaussian peak. The mode of this peak is concentrated in the negative/positive region respectively for ReLU and non-ReLU networks." It would be clearer if the authors discussed this purely in terms of the largest mode of the distribution.
	- Line 63–66 "the deviation from Gaussianity concentrates in the negative region of the pre-activation space ... ReLU-based models ... do not exhibit comparable structure in the negative space" is clearly wrong as stated since the ReLU-based neurons also have clear non-Gaussian structure across the positive and negative range of activations. 
- Lines 55–57 "Nevertheless, many interpretability workflows retain the ReLU-era heuristic, equating activity with positive pre-activations and implicitly treating the negative region as inert." I question whether this is true. I believe many workflows like SAEs are sensitive to negative activations. Maybe the authors believe that the field still implicit acts with this expectation (which would require weakening the language) or they can provide some references to examples of real-world workflows making this assumption. 
- Lines 16–17: "We show that this negative region is functional rather than simply favorable for optimization." What does this mean?
- Line 19 "entangled neurons" is not defined until line 38
- Line 161: "mildly more expressive in their distributions" what does this mean?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper argues that a tiny, entangled subpopulation of MLP "Wasserstein neurons" in LLMs computes in the negative pre-activation region of smooth activations, and that this sign-specific behavior is mechanistically important for syntax. Evidence: (1) WD/"mapping difficulty" characterization across models; (2) sign-specific ablations that zero only negative pre-activations for the top-WD neurons cause large PPL increases and large drops on BLiMP/TSE, unlike random or perplexity-matched controls; (3) effects localize to functional POS tokens and early layers, compounding across depth; (4) training-dynamics: Wasserstein neurons emerge early and the same sign-specific ablation harms grammar more as training proceeds. ReLU models (OPT) lack the negative-region structure, consistent with clamping.

### Strengths
1. Clear, testable mechanistic claim (negative pre-activation computation) linked to syntax; the sign-specific intervention is surgical and well-motivated for non-ReLU models.

2. The paper also provides compelling causal evidence. That is, clamping negatives of top-WD neurons yields large BLiMP/TSE drops even when perplexity is matched by ablating many low-WD neurons; random controls are much smaller.

3. The paper demonstrates cross-model coverage (Pythia, Llama-3.1-8B, Mistral-7B-v0.3, Qwen-3-8B) and ReLU vs non-ReLU comparison.

### Weaknesses
1. Zeroing negatives changes the neuron’s transfer function (effectively ReLU-izing selected channels), which can alter residual/RMSNorm statistics and induce broad PPL rises unrelated to "syntax-specific" computation. My worry is that the perplexity-matched control (many low-WD channels clamped) helps, but it does not fully control for distributional/variance changes in the same high-WD channels.

2. The figure-level observation that OPT (ReLU) lacks negative-region structure is interesting; strengthen by running the same sign-specific ablation (no-op in ReLU) plus a mirror positive-only clamp in non-ReLU models to show asymmetry is truly about negative region utility rather than general clipping.  

3. The negative-negative (NN) separation result is striking; however, the pipeline samples top MD pairs among similar inputs and counts sign patterns. To exclude lexical/position confounds, you can do better control for token frequency, position, and subword overlap.

4. Grammar drops are strong; it would be great if you could also report non-syntactic tasks (e.g., TruthfulQA/QA/calibration) to establish specificity: are you removing general capacity or truly syntax-critical computation? The POS analysis suggests scaffolding specificity, but a broader behavioral panel would solidify the claim.

### Questions
1. In training dynamics, if you freeze the early high-WD cohort vs. others, do grammar curves diverge as predicted?

2. Could you report CIs/seeds for BLiMP/TSE?

### Soundness
3

### Presentation
4

### Contribution
3

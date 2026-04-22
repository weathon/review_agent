# Measuring Sparse Autoencoder Feature Sensitivity

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 2, 4, 2

## Abstract
Sparse Autoencoder (SAE) features have become essential tools for mechanistic interpretability research. SAE features are typically characterized by examining their activating examples, which are often "monosemantic" and align with human interpretable concepts. However, these examples don't reveal *feature sensitivity*: how reliably a feature activates on texts similar to its activating examples. In this work, we develop a scalable method to evaluate feature sensitivity. Our approach avoids the need to generate natural language descriptions for features; instead, we directly use language models to generate text similar to a feature's activating examples and test whether the feature activates on these inputs. We demonstrate that sensitivity measures a new facet of feature quality and find that many interpretable features have poor sensitivity. Human evaluation confirms that when features fail to activate on our generated text, that text genuinely resembles the original activating examples. Lastly, we study feature sensitivity at the SAE level and observe that average feature sensitivity declines with increasing SAE width across 7 SAE variants. Our work establishes feature sensitivity as a new dimension for evaluating both individual features and SAE architectures.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a scalable method to evaluate SAE feature sensitivity—the probability that a feature activates on texts similar to its activating examples. The key innovation is using LLMs to generate semantically similar text without requiring natural language feature descriptions. The authors validate their approach through human evaluation and demonstrate that (1) many interpretable features exhibit poor sensitivity, and (2) average feature sensitivity declines with increasing SAE width across multiple architectures. This establishes feature sensitivity as a new dimension for evaluating both individual features and SAE architectures.

### Strengths
1. The explanation-free approach elegantly sidesteps the problem of imperfect feature descriptions that plague other interpretability evaluations
2. Human evaluation convincingly demonstrates that generated text genuinely resembles activating examples. The 79% "indistinguishable" rating provides strong support for the method's validity
3. Evaluation across 141 SAEs (29 GemmaScope + 112 SAEBench), spanning 7 architectures and multiple widths, provides robust evidence
4. The declining sensitivity with SAE width is an important discovery for the field, especially as it holds across architectures including Matryoshka SAEs designed to address scaling issues
5. Proper controls (frequency weighting), filtering justification, and statistical analysis strengthen conclusions

### Weaknesses
1. Filtering removes 35-79% of features (Table 2). While the 90% truncation threshold is justified, this substantially limits the generality of findings.
2. The 20-token total window (10 before + 10 after) may inadequately capture features requiring longer context. The paper acknowledges this but doesn't quantify the limitation
3. While documenting that sensitivity decreases with width is valuable, the paper provides limited insight into why: (1) Is this due to feature splitting/absorption? (2) Do wider SAEs learn more specific features that are harder to trigger? (3) What are the implications for practical use?

4. The binary activation criterion discards potentially valuable information about activation strength. Features might "weakly activate" on generated text, which could indicate partial sensitivity

### Questions
1. Have you analyzed how sensitivity varies with the typical context window required for feature activation? Can you categorize features by their context requirements?
2. Can you provide more analysis or hypotheses about why sensitivity declines with width? Does this relate to feature splitting, absorption, or other known SAE phenomena?
3. How do activation magnitudes on generated vs. activating text compare? Could weak activations indicate partial sensitivity?
4. Could you adapt your method for rare features by using the LLM to generate larger corpora of semantically similar text?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper examines how SAEs can be quantitatively evaluated for interpretability and performance. It revisits metrics such as sparsity, reconstruction loss, and feature interpretability, highlighting trade-offs and limitations. The authors argue that current metrics fail to capture the nuanced balance between feature disentanglement and reconstruction quality. They propose improved metrics and evaluation setups that better reflect the practical goals of mechanistic interpretability.

### Strengths
1. The paper targets a crucial issue, how to measure SAE interpretability rigorously, an under-explored yet central topic in mechanistic interpretability research.

2. It systematically examines existing metrics (e.g., dead neuron rate, NMSE, feature overlap) and empirically demonstrates where they diverge from qualitative interpretability assessments.

3. Multiple model families (e.g., Pythia, Gemma) are analyzed. The paper reports trends across sparsity regimes and datasets rather than cherry-picking results.

### Weaknesses
1. The paper mostly refines and synthesizes existing metrics rather than introducing fundamentally new methods. Its contribution lies more in analysis than in methodological innovation.

2. While improved metrics are proposed, their theoretical justification is sometimes heuristic and lacks formal grounding or proof of desirable properties (e.g., invariance or sensitivity bounds).

3. The study shows correlations but doesn’t deeply explain why certain metrics diverge or what model mechanisms underlie these discrepancies.

4. There’s minimal comparison between metric-based interpretability and human judgments of monosemanticity or clarity, which would have strengthened validation.

### Questions
1. How robust are the proposed metrics to model size and dataset domain changes?

2. Can the metrics generalize to non-text modalities (e.g., vision, multimodal SAEs)?

3. How sensitive are metric rankings to the specific sparsity regularization used?

4. Do the proposed evaluation metrics correlate with human feature-interpretability scores?

### Soundness
3

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
2

### Summary
This paper introduces "feature sensitivity," a novel metric for evaluating Sparse Autoencoder (SAE) features that measures a feature's ability to activate on semantically similar inputs, not just its canonical examples. The authors propose a clever pipeline that uses an LLM to generate new test cases directly from a feature's activating examples. The work's central findings are that many highly interpretable features surprisingly exhibit poor sensitivity and, more critically, that average feature sensitivity consistently declines as SAE width increases. This result is presented as a new, fundamental challenge for scaling SAEs.

### Strengths
1. The paper identifies and formalizes a critical gap in SAE evaluation. The distinction between interpretability (cohesive activating examples) and sensitivity (robust generalization) is a valuable conceptual contribution that pushes the field toward a more rigorous understanding of feature quality.

2. The authors perform solid validation of their method, including a blinded human evaluation that confirms the quality of the generated text and an automated analysis of text novelty.

### Weaknesses
1. The paper identifies that features have varying sensitivity but provides no actionable insights: What should practitioners do with features that have low sensitivity? There is no evidence that sensitivity (as defined here) correlates with utility for downstream tasks (steering, circuit analysis, unlearning, etc.). No solutions or training modifications are proposed to improve sensitivity.

2. The paper filters out 35-79% of features, with exclusion rates dramatically increasing with SAE width (51% for 65K to 79% for 1M). The main finding (wider SAEs → lower sensitivity) could be entirely explained by differential filtering rates rather than true sensitivity differences. Filtering for features that "activate on their own truncated examples" explicitly selects for simpler, context-independent features, which is acknowledged but not adequately addressed. Table 2 shows that this filtering becomes more severe precisely where the sensitivity decline is observed, creating a plausible confound. The robustness checks in Figure 9 don't fully resolve this since different cutoffs still show the same filtering pattern.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Sparse Autoencoders are a very popular method to decompose LLM activation space into interpretable features but because they are trained in an unsupervised way we don't have ground truth features. As a result, evaluation of feature quality is difficult. The authors propose a new method to measure one specific aspect that makes a good feature: Sensitivity, i.e. whether the feature activates for prompts semantically similar to activating examples.

The major flaw in their approach is the assumption that the LLM samples **should** activate the feature. This is not true in practice, and as a result, their metric measures either whether the feature is sensitive, or whether the generator LLM's implicit feature description is good and we cannot distinguish both, thereby limiting the usefulness of this metric.

### Strengths
- the research question (how to evaluate SAE feature quality) is very important given SAE's popularity and also neglected
- the experiments done are sound and mostly well executed

### Weaknesses
The central assumption that the entire method relies on is that LLM-generated, semantically similar samples should activate the feature. Why should this assumption hold? The authors don't present any idea or empirical data and simply assume this is true. But it is important: If the LLM generates samples where the features should NOT activate (examples below), then the sensitivity score could be entirely wrong.

The authors present not needing a feature annotation as a main advantage of their strategy. However, although they don't require an **explicit** annotation, the LLM sample generator needs to make an **implicit** guess about what the feature might encode, in order to generate activating examples. If this guess is wrong, the LLM would generate samples that should not activate and the sensitivity score is low not because the SAE feature is low-quality, but because the LLM doesn't understand what makes the shown activating examples really activate.

To illustrate why I expect this problem to be the **norm** and not an edge case, I'd like to give some examples.

Example 1: Ground truth feature the SAE found is "The subject of this sentence is the second noun", and the feature fires at positions that predict a noun.
This is a feature that should exist in real language models as it's part of the IOI circuit (Wang, 2022) and real SAE features have actually been found (Makelov, 2025). So this is a feature that is great, monosemantic, with a clear interpretation, and many such features are present in real SAEs.
Let's pretend this SAE feature is perfect, ie sensitivity=100% and think through how high the proposed method would score the sensitivity. Input prompts with activating examples would be seemingly arbitrary text, with many activations being at tokens like "to", "the", "for", etc, since these words often predict a noun that should possibly be copied from earlier in the context. The LLM sample generator would now just generate arbitrary sentences that just so happen to contain those tokens but no specific sentences where the subject/object should be copied and thus the proposed sensitivity metric would be low although the "ground truth" sensitivity is high.

Example 2: Features, that are better explained by their role on the LLM output. An LLM has to understand input but also generate output and many features, especially in late layers have shown to be better explained by their role in crafting the model's prediction. Often, activating this neuron leads to promoting or inhibiting a meaningful set of tokens as has been shown by simply calculating f @ W_dec @ W_unemb and most features in late layers behave like this. For example, let's assume a feature is active whenever the model predicts a form of the word "sell". This feature could be identified quite easily by calculating f @ W_dec @ W_unemb and showing that activating it leads to the logits of [" sold", "sold", " sell", " resold", ...] being higher. 
Here's the link for this feature on neuronpedia: https://www.neuronpedia.org/gemma-2-2b/19-clt-hp/40860
Looking at activations alone, it activates on words like "which", "it", "to", for" in various different contexts. Only a small subset of prompts actually contain a form of the word sell at the next token position, because often, the model's prediction is different than the actual text. Thus, again, the LLM sample generator would generate samples where most don't activate the feature because the LLM has no idea what the feature really activates for. So sensitivity would be low although the feature learned is perfect.

This problem gets worse with scale:
- larger models have more layers and CoT tokens, so most features need to be explained in terms of their interactions with other features rather than inputs.
- more intelligent models might have more abstract, complex features, still monosemantic, but in a nontrivial way. Simply generating some sentences that appear similar wouldn't cut it.

As a result, the proposed metric is not a principled method to evaluate SAE feature quality because it cannot distinguish whether the LLM sampler understands the nature of this feature vs the feature itself has low sensitivity.

### Questions
- methodological details are missing for the human study: number of human evaluators, who they were (e.g. MI researchers, undergraduates, etc), how many prompts did each person score, how was blinding done, etc.
- ll 448 mentions results for Matryoshka SAEs but no data or figure is linked. Where can I find the results of this experiment?
- Figure 3b, could the authors clarify which Auto-Interp Score was calculated here? (e.g. detection, fuzzing, simulation, something else)
- Figure 3b, c, d, why are the scores clustered around decimals?
- Only 2M tokens were used to produce activating examples. Since many features occur rarely (ie frequency 1e-5 or less), I expect that many features only have few activating examples. The authors acknowledge this limitation but there's a deeper flaw: Because you sample so few, you likely miss the highest activating, often most clear and interesting examples. If more activations were used, the activating examples might have been clearer, the LLM would have been able to generate better samples, and the sensitivity score could've been possibly much higher.

### Soundness
2

### Presentation
2

### Contribution
2

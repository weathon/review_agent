## Summary
This paper proposes an experimental framework based on formal grammars (random and context-free hierarchical) to disentangle two learning modes in LLMs: memorization ("learning by rote") and generalization ("learning with understanding"). Using three open-source model families (Pythia-1B, Llama3-8B, Mistral-Nemo-12B), the authors characterize overlapping memorization and generalization phases during fine-tuning, argue that recollection accuracy alone cannot determine which mode is active, and demonstrate that dataset entropy inversely predicts ease of generalization while predicting ease of rote memorization.

---

## Strengths

- **Controlled grammar-based experimental design**: Using probabilistic formal grammars provides genuine experimental control unavailable with natural language: exact distributional matching between train and test, no pretraining contamination of the specific strings, and precise manipulation of entropy. This is a substantive methodological contribution, not just a convenience.

- **Consistent cross-model replication**: The core observations—the overlap between generalization and memorization phases, the train/test divergence pattern, and the entropy effects—hold consistently across three architecturally distinct model families spanning more than an order of magnitude in parameter count. This significantly strengthens confidence that the dynamics are not artifacts of a single architecture or scale.

- **Compelling argument against recollection-only memorization metrics**: Figure 3 concretely demonstrates that two models can achieve identical training loss on a dataset while one is in the generalization phase and the other in the memorization phase. This is a specific, falsifiable point with clear implications for prior work on memorization auditing (e.g., Carlini et al. 2022, Tirumala et al. 2022), and it is well-supported within the paper's framework.

- **Interesting sequential training observation**: The finding in Section 4/Figure 4—that training on a second dataset first re-generalizes the model to the original distribution (briefly recovering test loss), then erases previously rote-memorized data—is a specific and non-obvious result that goes beyond standard catastrophic forgetting descriptions by identifying this re-generalization interlude.

---

## Weaknesses

### Fatal
None.

### Major

- **Missing connection to "grokking" literature**: The two-phase learning dynamics described in this paper (initial generalization phase followed by a memorization phase, with overlap) have a striking structural resemblance to the "grokking" phenomenon in deep learning. The paper makes no mention of grokking. This omission is significant: grokking research provides theoretical framing and experimental precedent for exactly the kind of phase transitions studied here, and readers familiar with that literature will immediately notice the gap. The authors should either explain why their findings are distinct from, or subsume, the grokking phenomenon, or explicitly connect to it.

- **Extremely small training set sizes limit the generalizability of findings**: The main experiments use n=8 and n=64 training samples—effectively toy regimes. These sizes are far outside the scale of any practical LLM training scenario and may be specifically tailored to produce memorization artifacts. The paper provides no evidence that the observed phase transitions, entropy effects, or sequential forgetting dynamics persist at n=512, n=2048, or larger. Without this, the claim that findings have "significant downstream implications" for real LLM training remains unsubstantiated.

- **The proposed memorization measure is introduced but not validated**: Section 3.2 proposes `memorization(M, D_train, D_test) = 1 - Loss(M, D_train) / Loss(M, D_test)` as an alternative measure, but this measure is never applied, analyzed over training epochs, stress-tested against edge cases, or compared to alternative formulations in the rest of the paper. The paper acknowledges that it requires access to a held-out test set from the same distribution, which is generally unavailable to auditors—but this critical limitation appears only briefly. Introducing a new measure without systematic validation or use weakens the methodological contribution.

- **The 5% threshold for memorization onset is arbitrary without sensitivity analysis**: The entire characterization of "when memorization starts"—which defines the phase boundary used throughout the paper—rests on the rule `Loss(M, D_test) / Loss(M, D_train) > 1.05`. The paper gives no justification for 5% over 2% or 10%, and no robustness check. Given that derived quantities (e.g., phase overlap duration, entropy comparisons) depend directly on this threshold, all quantitative claims about phase structure are partly artifacts of an unsupported parameter choice.

- **Entropy effects are insufficiently distinguished from confounded distributional changes**: Changing alphabet size from ℓ=2 to ℓ=26 simultaneously changes entropy, token identity space, tokenizer behavior, effective sparsity, and base string complexity. Oversampling "a" changes unigram frequencies, not just entropy. Skewing production rule probabilities changes structural depth distributions as well. The paper claims entropy is the explanatory variable, but these manipulations are not isolated. The entropy results in Section 5 are described qualitatively; no numerical entropy values, regression analyses, or matched controls are provided to pin down entropy as the causal factor.

### Minor

- **"Understanding" is terminologically misleading**: Using "learning with understanding" to describe in-distribution generalization on synthetic grammar tokens will invite skepticism at ICLR, where "understanding" carries semantic and cognitive connotations. The paper defines the term operationally but does not consistently flag when it uses the loaded versus technical sense. "Generalization" is the defensible term; the "understanding" framing should either be heavily caveated throughout or replaced.

- **The sequential training result is not clearly distinguished from catastrophic forgetting**: The observation that training on D_train,2 causes forgetting of D_train,1 is standard catastrophic forgetting. The authors' genuine contribution here is the observation that forgetting passes through a brief re-generalization phase—a specific empirical detail. The paper should more explicitly identify this as the novel element rather than presenting the broader forgetting result as novel.

- **The practical implication about cryptographic key forgetting is speculative**: The suggestion that one could trigger forgetting of memorized cryptographic keys by training on new random keys of the same format goes well beyond what the experiments demonstrate. The experiments use tiny fine-tuned datasets in fully controlled conditions; this implication should be framed as a speculative conjecture requiring independent validation.

- **Training protocol details are underdescribed in the main text**: For a paper whose central claims are about training dynamics, key hyperparameters (learning rate, scheduler, optimizer, whether full-model or adapter fine-tuning, sequence packing, number of update steps) are not summarized in the main text. While they may appear in the appendix, their absence makes it harder to assess whether the dynamics are robust or specific to one training configuration.

### Tiny

- **"Impossible" is used informally without formal support**: The paper says it is "impossible" to determine memorization from recollection alone (Q1, Abstract, Section 3.2). The argument is compelling and the supporting example (Figure 3) is effective, but "impossible" implies a formal identifiability theorem. The claim should be worded as "insufficient" or "cannot, in general" to accurately reflect what is shown.

- **The larger-dataset-delays-memorization observation is expected**: The finding that n=64 delays memorization compared to n=8 is presented as an empirical observation but is fully consistent with standard overfitting theory. It is useful as validation of the framework, but should not be presented as a surprising finding.

---

## Nice-to-Haves

- **Grammar-violating string evaluation**: Testing model loss on strings that deliberately violate specific grammar rules (e.g., swapping tokens to break a production rule) vs. valid unseen test strings across epochs would provide a more direct test of whether the model has internalized structural grammar rules or merely surface statistics. This would strengthen the "learning with understanding" interpretation.

- **Non-LLM baselines**: Showing whether a simple n-gram model or small RNN exhibits the same phase transition dynamics would establish whether the findings are LLM-specific. If simpler models show identical patterns, the contribution is about statistical learning in general, not specifically LLMs.

- **Natural language validation at small scale**: Even one controlled experiment on a semi-synthetic or structurally-constrained natural language dataset (e.g., templated text, code with known grammar, parallel translations) would help bridge the paper's explicit claim about "significant downstream implications for LLMs" to an empirical footing.

- **Quantitative summary of entropy effects**: A table or scatter plot of minimum test loss vs. entropy value, or memorization onset epoch vs. entropy, across conditions, would replace the current qualitative trend-reading from Figure 5 with concrete, citable numbers.

- **Attention or probe analysis**: Even basic linear probes on intermediate representations at different training epochs could test whether the model's internal representation of grammar structure changes qualitatively between the generalization and memorization phases, providing a mechanistic angle the paper explicitly acknowledges is missing.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[Removed] Criticism that model comparison is unfair**: The harsh critic flagged comparing Pythia-1B vs Llama3-8B vs Mistral-12B as unfair due to scale differences. In fact, the paper presents cross-model consistency as a robustness check, not a controlled ablation. Differences in scale and architecture are expected, and showing the phenomena hold across all three strengthens the findings.

- **[Removed] Criticism of missing training protocol from main text as a validity concern**: While protocol details belonging in the main text is a valid presentational concern, the appendix is referenced (Appendix A.2), and the experiments are reproducible given open-weight models and formally defined grammars. This is a presentation note, not a scientific validity concern.

- **[Removed] Criticism that "generalization ends at minimum test loss is just early stopping"**: This is accurate but not a meaningful criticism—the paper is not claiming early stopping as a discovery; it is using the epoch of minimum test loss as a natural, reproducible operational marker for a training-dynamics analysis. Relabeling this as "just early stopping" conflates a methodological tool with a claimed finding.

- **[Removed] Criticism that entropy findings are trivially explained by information theory**: The harsh critic and spark finder both suggest the entropy-memorization relationship is trivially explained (lower entropy = more compressible, higher entropy = more unique prefixes). The paper itself explicitly offers exactly these conjectures as candidate explanations. Dismissing a finding because the authors provide a plausible explanation is circular; the value is in empirically establishing the relationship and providing the mechanistic conjecture for follow-up work.

- **[Removed] Claim that the paper never acknowledges the gap between synthetic and natural language**: Section 6 explicitly states: "Our insights on memorization rely on synthetic data generated with formal grammars, and it is possible that some of the observations might change for real-world data." The limitation is acknowledged.

- **[Removed] Strength that "the paper is well-written and clearly organized"**: Generic, applies to any readable paper.

---

## Novel Insights

The most genuinely novel observation in this paper—one that goes beyond what any of the three reviewers highlight as primary—is the re-generalization interlude documented in Figure 4: when a model that has fully memorized D_train,1 is subsequently trained on D_train,2, the loss on D_train,1 does not simply rise monotonically but instead tracks upward toward the test loss level before diverging from it. That is, the model briefly recovers distributional generalization for the original dataset as a transient intermediate state before fully forgetting the rote-memorized content. This is a specific, structurally interesting phenomenon that is not simply catastrophic forgetting, nor is it predicted by standard overfitting dynamics, and it suggests that the rote-memorization representation and the distributional generalization representation compete over the same weight capacity in a non-trivial, ordered way.

---

## Suggestions

1. **Connect explicitly to the grokking literature** and explain whether the paper's phase dynamics are the same phenomenon, a variant, or distinct. The conceptual framework and empirical patterns are closely related.

2. **Add a sensitivity analysis for the 5% threshold**: Report how the memorization onset epoch and all downstream quantities change under 1%, 2%, and 10% thresholds. This would substantially strengthen the reliability of the phase characterization.

3. **Extend training set sizes to at least n=256 or n=512**: Even one additional size would let the paper address whether the observed phase transitions persist in less extreme data regimes, which is critical for the claimed practical implications.

4. **Either validate and use the proposed memorization measure systematically, or demote it**: If the measure is not going to be used across the paper's experiments, it should be positioned clearly as a proposal for future work rather than a methodological contribution of this paper.

5. **Replace or heavily caveat "understanding"**: Use "generalization" as the primary term throughout, reserving "understanding" only for high-level framing, and add a consistent explicit disclaimer that "understanding" means in-distribution syntactic generalization to unseen grammar strings, not semantic or cognitive understanding.

6. **Provide explicit entropy values for each experimental condition in Section 5**: Even a simple table mapping ℓ ∈ {2, 7, 26}, oversampling levels, and p ∈ {0.5, 0.75, 0.9} to computed H(G) would allow readers to assess whether entropy is the causal variable rather than correlated distributional properties.

---

## Evaluation on Key Axes

**Originality**: Moderate-to-high. The formal-grammar framework for studying LLM memorization dynamics is a genuinely novel methodological contribution. Several specific empirical observations (the overlap of phases, the re-generalization interlude, the entropy inversion) are not obviously available from prior work. However, some findings are reframings of standard overfitting dynamics, and the grokking literature represents closely related intellectual terrain that is not engaged.

**Importance of research question**: High. Disentangling rote memorization from distributional generalization matters for privacy auditing, copyright, and LLM evaluation. The specific argument that recollection-based metrics systematically mislead is practically significant.

**Whether claims are well supported**: Partially. The core qualitative observations are replicated across three model families, which is a genuine strength. However, the quantitative claims (phase boundaries, entropy effects) are underpinned by arbitrary thresholds and confounded manipulations, and the validity of findings is limited by extremely small training sets.

**Soundness of experiments**: Moderate. The controlled grammar design is elegant, the multi-model replication is good practice, and five random seeds with standard deviation reporting is appropriate for this type of study. The main concerns are the very small n, the unvalidated 5% threshold, and the confounded entropy manipulations.

**Clarity of writing**: Adequate. The framing is accessible and the figures communicate the key trends. The terminology issue with "understanding" creates avoidable ambiguity, and quantitative summaries are largely absent.

**Value to the research community**: Moderate-to-high. The framework itself (train LLMs on formal-grammar data to study memorization dynamics) could be adopted and extended by others. The entropy-memorization finding and the recollection-insufficiency argument are actionable insights for the memorization-auditing community.

**Contextualization relative to prior work**: Weak. The paper engages with the privacy/memorization auditing literature but misses the grokking literature (directly relevant mechanistic framing) and does not engage deeply with catastrophic forgetting literature, both of which are needed for honest contextualization of the observed dynamics.
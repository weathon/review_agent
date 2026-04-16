## Summary

The paper introduces a tunable hierarchical generative model for sequences on a fixed binary tree, with a filtering parameter \(k\) that controls how much long-range correlation is present. Using this setting, the authors train small encoder-only transformers on root classification and masked language modeling (MLM), compare them to Belief Propagation (BP) oracles, and study training dynamics, attention patterns, and probes. They report that transformers closely match BP’s marginals, learn longer-range correlations sequentially over training, and exhibit attention/probe structure aligned with the underlying tree.

## Strengths

- **Well-crafted synthetic framework with controllable structure.**  
  The hierarchical tree model with a filtering parameter \(k\) (Sec. 2.1–2.2, Fig. 1(a)) is a strong contribution in its own right: it provides an exact inference oracle (BP), lets the authors “dial” the effective correlation range, and isolates hierarchy in a clean way. The non-ambiguity assumption at \(k=0\) and the definition of \(\text{BP}_k\) are clear and mathematically sound.

- **Strong functional matching to the BP oracle.**  
  On both root classification and MLM, the trained transformers match BP’s *input–output behavior* very closely for the chosen grammar:  
  – They reach BP-level accuracy in- and out-of-sample across different \(k\) (Figs. 3, 4, 5).  
  – Their predicted probability vectors are well calibrated w.r.t. BP marginals, with small average KL divergence and tight scatter around the identity line (Fig. 1(b–d), and text in Sec. 3.2–3.3).  
  – The matching extends to controlled distribution shifts across \(k_{\text{train}}\) / \(k_{\text{test}}\).  
  This convincingly shows that, in this setting, small transformers approximate the Bayes-optimal posterior for the training generative model.

- **Clear and interesting learning dynamics.**  
  The staged behavior over training—where predictions initially align with \(\text{BP}_\ell\) (short-range correlations and leaf-to-root information) and then successively with \(\text{BP}_k\) for decreasing \(k\) (Figs. 1(c–d), 4, 5)—is a genuine insight. It provides a concrete and well-quantified example of “hierarchy discovered over time”: shorter-range correlations are learned first, then progressively longer-range ones.

- **Consistent hierarchical structure in attention and probes.**  
  The attention maps (Fig. 6) for MLM-trained models show an appealing alignment: for low \(k\), attention blocks grow to sizes \(\sim 2^{\ell-k}\) and for \(k=0,1\) the layers form a clear bottom-up hierarchy that visually mirrors the tree. Probing experiments (Fig. 7) show that deeper layers’ token representations carry more information about higher-level ancestors, roughly in a layer-by-layer fashion. These are informative, concrete windows into the learned representations.

- **Useful connection to pre-training.**  
  The MLM pre-training followed by root classification (Fig. 1(f)) illustrates, in a fully controlled setting, how unsupervised learning of the hierarchy reduces labeled sample requirements. This gives a didactic, mechanistic example of a widely observed phenomenon in practice.

## Weaknesses

### Fatal

None. The paper is a genuine paper with a coherent setup, sound experiments at the scale considered, and real positive findings. The main issues are overclaim and limited scope, not fundamental invalidity.

### Major

- **Overclaim: from functional equivalence to “implementing BP”.**  
  The central interpretability claim is that transformers “approximate the exact inference algorithm” and “learn to implement a close approximation of the exact [BP] algorithm” (abstract; Sec. 3.3; Sec. 4). The evidence, however, supports only *functional* approximation of the BP oracle on this specific grammar, not mechanistic equivalence:

  - Accuracy and marginal matching (Secs. 3.2–3.3, Fig. 1(b–d), 3–5) show that the network’s mapping from leaves to posteriors closely matches BP’s outputs on the distributions tested. This is necessary but not sufficient to establish that the internal computation is BP-like; any sufficiently expressive network trained with cross-entropy can approximate the same conditional distribution without using BP-style message passing.

  - The out-of-distribution experiments across \(k_{\text{train}}, k_{\text{test}}\) (Secs. 3.2–3.3, Figs. 3–5) are interpreted as evidence that the transformer “implements an approximation of \(\text{BP}_{k_{\text{train}}}\)”. But under the generative model, \(\text{BP}_{k_{\text{train}}}\) is just the Bayes-optimal predictor *for the training distribution*; any model that learns that conditional will exhibit the same degradation under mismatch. This is strong evidence that the network has learned the Bayes posterior for the \(k_{\text{train}}\) model, but it does not distinguish BP from any other computational realization of that posterior.

  - The existence construction in Appendix E is explicitly “introduced for the sake of interpretability” and “does not represent an exact explanation of the trained transformer computation.” It shows that *some* transformer with contrived embeddings and attention could implement BP in \(\ell\) layers, but there is no evidence that the *trained* network’s hidden states or weights resemble this construction.

  - The attention patterns (Fig. 6) and probes (Fig. 7) are compatible with hierarchical, bottom-up aggregation, but they do not uniquely identify the BP update equations. One could imagine many alternative hierarchical algorithms that produce similar blocky attention and ancestor-decoding behavior.

  As a result, the paper currently blurs three distinct statements: (1) the learned function closely matches BP marginals on this task; (2) the learned computation is hierarchical and tree-aligned; (3) the learned computation is (or is very close to) BP. The evidence solidly supports (1) and a moderate version of (2); it does not substantiate (3). This is a structural overclaim in the narrative and abstract.

- **Narrow experimental regime: single small grammar and architecture.**  
  All main results are for a **single** randomly sampled tensor with \(q=4\), \(\ell=4\) (sequence length 16), and a transformer whose depth is exactly \(\ell\), with a single attention head and standard sinusoidal positional encodings (Sec. 3.1). The paper notes that “results remain qualitatively unchanged in experiments on different grammars, see Appendix D.2,” but no main-text quantitative robustness is shown. In particular:

  - There is no systematic variation of the grammar: no exploration of less/non-log-normal transitions, higher ambiguity, different strength of correlations, or degenerate/extreme grammars.

  - The tiny setting (\(q=4, \ell=4\)) is especially brittle for interpretability claims: it is easier for a model to learn idiosyncratic “shortcuts” that happen to match BP’s outputs without general, scalable message-passing.

  - Architectural sensitivity is largely unexplored in the main text: multiple heads, different embedding dimensions, or depth \(\neq \ell\) are only mentioned briefly (Appendix D.1 for \(n_L<\ell\)), yet the conclusion discusses “vanilla encoder-only transformers” more generally.

  This does not make the findings wrong, but it does meaningfully limit originality and importance: in its current form, the work is best viewed as a careful case study of one grammar/architecture, not a general statement about transformers on hierarchical data.

- **Ambiguous use of mismatched-BP experiments as “algorithmic” evidence.**  
  In Sec. 3.2 (“Out-of-sample testing”) and analogous parts in Sec. 3.3, the authors argue that matching the performance of \(\text{BP}_{k_{\text{train}}}\) on data with \(k_{\text{test}}\neq k_{\text{train}}\) is “evidence that the transformers are implementing an approximation of the \(\text{BP}_{k_{\text{train}}}\) algorithm”. What is actually demonstrated is that the trained transformer behaves like the *Bayes-optimal predictor for the training generative model* when applied under distribution shift—exactly as \(\text{BP}_{k_{\text{train}}}\) does.

  From the perspective of statistical learning, this is expected behavior from any sufficiently expressive classifier trained to minimize cross-entropy on that training distribution; it does not reveal *how* the conditional distribution is internally computed. Treating this behavior as direct algorithmic evidence overstates what is shown and risks confusion: readers may infer that the network’s forward pass is somehow implementing the same message-passing graph as BP, which is not established.

### Minor

- **Interpretation of probes limited to representational evidence.**  
  The probing setup in Sec. 4 (“Probing the encoder representations”) uses fairly expressive two-layer readouts trained per layer and ancestor level on fresh labeled data (Appendix D.6–D.7). The results (Fig. 7) clearly show that deeper layers allow more accurate prediction of higher-level ancestors—consistent with progressive aggregation over larger blocks. However, probes only demonstrate that information *is present* in hidden states, not that it is *causally used* in the way BP would. The paper partially acknowledges this but also implies that “overfitting” scenarios can be ruled out, which is too strong given the probe capacity and absence of causal ablations.

- **Heavy reliance on non-ambiguous \(k=0\) case.**  
  The model at \(k=0\) is explicitly non-ambiguous: each child pair has a unique parent, hence the root is deterministically recoverable from the leaves (Sec. 2.1). In this setting, the root labels’ one-hot targets coincide exactly with the BP marginals, and calibration at \(k=0\) is therefore expected. The truly nontrivial calibration is at \(k>0\), where the root is not deterministically determined by the leaves; although the paper does report matching behavior there (text and Fig. 11 in the appendix), the main narrative leans heavily on the visually cleaner \(k=0\) figures.

- **Limited quantitative analysis of where transformer deviates from BP.**  
  KL divergences are averaged over large test sets (Fig. 1(c–d)), and scatter plots show overall tight correlations (Fig. 1(b)), but there is little analysis of *error structure*: e.g., performance on rare leaf configurations, systematic deviations for specific regions of the state space, or sensitivity to unusual inputs. Such analysis would clarify whether the model is uniformly BP-like or relies on shortcuts that break in defined regimes.

- **Connections to large-scale NLP are mostly aspirational.**  
  The conclusion and introduction mention implications for language models and potential curriculum learning strategies. Given the highly idealized setting (binary tree, fixed topology, short length, small vocabulary, known grammar), this is plausible but speculative. The paper would benefit from clearer framing that these are conceptual hints rather than validated practical prescriptions.

### Trivial

- Minor textual redundancies (e.g., duplicated figure captions for Figs. 1–7) and occasional slight inconsistencies in figure references appear to be artifacts of the extraction; nothing severe for clarity in the actual paper.

## Nice-to-Haves

- **Causal/mechanistic interventions.**  
  Ablations or perturbations targeted at specific layers or attention patterns (e.g., scrambling attention at layer \(k\), or zeroing certain token subspaces) to see whether performance degrades in ways predicted by the BP-inspired interpretation would substantially strengthen the mechanistic story.

- **Direct comparison of hidden states to BP messages.**  
  Rather than focusing only on accuracy and ancestor labels, one could project hidden states onto the BP message space (upgoing/downgoing vectors) and quantify alignment across layers. Even imperfect but structured alignment would be much stronger evidence of BP-like computation.

- **Broader grammar and architecture sweep.**  
  Including at least one additional main-text experiment varying either \(\ell\) or \(q\), or using a more ambiguous tensor, would improve confidence that the observed phenomena are not fragile to the particular sampled grammar.

- **Baseline architectures.**  
  Comparing to a simple CNN or RNN on the same hierarchical data would help isolate what is special about transformers in this setting, versus what any expressive sequence model would learn.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Claims that BP implementation is “falsified” by the existence proof in Appendix E being different from trained networks.**  
  The harsh critic notes that the existence construction uses disentangled embeddings and position-based attention “very different from the trained networks” and thus “does not support the claim that ‘the same architecture’ actually learned BP.” While it is correct that the construction is not evidence of what *was* learned, the paper itself is careful: “this does not represent an exact explanation of the trained transformer computation.” Using the existence proof as *supporting plausibility* is reasonable; the criticism that it is “misleading” or “falsifies” the claim is too strong. The real issue (kept above) is overinterpretation of empirical evidence, not misuse of the existence construction.

- **Any implication that the model or BP oracle might not exist or be unreleased.**  
  No such criticism arose explicitly, but per the instructions, any concerns about existence or release status of the BP implementation or transition tensors would be inappropriate: the paper clearly specifies them, and they must be treated as available.

## Novel Insights

The paper’s most novel insight is the detailed characterization, in a clean and exactly analyzable setting, of *how* a transformer trained with gradient descent sequentially incorporates higher levels of hierarchical correlation—first learning leaf-to-root and short-range dependencies, then progressively aligning with BP oracles that encode longer-range structure, as visible both in cross-\(k\) performance and KL alignment over training. This provides a concrete, data-driven example of staged hierarchical learning, and shows that in such settings transformer outputs can approximate exact BP marginals remarkably well, including under controlled distribution shifts, even though the mechanistic story remains incompletely pinned down.

## Suggestions

- **Recalibrate the central claims.**  
  Rephrase the abstract and conclusions to emphasize that transformers *approximate the BP oracle’s posterior* and *learn hierarchical, tree-aligned computations*, rather than that they “implement” BP. For instance, “approximate BP’s marginals” and “exhibit computation consistent with a hierarchical message-passing interpretation” would be accurate and still strong.

- **Make the limitations of scope explicit in the main text.**  
  In Sec. 3.1 or the introduction, clearly state that all results are for \(q=4\), \(\ell=4\), a single grammar, and architectures with \(n_L=\ell\). Frame the contribution as a detailed case study rather than a general statement about all vanilla transformers. Briefly summarize Appendix D.2’s robustness checks and, if possible, add one additional grammar in the main text.

- **Deepen the mechanistic analysis where possible.**  
  If feasible, add a measurement of alignment between hidden states and BP messages across layers (e.g., via linear projection to the true BP message vectors). This would bridge the current gap between existence proof and attention/probe observations.

- **Clarify the interpretation of mismatched-\(k\) experiments.**  
  In Secs. 3.2–3.3, explicitly explain that matching \(\text{BP}_{k_{\text{train}}}\) under \(k_{\text{test}}\neq k_{\text{train}}\) primarily demonstrates that the network has learned the Bayes predictor under the training distribution. Avoid framing this as direct algorithmic evidence unless supplemented with mechanistic alignment.

- **Add at least one more dimension of variation.**  
  If compute allows, extend one key figure (e.g., Fig. 5’s MLM staircase) to \(\ell=5\) or a different grammar in the main text, to demonstrate that the sequential learning behavior is not an artifact of the specific toy instance.

### Axis-based evaluation

- **Originality:** Moderate. The filtered hierarchical model and its use as a BP playground are neat, but conceptually related to prior synthetic CFG/RHM work. The staged learning analysis is incremental but well executed.
- **Importance of question:** Moderate to high for interpretability: understanding how transformers process hierarchical structure is a central and timely question, though this work targets a very idealized regime.
- **Support for claims:** Strong for functional-optimality and staged hierarchical learning; weak for the specific claim of implementing BP as an algorithm.
- **Soundness of experiments:** Solid within the chosen regime; lacking in robustness across grammars and architectures.
- **Clarity of writing:** Generally clear and well organized, with careful explanations of the model and BP, though some interpretive language is stronger than the evidence warrants.
- **Value to the community:** Useful as a case study and as an experimental framework; less so as a definitive mechanistic account unless claims are calibrated.

## Score and Decision

For calibration, I compared against:

- **0GzqVqCKns (“Probing the Latent Hierarchical Structure of Data via Diffusion Models”)** — scores 6, 8, 6, 6, accepted. That paper offered a clear synthetic model plus robust experiments across multiple real datasets, with appropriately cautious claims about probing latent structure.
- **J6qrIjTzoM (“Interpretability of Language Models for Learning Hierarchical Structures”)** — scores 6, 8, 3, 8, rejected. It similarly used synthetic hierarchical grammars and transformers, with interesting insights but concerns about overgeneralization and limited robustness.
- **v675Iyu0ta (“Interpretability Illusions in the Generalization of Simplified Models”)** — scores 3, 6, 6, 8, 5, rejected but influential as a caution against overinterpreting simplified representations.

Relative to these:

- This paper’s *experimental cleanliness and functional alignment with a known oracle* are strong, somewhat comparable to 0GzqVqCKns in their respective domains.
- However, its *overinterpretation of mechanistic claims* and narrow regime (single tiny grammar, limited robustness) align it more with J6qrIjTzoM and v675Iyu0ta, which were ultimately rejected despite having interesting ideas.
- Overall, I view the contribution as solid but not yet at the level of robustness and calibrated interpretation that warranted acceptance in the high-scoring anchor; the central interpretability claim needs revision and/or stronger mechanistic evidence.

Balancing these factors, I assign:

MY FINAL SCORE: <pineapple>5.5</pineapple>  
MY FINAL DECISION: <orange>Reject</orange>
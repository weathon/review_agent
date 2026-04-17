# LangPert: LLM-Driven Contextual Synthesis for Unseen Perturbation Prediction

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 2

## Abstract
Predicting cellular responses to previously unseen genetic perturbations remains a fundamental challenge in computational biology, with broad applications in understanding gene function, disease mechanisms, and therapeutic development. Despite advances in computational approaches, developing models that generalise effectively to novel perturbations continues to be difficult. Large Language Models (LLMs) have shown promise in biological applications by synthesizing scientific knowledge, but their direct application to high-dimensional gene expression data has been impractical due to numerical limitations. We propose LangPert, a novel hybrid framework that leverages LLMs to guide a downstream k-nearest neighbors (kNN) aggregator, combining biological reasoning with efficient numerical inference. We demonstrate that LangPert achieves state-of-the-art performance on single-gene perturbation prediction tasks across multiple datasets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In this paper, the authors leverage a pretrained LLM to predict cellular responses to unseen genetic perturbations. Given a set of train perturbation labels, they prompt the LLM to output a set of $k$ functionally related neighbors for a test perturbation $x$. The final prediction for $x$ is the average of these neighbors' gene expression vectors. The authors evaluate their method using two perturbation datasets and show that their approach outperforms prior works.

### Strengths
- The authors perform cross-validation for all experiments and also include a data-scaling analysis, where models are compared across different sizes of the train set.
- The analysis of how different LLMs affect the performance of LangPert is insightful and useful for the community.

### Weaknesses
- A significant limitation of this study is the absence of direct, simpler baselines that would be essential for validating the paper's central hypothesis. To isolate the LLM's contribution, the authors should have compared LangPert to simpler kNN aggregators guided by structured biological knowledge, such as a Gene Ontology similarity search or PPI graphs (e.g., SPACE embeddings [1]).
- The authors evaluate performance by calculating metrics on the top 20 DE genes. While it is true that this has been employed by other works such as GEARS, performance on this small subset may not be representative of the model's accuracy across the transcriptome. Furthermore, this metric is asymetric as DE genes are computed in the ground truth, not the model's predictions, hence, it does not properly penalize "false positives". I'd recommend including metrics computed on the full gene set in the appendix to provide a more complete assessment of performance.
- The authors claim that the LLM identifies the appropriate similarity criterion based on the context, however, there is no quantification of this claim. Only a few examples are given and the paper does not correlate the type of reasoning strategy with predictive accuracy for different classes of genes. Without this quantitative validation, the claim remains anecdotal.

Overall, given the number of methods being published in this space, the paper's benchmarking needs to be more rigorous by including other key foundation models and simpler baselines.

[1] https://academic.oup.com/bioinformatics/article/41/9/btaf496/8250101

### Questions
- Can the authors comment on the possibility of information leakage in the LLM's training data? The data is from 2022, with many papers discussing this specific dataset.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces LangPert, a hybrid framework that integrates large language models (LLMs) with a k-nearest neighbors (kNN) aggregator to predict transcriptional responses to *unseen* genetic perturbations. The core idea is to decouple reasoning and computation: LLMs identify biologically related perturbations by reasoning over literature and gene-function context, while kNN performs the numerical aggregation of expression responses. LangPert is evaluated on Perturb-seq datasets (K562 and RPE1 cell lines) and demonstrates state-of-the-art performance across MAE, MSE, and correlation metrics, outperforming models such as scGPT, GEARS, and GP+LLM. The authors further analyze how LLMs perform context-dependent reasoning, adapting their similarity criteria based on biological context (e.g., pathway, complex, or functional process), and show that stronger LLMs consistently yield better results.

### Strengths
- **Novel hybrid architecture:** Combining LLM-guided reasoning with a numerical aggregator is a clear and effective way to address the high-dimensional limitations of LLMs.  
- **Well-motivated and positioned:** The paper clearly identifies the shortcomings of prior VAE- and transformer-based models (e.g., scGPT, GEARS) that cannot generalize to unseen perturbations.  
- **Strong empirical performance:** Across both K562 and RPE1 datasets, LangPert achieves consistent improvements in MAE and MSE over prior methods, while maintaining strong correlation metrics.  
- **Interpretability and biological faithfulness:** The reasoning examples (e.g., MTOR, EIF3E, PSMD11) show biologically plausible relationships, indicating that LangPert’s outputs are consistent with real biological organization.  
- **Comprehensive evaluation across LLMs:** The comparison across multiple backbones (Claude, OpenAI o1, DeepSeek, Llama) establishes the generality and robustness of the approach.  
- **Good clarity and presentation:** The paper is well structured and clearly written.

### Weaknesses
- **Limited novelty in aggregation:** While the LLM-guided kNN idea is creative, the aggregation component (simple averaging) is rudimentary. More sophisticated probabilistic or weighted methods could potentially improve performance, as the authors also acknowledge.  
- **Dependence on proprietary LLMs:** The method relies on large closed-weight models (Claude, OpenAI o1), which limits reproducibility and accessibility. The reported results for open-weight LLMs indicate a notable performance drop.  
- **Evaluation limited to single-gene perturbations:** Since performance appears largely dependent on the LLM’s knowledge base rather than the aggregation method, it remains unclear whether similar reasoning holds for multi-gene or combinatorial perturbations. Such settings could amplify LLM limitations (e.g., hallucination, reasoning instability).  
- **Lack of prompt and context ablation:** The study would benefit from an analysis of how prompt structure and contextual cues (e.g., cell-line-specific information) influence results. Because K562 and RPE1 are well-studied lines with abundant literature coverage, the LLM’s prior exposure may strongly influence outcomes.  
- **Uncertainty quantification missing:** The results are presented as point estimates, though biological perturbations are inherently noisy. Uncertainty may stem both from the LLM’s retrieval of neighbors and from numerical aggregation; disentangling these sources would make the framework more informative and reliable.

### Questions
1. How sensitive is LangPert’s performance to the number of neighbors $(k)$ and the specific prompt formulation? Were these tuned per dataset or fixed globally?  
2. Have you quantitatively validated whether LangPert’s selected gene subsets overlap with known biological interaction networks (e.g., STRING, Reactome)?  
3. Given that performance depends heavily on LLM reasoning, would incorporating literature-derived embeddings (as in GP+LLM) into the aggregation step improve or destabilize predictions?  
4. How well would LangPert generalize to combinatorial perturbations involving multiple genes, and does the reasoning quality of LLMs degrade in these more complex contexts?  
5. Since fine-tuned open-weight models such as TxGemma still lag behind frontier LLMs, do the authors see potential in retrieval-augmented strategies or hybrid pipelines (e.g., combining smaller LLMs with domain-specific retrieval) to narrow the performance gap without full reliance on closed models?
6. Beyond predictive accuracy, how reproducible are the LLM-derived reasoning traces? For instance, do repeated runs with identical prompts yield consistent gene sets, or is there notable stochasticity?

### Soundness
2

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
4

### Summary
This paper tackles a core problem in perturbation biology:
Given a gene knockout (or other genetic perturbation) that the model has never seen before, can we predict that perturbation’s full transcriptional effect (i.e. the differential expression vector across thousands of genes in that cell type)? 

The authors propose LangPert, a hybrid method that combines:
1. An LLM “reasoner” — given a novel perturbation (e.g. knock out gene X) and a library of previously tested perturbations, the LLM selects a biologically relevant subset of training perturbations that it thinks will behave similarly. This uses biological prior knowledge (literature, pathway membership, complex membership, feedback loops, etc.). 

2. A numerical aggregator (kNN-style) — the model then averages the actual measured expression shifts for those selected neighbors to produce a prediction for the unseen perturbation. No high-dimensional generation is done by the LLM itself.

The authors evaluate LangPert on large Perturb-seq datasets (K562 and RPE1 cell lines, >1000 perturbations each). They compare against:
scGPT fine-tuning, GEARS (a GNN using gene–gene priors), GP+LLM (Gaussian Process using LLM-derived gene embeddings)
A surprisingly strong baseline: the mean non-control response (just average observed shifts from other perturbations)

### Strengths
1. Strong empirical results vs. tough baselines
The paper does not cherry-pick easy baselines. It compares against GEARS (graph prior), GP+LLM (Gaussian Process + LLM-derived embeddings), and even the “non-control mean,” which is known to be obnoxiously competitive in this domain. LangPert consistently wins on MAE and MSE and usually wins or ties on correlation, across two very different cell lines. 
This is the most convincing kind of win: beating “dumb but annoyingly good” baselines.
2. Generalizes to unseen perturbations
Most older VAE-style perturbation models just learn embeddings for perturbations they’ve actually seen. That means they basically can’t extrapolate to a new gene. LangPert can, by construction, because it never needed to learn a continuous latent for that gene — it just reasons about which known genes are most similar and aggregates their measured effects. 
That’s exactly the scientific use case: “I haven’t assayed gene X yet — predict it anyway.”

### Weaknesses
1. Heavy dependence on external LLMs
Performance depends on using a strong LLM (e.g., Claude 3.5 Sonnet). That raises:
Reproducibility concerns under double-blind / future access restrictions.
Fairness concerns: do labs without frontier API access get worse science?
The paper does explore smaller LLMs, which is good, but reproducibility and openness will almost certainly come up in ICLR discussion. 

2. No explicit handling of uncertainty
The kNN-style averaging will always return some answer, even if the LLM’s “neighbors” are a bad match. There’s no uncertainty score, confidence interval, or abstain option. That’s risky in experimental design, where false confidence can waste wet-lab time.
This feels solvable (bootstrap over neighbors, entropy over LLM-selected sets, etc.) but it’s not addressed.

3. Biological scope is still single-gene, steady-state transcriptomics
The paper evaluates only single-gene perturbations in two immortalized human cell lines (K562, RPE1). No combos (A+B knockouts), no time series, no protein-level phenotypes, no spatial interactions. That’s fine scientifically (still a hard task), but the Discussion gestures toward “broad cellular systems modeling,” which is a little ahead of where the method has actually been tested.

### Questions
1. Reasoning vs Retrieval
Since LangPert doesn’t fine-tune the LLM, how much of the gain comes from retrieval-style reasoning versus latent biological knowledge inside the pretrained LLM?
Does the LLM’s reasoning produce genuinely novel pairings or mostly recover known relationships (protein complexes, shared pathways)?

2. LLM Prompting and Stability
How sensitive are results to prompt phrasing, temperature, or the number of retrieved “neighbor” perturbations (k)?
Have you evaluated reproducibility under re-prompting the same LLM (variance across runs)?

3. Metrics and Evaluation
Why limit evaluation to the top 20 DE genes? Does LangPert still outperform baselines genome-wide?
Are results consistent across genes with small but biologically relevant changes (e.g., transcription factors)?
How do errors vary by gene function or perturbation type (TF vs enzyme vs chaperone)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a framework, LangPert, that uses Large Language Models (LLMs) with k-nearest neighbors (kNN) for predicting cellular responses to unseen genetic perturbations.
Instead of directly modeling gene expression vectors, the model uses the LLM to identify biologically relevant perturbations in the training set and aggregates their corresponding expression profiles using kNN. Finally, they show that LangPert outperforms baselines such as GEARS and scGPT.

### Strengths
- It is interesting to see how LLMs could perform as good and in some cases better than models used in this domain.

### Weaknesses
- The contribution and findings of this paper is more suitable for a workshop paper rather than a full conference paper. 
- This paper does not include a systematic study of why the LLM makes certain biological choices.
- The LLMs used in this paper were all trained (or fine-tuned) on large portions of the public internet, biomedical papers, and open-access databases e.g. they might have even been trained on datasets derived from the same cell lines (e.g., K562, RPE1) or papers describing the very Perturb-seq results used in evaluation. So when LangPert asks, “Which genes are similar to MTOR in K562?”, the LLM may be recalling previously reported associations rather than reasoning biologically de novo. This is a very critical problem.
- Some baselines in this domain such as sclambda and PRESAGE are missing in the evaluation setting.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

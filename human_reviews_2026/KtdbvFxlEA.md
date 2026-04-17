# Unlocking Volition: Proactive Intention Decoding via Interpretable Graph Learning of Multi-Region ECoG

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Current brain–machine interfaces (BMI), face fundamental limitations due to inherent latency from reliance on delayed motor cortical signals, and computational overhead, restricting their effectiveness in real-time applications such as rehabilitation therapy. Recent neuroscience indicates prefrontal and sensory cortical activities precede motor execution, offering an opportunity for proactive intent prediction. However, challenges remain in acquiring multi-region neural data, efficiently decoding high-dimensional signals, and ensuring model interpretability. To address these, we developed a high-density electrocorticography (ECoG)-based paradigm based on marmosets and introduced an information-bottleneck-driven graph transformer (ECoG-IBGT), reframing neural decoding as graph classification. Our method achieves 99.29\% accuracy up to 400 ms before action onset with inherent interpretability, laying a foundation for reliable, low-latency BMIs. Code is available at  [*********************************URL Blinded for Review*********************************].

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors developed a high-density ECoG-based framework on marmosets and introduced an information-bottleneck-driven graph transformer for intention detection. Specifically, subgraphs are generated with mutual information estimation learning.

### Strengths
[1] A high-quality, multi-subject, multi-context ECoG dataset\
[2] Comprehensive experiments and evaluations

### Weaknesses
[1] English language – The authors should substantially improve the quality of the manuscript.\
[2] Graph transformer – The authors should cite the work when they mention it in Section 3.3.\
[3] Math notations – The authors should indicate all the notations in their equations. The authors should substantially improve the manuscript quality.\
[4] Motivation – The motivation for applying subgraphs is still unclear.

### Questions
[1] Why did the authors apply a graph transformer instead of using the transformer directly?\
[2] Why did the authors generate subgraphs? Why don’t they just use the full graph for more training efficiency?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes an information-bottleneck graph-transformer for proactive intention decoding from dual-region high-density ECoG. Short pre-onset (vs. rest) windows are converted into functional graphs; the model jointly learns compact node/edge subgraphs for classification and inspection. The writing is clear, and the idea targets lower-latency and more interpretable BMI pipelines.

### Strengths
1. Originality: Reframes proactive decoding as graph classification with a learned subgraph (masking) mechanism rather than post-hoc attribution.

2. Quality: Sensible pipeline design with basic sanity/robustness checks around graph construction and perturbation of important connections.

3. Clarity: The core components (graph building, IB/masking, encoder) are explained cleanly with helpful figures.

4. Significance: If generalizable, anticipatory decoding with compact subgraphs could inform low-latency BMI design and yield testable neuro hypotheses.

### Weaknesses
1. Main results rely on random within-session splits, which can overestimate performance when near-duplicate windows appear across train/val/test. Cross-session evaluation is not foregrounded, so the true ranking of methods (including simple baselines) under shift is unclear. This is especially important considering Supplementary materials provide cross-session metrics for the main model and they show near perfect 100% metrics for validation (same session), but 80% test metrics (differertn session). Combined with (most likely) a large enough model in terms of trainable parameters and 233 minutes of data across all sessions such overfitting might happen and within-session estimation with random splits makes it invisible (as test and train sampes are mixed and can easily be similar to each other)

2. Learned masks and motif plots are primarily associational; stronger validity checks (stability across seeds/sessions, model-randomization, counterfactual edits) are needed.

3. Runtimes are reported on high-end GPU and do not provide end-to-end CPU latency (including graph building) or an asynchronous detection analysis which is a key for practical BMI.

4. The description around “windows,” graph instances, and “node features” can be misread; rest-window sampling and temporal separation need clearer, leakage-resistant definitions.

5. The paper states code is available as supplementary, but the submission lacks an accessible anonymized repo/archive; this blocks verification.

### Questions
1. Please report cross-session performance in the main text for all baselines (incl. EEGNet) with variance across seeds to establish robust ranking.

2. Precisely define rest-window sampling and temporal separation; add blocked/time-shifted splits and leave-session-out/leave-animal-out protocols.

3. Provide mask-stability across seeds/sessions, model-randomization tests, and counterfactual edge/node removals to support causal importance.

4. Report full pipeline latency (acquisition, preprocessing, graph, inference) on CPU-class hardware and an asynchronous detection analysis (e.g., false positives per minute at a fixed TPR).

5. Explicitly state how many graphs are produced per event, how node features are formed, and whether pre-vocal vs. rest graphs are paired or independent. And how many samples are available in total for training and testing splits

6. Supply an accessible anonymized repo/supplement (configs, split scripts, seeds) to reproduce the main tables and cross-session results.

### Soundness
2

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
2

### Summary
The paper proposes ECoG-IBGT, an information-bottleneck driven graph transformer that converts multi-region ECoG windows into functional brain graphs. It learns a compact behavior-relevant subgraph via node/edge soft masks and connectivity loss, and classifies vocalization vs rest to achieve proactive intention decoding up to 400 ms before vocal onset. Experiments on a high-density dual-region ECoG dataset show very high predictive performance (99.29% accuracy) and interpretable subgraphs implicating fronto-auditory motifs.

### Strengths
1. The paper presents a novel framing that reformulates proactive intention decoding as a graph classification problem combined with information bottleneck–based subgraph learning, which aligns well with the characteristics of multi-region ECoG data. The model also emphasizes real-time applicability, reporting low inference latency and highlighting how the graph-based representation can improve efficiency compared to conventional temporal models.
2. The approach achieves interpretability in a principled way through the use of joint node and edge soft masks, a connectivity loss, and an HSIC-based mutual information term, allowing the model to learn compact and behavior-relevant subgraphs rather than relying on post-hoc explanations.
3. The dataset is of high quality, featuring dense dual-region ECoG recordings from freely behaving marmosets, which provides valuable experimental data for the field and demonstrates translational ambition toward brain–machine interfaces.
4. The experiments are thorough, with comparisons against a wide range of baselines and comprehensive ablation studies that clearly demonstrate the contribution of each component in the proposed method.

### Weaknesses
1. The dataset includes recordings from only two subjects, which limits the ability to generalize across individuals and raises the possibility that the model might capture subject-specific features related to electrode placement or physiology.
2. The reported accuracy of 99.29% at 400 ms before vocal onset appears unusually high for anticipatory decoding and may indicate potential data leakage or overly favorable experimental design, particularly if training and test splits were not separated by session or if overlapping windows were used.
3. The graph construction relies on Pearson correlation and a fixed top-10% edge retention threshold, which may be brittle and potentially encode label-related signal differences, yet the paper provides only limited sensitivity analysis of these design choices.
4. The information bottleneck regularization weights are extremely small, and it remains unclear whether the IB loss meaningfully influences optimization or whether the model performance is dominated by the cross-entropy term.
5. The statistical reporting is limited, with only means and standard deviations provided; additional per-subject results, confidence intervals, or p-values for baseline comparisons would strengthen claims of significance.
6. The reproducibility of results could be constrained by data access limitations, since invasive ECoG recordings in primates often require controlled access. This makes it uncertain whether external researchers will be able to replicate the findings. In addition, the authors could release an anonymous code repository for review.
7. Some of the biological interpretations may be overstated, as the identified subgraph motifs do not establish causal connectivity, and the electrode coverage is limited to A1 and PFC, leaving out motor areas that are important for volition studies.

### Questions
1. Please clarify how the training, validation, and test splits were created. Were these splits stratified by session or randomized across trials? It would be helpful to report per-subject and per-session test performance to demonstrate generalization.
2. Please describe the window length used to form each graph and indicate whether the windows overlap between trials. 
3. Please report performance separately for each subject to reveal whether the model’s effectiveness is consistent across individuals or dominated by a single subject’s data.
4. Please provide more details on the perturbation experiments used to validate interpretability. How much performance degradation occurs when top-ranked edges or nodes are removed, and are the identified motifs stable across random seeds or training runs?

Overall, my main concern is that the ultra high accuracy is due to data leakage or unfair experiment settings, since this often happens in this area.

### Soundness
2

### Presentation
3

### Contribution
3

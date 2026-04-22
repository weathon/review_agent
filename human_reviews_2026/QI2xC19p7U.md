# CoPeP: Benchmarking Continual Pretraining for Protein Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
In recent years, protein language models (pLMs) have gained significant attention
for their ability to capture the structure and function of proteins, accelerating
the discovery of new therapeutic drugs. These models are typically trained on large,
evolving corpora of proteins that are continuously updated by the biology community.
The dynamic nature of these datasets motivates the need for continual learning, not
only to keep up with the ever-growing dataset sizes, but also as an opportunity to
take advantage of the temporal meta-information that is created during this process.
As a result, we introduce the Continual Pretraining of Protein Language Models
(CoPeP) benchmark, a novel benchmark for evaluating continual learning approaches on
pLMs. Specifically, we curate a sequence of protein datasets from the UniProt
database spanning 8 years and define metrics to
assess the performance of pLMs on diverse protein understanding tasks. We evaluate
several methods from the continual learning literature, including replay,
unlearning, and plasticity-based methods, some of which have never been applied to
models and data of this scale. Our findings reveal that incorporating temporal
meta-information improves the perplexity over training on the latest snapshot of the
database by up to $20\%$, and several continual
learning-based methods outperform naive continual pretraining. The CoPeP benchmark
presents an exciting opportunity for studying these methods at scale on an
impactful, real-world application.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Here, the authors propose a continuous learning benchmark on UniProt, a large-scale protein sequence database. Their argument is that protein sequence databases expand and remove many sequences from year-to-year, so it makes sense to explore continuous learning approaches on these databases, as opposed to only employing protein language models fixed on a static snapshot of the data, and to benchmark this, they provide versions of UniProt stratified year-to-year from 2015 to 2022, and evaluate a range of continuous learning methods.

### Strengths
The concept behind this paper is quite good: the authors have identified an impactful research gap, where even as protein language models are now highly prevalent and validated to be useful for downstream biological applications, most are trained on a single "snapshot" of existing protein datasets as stands, which does not reflect the continuously evolving nature of these datasets as experimentation stands. Nowadays, these models have scaled to the point that re-training models is extremely expensive. Together, these facts make the argumentation that the authors are advancing in this paper, to understand if continual learning can be applied to pLMs, highly compelling, and this could open up a new line of inquiry that is highly impactful for the field. I appreciate that the authors have curated a dataset that can actually test for this, by providing year-to-year splits of UniRef100, and additionally, the paper is very well-written, communicating the problem in an accessible way.

### Weaknesses
However, I do not find the evaluation and validation to be fully compelling:

1. Their validation split of UniRef is likely not independent enough from their training. 90% sequence identity is very high, and it's probable that at this level of similarity, much of their results are driven largely by memorization of the training dataset - this is evidenced by their values in Figure 3, as 40-50% accuracy on sequence reconstruction is very high. 30-50% identity is a more reasonable threshold for an independent validation dataset.

2. The evaluation benchmark suite is very limited in terms of its task diversity. The authors only assess downstream "generation-like" tasks with their models - accurately reconstructing held-out real sequences and predicting the likelihood of mutations via ProteinGym. However, this does not truly reflect the range of tasks that researchers are interested with protein language models, as it is regular to treat these models as pretrained models that can be fine-tuned or transferred zero-shot to different kinds of functional or structural predictions for proteins (e.g. predicting thermostability, subcellular localization, disordered regions, etc). Many benchmark suites exist for this (e.g. TAPE, FLIP, ProteinGLUE, etc), and this work would better align with downstream applications of pLMs if the authors assessed if continuous learning impacts their role as pretrained models, instead of just assessing their ability to conduct generative tasks aligned with their pretraining. 

3. I'm concerned that the authors' results may reflect the behavior of saturated models, as opposed to generally holding true for pLMs. The authors are using a relatively lightweight model (120M parameters) compared to current models or even those from several years ago (e.g. ESM3 has 98B parameters and ESM2 has 15B parameters). This is apparent from Figure 3, where models clearly stop scaling in performance past ~30M sequences, and possibly even as small as ~15M sequences following their diagonal in Figure 3a. Thus, it is unclear if their main claim in Figure 3, that higher quality data via temporal filtering improves models even if dataset sizes are smaller, holds true in general, or just for saturated models. 

4. I wish there were further investigation of co-variate shifts over time. I suspect that the expansion of the dataset is not happening at uniform, but being informed by technological or topical expansions over time - e.g. I expect that there will be a lot more metagenomics data impacting this in recent years. For this reason, I don't know if improvements on the validation dataset is because that dataset reflects organism composition of UniProt in recent years, and the model has improved on e.g. prokaryotic sequences due to greater representation, while keeping e.g. performance of model organisms relatively constant. To test this, the authors could stratify their validation metrics by domain/kingdom/phylum/class, as well as offer a deeper dive into pretraining dataset composition over time.

### Questions
My suggestions to the authors would be:
1. Repeat their validation set experiments, but with 30-50% sequence similarity instead of 90% sequence similarity.
2. Incorporate functional and structural predictions in their benchmark suite instead of just generational accuracy/likelihood tasks.
3. Train a higher parameter model, at least for their intersection-of-datasets experiment, to confirm that their results are not due to saturation of the dataset.
4. Stratify validation set metrics by domain/kingdom/phylum/class to understand covariate shifts by composition of the dataset over time.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CoPeP, a continual pretraining benchmark for protein language models built from eight yearly UniRef100 snapshots (2015–2022). The authors aim to simulate realistic temporal evolution of protein sequence corpora and evaluate whether models that exploit temporal metadata (e.g., sequence persistence) via continual pretraining can outperform both naïve continual updates and single-year training. Evaluation is conducted primarily on validation perplexity (UniProt) and mutational fitness prediction (ProteinGym). Several continual training strategies (replay, plasticity-based, unlearning) are benchmarked, and the authors conclude that incorporating temporal meta information yields the best performance.

### Strengths
- Try to address a timely and relevant topic: continual adaptation of pLMs as curated biological databases evolve over time.
- Uses real yearly UniRef100 snapshots rather than synthetic domain shifts.
- Temporal metadata exploitation (sequence persistence) is a novel and biologically meaningful idea.
- Includes multiple continual learning strategies (replay, plasticity-preserving, unlearning) with consistent experimental protocol.
- Transparent about engineering details.

### Weaknesses
1. The benchmark problem definition is incomplete and too narrowly evaluated.
Evaluating a protein LM benchmark only on perplexity and a single mutational-fitness metric (ProteinGym) is far from sufficient. Real downstream utility of pLMs spans structure prediction, binding, stability, low-shot generalization, sequence design, etc. A benchmark should reflect that diversity, not just perplexity-like proxies.

2. The proposed notion of “continual learning” is too narrowly framed as purely temporal growth.
Many realistic continual pretraining scenarios involve domain-shift or function-shift across protein families, not only chronological updates from the same database family — which is explicitly covered by prior work (e.g. domain-adaptive continual pretraining). The current framing is not general enough to claim “a benchmark”.

3. The “single-year baseline” is ambiguously defined and methodologically problematic.

- If it uses the full UniRef snapshot for that year (which already implicitly contains past data), its performance should monotonically improve as data grows — but the paper’s plots do not show monotonic behavior, suggesting instability or evaluation inconsistency.

- If it uses only the incremental new-year data, the benchmark fails to compare against a pooled full-data baseline, making it unclear why continual pretraining is preferable to simply training on all available data at once — which is a very strong and standard baseline in practice.
In both interpretations, the foundational baseline comparison is incomplete or misleading.

4. Continual learning definition is not rigorously enforced.
The paper allows re-accessing past data during future stages, which deviates from standard CL constraints where previous data is not accessible.
If replay is allowed, then the paper must explicitly account for increased total token count / compute cost / efficiency, otherwise it undermines the practical motivation for continual learning (i.e., avoiding full retraining).

### Questions
1. How exactly is the single-year baseline defined? Does it use the entire UniRef snapshot of a given year, or only the incremental new sequences? Why is there no pooled full-data baseline to isolate the effect of continuality vs. simply using more data?

2. Can you justify why perplexity + ProteinGym alone is sufficient to define a benchmark for protein LM continual pretraining? Why are no functional downstream tasks (e.g., binding, structural, low-shot or design tasks) included?

3. If past data is accessed during later years (e.g., replay), do you track and control for total token exposure? How is efficiency measured or justified relative to simply fully retraining on all data?

4. How do you ensure that your benchmark is not merely learning increasingly large datasets (or benefiting from accidental leakage in ProteinGym) rather than demonstrating genuine temporal adaptation capability?

5. Do you plan to include domain-shift scenarios beyond temporal evolution (e.g., different protein types or functional subfields), which is a critical part of continual pretraining in real scientific pipelines?

### Soundness
3

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
4

### Summary
The paper introduces CoPeP, a benchmark for continual pretraining of protein language models. CoPeP is constructed from eight consecutive UniRef100 releases (2015–2022), capturing the temporal dynamics of biological databases where proteins are regularly added, removed, or revised. Using the AMPLIFY-120M encoder as a base, the authors evaluate continual learning methods such as replay, Shrink and Perturb, Hare and Tortoise, gradient ascent, and random label unlearning. Evaluation on a curated UniProt validation set and the ProteinGym benchmark shows that leveraging temporal metadata improves performance relative to single-year training, and that certain continual learning methods outperform naive continual pretraining.

### Strengths
The paper introduces a large-scale, realistic benchmark for continual pretraining in protein language models, moving beyond small synthetic datasets commonly used in continual learning research. It evaluates a diverse set of methods, demonstrates the utility of temporal metadata, and highlights performance gains of continual learning approaches over naive baselines. The benchmark is extensible as new UniProt releases become available, making it a good long-term resource for protein modeling communities.

### Weaknesses
- Baseline fairness and clarity: The paper does not clearly explain how baseline numbers (AMPLIFY-1M, single-year training) were obtained or whether training budgets, data exposure, and deduplication policies were matched across methods. In particular, it is unclear which dataset AMPLIFY-1M was trained.
- Potential underoptimization: The continual training setup shows performance gains simply from sequential exposure to data. This raises the concern that the 2015 baseline model used for comparison may be underoptimized. A stronger control would be training the 2015 model longer on the same year’s data to determine whether improvements are due to continual learning or just additional optimization budget.
- Evaluation scope: The empirical evaluation is limited to UniProt validation and ProteinGym, both of which are valuable but narrow in scope. These benchmarks primarily assess natural distribution modeling and mutational effect prediction. Broader evaluation on diverse protein tasks, such as secondary structure prediction, remote homology detection, or sequence design (e.g., TAPE, PEER), would provide a more comprehensive assessment of generalization.
- Potential distribution bias: The training corpus was deduplicated against the validation set using a 90% sequence identity threshold, which remains relatively permissive. This raises the possibility that models benefit from near-duplicate sequences across years, especially in the temporal replay setting. It is unclear whether reported improvements are partly due to sequence similarity rather than genuine knowledge transfer across time. Many protein benchmarks use stricter thresholds like 30–40% sequence identity to ensure true novelty between train and test sets.

### Questions
- Could you clarify how AMPLIFY-1M and single-year baselines were trained? Specifically, which dataset was used for AMPLIFY-1M, and were training steps, data exposure, and deduplication policies matched across baselines and continual methods?
- Have you compared the continual training setup against a baseline where the 2015 model is trained longer on the same year’s data?
- Have you considered benchmark beyond UniProt validation and ProteinGym, for example by including tasks such as secondary structure prediction, remote homology detection, or sequence design?
- Could provide quantitative analysis of sequence overlap across years and against the validation set? Would stricter deduplication thresholds (e.g., 30-40% identity) change the observed performance trends?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper discovers that many proteins evolve over years, and it is necessary to recognize the temporal and sequential nature of proteins so as to capture the evolutionary process. To achieve this goal, this paper proposes a  Continual Pretraining of Protein Language Models (CoPeP) benchmark to assess existing protein language models. Specifically, authors create a sequence of protein datasets and define evaluation metrics. Experiments are conducted to show the performance of existing approaches on the proposed benchmark.

### Strengths
1. Overall, the paper is well written, with figures as visual illustrations. The Introduction section clearly explains the motivation behind the benchmark. It also makes a comparison to existing works and identifies their drawbacks.

2. Proposing a benchmark to capture the temporal and sequential nature of proteins is novel and significant to me. It is important to have such benchmark to advance research community and open future research to study evolutionary process of proteins.

3. Experiments are comprehensive with different methods. Analysis is also provided for readers to gain insightful understand behind numerical results.

### Weaknesses
1. Usually when we do experiments, we encourage authors to repeat the same experimental setting multiple times and report both mean and standard deviation. However, this paper shows mean but not stddev, which is difficult for readers to judge how significantly the proposed method outperforms baselines.

2. Authors are suggested to provide more evaluation metrics to comprehensively test the proposed benchmark, such as property prediction for newly discovered proteins given previously known proteins.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

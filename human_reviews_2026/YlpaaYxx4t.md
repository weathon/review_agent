# Detecting Data Contamination in LLMs via In-Context Learning

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
We present Contamination Detection via Context (CoDeC), a practical and accurate method to detect and quantify training data contamination in large language models. CoDeC distinguishes between data memorized during training and data outside the training distribution by measuring how in-context learning affects model performance. We find that in‑context examples typically boost confidence for unseen datasets but may reduce it when the dataset was part of training, due to disrupted memorization patterns. Experiments show that CoDeC produces interpretable contamination scores that clearly separate seen and unseen datasets, and reveals strong evidence of memorization in open-weight models with undisclosed training corpora. The method is simple, automated, and both model- and dataset-agnostic, making it easy to integrate with benchmark evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a method called CoDeC (Contamination Detection via Context) for detecting dataset-level contamination. The key idea is simple yet effective, for unseen data ICL examples will boost its confidence, so Delta (=ICL - no ICL) would be positive, but for trained data the ICL examples could be confusing hence result in negative Delta. The authors conducted wide range of experiments to support their method.

### Strengths
1. Clean and simple idea. I always appreciate simple ideas that work, and this paper's idea is novel and effective. 
2. Extensive experiments. The authors conducted experiments on a wide range of model families (even for RWKV ones), all with strong experiment results.
3. Well-written and of importance to the field. The writing is clear and the topic is important to the community.

### Weaknesses
I have 2 key questions:
1. Does CoDeC work when you rephrase the question? (either train on rephrased versions then test on original, or train on original then test on the rephrased) I think it's getting widely known that many contamination cannot detect contamination unless it's under exact same phrasing, and arguably the value is low if a method cannot work for general use cases.
2. Did you try training with a mix of other datasets in your experiments? i.e. not only the contamination test set, but with general corpus? This might improve models general ICL ability and render CoDeC ineffective, and people usually train their model this way instead of only using the test set.

### Questions
1. Does model's capability have a correlation with the method effectiveness? Could you also report the general eval metrics (for benchmarks other that the contaminated data) of the models before and after finetuning?

### Soundness
3

### Presentation
3

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
This paper introduces Contamination Detection via Context (CoDeC) — a simple, scalable method for detecting and quantifying training data contamination in large language models (LLMs) using in-context learning dynamics.


The key idea

* When given in-context samples from a dataset it has not seen during pre-training, an LLM tends to increase its confidence on the target sample.
* When the dataset has been seen (i.e., contaminated), additional context disrupts memorization and reduces confidence.

The method computes a contamination score by measuring the fraction of samples for which adding in-context examples decreases the model’s confidence.


Main contributions

* Proposes CoDeC — a model- and dataset-agnostic contamination detection method requiring only gray-box access to logits.
* Demonstrates near-perfect separation (AUC ≈ 99.9%) between seen and unseen datasets across various models (Pythia, GPT-Neo, RWKV, OLMo, Nemotron).
* Shows robustness to dataset diversity and training stage, and scalability to large benchmarks.
* Highlights how contamination transfers across related datasets and how finetuning affects CoDeC scores.
* Positions CoDeC as an interpretable, efficient alternative to classical membership inference methods.

### Strengths
Originality

* While prior work relies heavily on membership inference, loss-based calibration, or reference models, CoDeC offers a novel formulation by exploiting in-context learning behavior as a contamination signal.
* The idea is conceptually elegant — turning a standard model property (ICL behavior) into a contamination test.

Quality

* The experiments are extensive: multiple models (spanning architectures and sizes), datasets (training vs unseen), and baselines.
* Ablation studies are included (context size, dataset size) and finetuning experiments provide additional validation.
* Clear definition of the contamination score and rationale for why it works.

Clarity

* The paper is very well written, with clean exposition of the problem statement and intuition.
* Figures are informative and minimalistic, making the method easy to grasp.
* The pipeline (Fig. 1) clearly communicates the method steps.

Significance

* Data contamination detection is increasingly critical for fair evaluation of LLMs.
* A method that is scalable, interpretable, and does not require access to training data or reference models is very valuable.
* The simplicity and efficiency of CoDeC make it applicable at scale, which is practically important.

### Weaknesses
Experiment rigor to test the generality of the findings

* The paper claims with the help of CMA that LLMs 
* Similar unsubstantiated claims appear in Section 2.2 (Key Idea), e.g., that adding in-context samples usually improves confidence for unseen datasets — no citations or empirical backing are provided there.

Limited theoretical grounding

* While intuition is discussed in depth (Section 2.5), the theoretical explanation remains qualitative.
* Formal guarantees or bounds on false positive/negative rates for contamination detection would strengthen the contribution.

Evaluation scope

* Although contamination transfer across related datasets (e.g., MMLU) is explored, adversarial or edge cases (e.g., near-identical but unseen datasets, noisy mixtures) are only lightly discussed.
* The method might conflate contamination with related distribution overlap, as the authors acknowledge, but do not quantitatively evaluate how severe this is.

Novelty relative to related work

* Some elements overlap conceptually with loss-based or entropy-based detection. Positioning CoDeC more clearly in terms of unique advantages and trade-offs would help.

### Questions
Empirical clarification

* Can the authors provide empirical evidence or references to support the claims in Section 2.2 and Section 3.5 regarding confidence shifts and generalization ability?
* Can they quantify how CoDeC correlates with generalization metrics across models with similar benchmark accuracy?

Contamination vs distributional similarity

* How robust is CoDeC to partial contamination or to unseen but stylistically similar datasets?
* Could the authors provide additional results where datasets are synthetically perturbed to vary similarity to training data?

Thresholding strategy

* The paper mentions thresholding and comparison against other models. Could the authors elaborate on practical guidelines for selecting these thresholds in real-world evaluation scenarios?

Theoretical properties

* Do the authors have any theoretical guarantees (e.g., consistency or error bounds) on CoDeC scores? Or atleast outline what such analysis might require?

Reproducibility and usage

* How sensitive is the method to the number of in-context samples and the randomness of context selection?
* Would deterministic context selection (e.g., nearest neighbors) further improve stability?

### Soundness
2

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
3

### Summary
The paper introduces CoDeC, a dataset-level method to detect training-data contamination in LLMs by measuring how few-shot, same-dataset context changes model confidence on a target sample. If added context improves confidence, the dataset is likely unseen; if it lowers confidence, the model likely memorized the dataset or closely related data. The per-sample confidence shift Δ(x) is aggregated into a dataset contamination score SCoDeC(D) = (1/N)∑1[Δ(x)<0], requiring only two forward passes per sample and gray-box access to token probabilities. The authors show near-perfect separation of seen vs. unseen datasets (dataset-level AUC ≈ 99.9%) across many models, and provide analyses on training dynamics, finetuning-induced contamination, and robustness.

### Strengths
1) Clear, elegant idea with strong intuition: leverage in-context learning as a probe—unseen datasets benefit from added context, while memorized datasets get perturbed—yielding a direct, interpretable signal. 

2) Practicality & efficiency: model- and dataset-agnostic; only two forward passes per sample; no threshold tuning or access to training corpora needed. 

3) Simple, interpretable metric: a percentage score (fraction of samples with negative Δ) that aligns with practitioner intuition and enables straightforward ranking across datasets. 

4) Strong empirical evidence: near-perfect dataset-level AUC over diverse models; analyses show early emergence of contamination in training, finetuning-induced contamination, and size trends (larger models memorizing less). 

5) Scope beyond strict membership inference: also detects contamination via related/shadow distributions, broadening practical utility for benchmark hygiene.

### Weaknesses
1) Adversarial/degenerate datasets: repeated or highly heterogeneous mixtures can distort the score; the paper notes such edge cases but they remain a limitation for fully automatic use. 

2) Calibration to “absolute” labels: SCoDeC is excellent for ranking datasets by contamination risk, but stakeholders may still desire thresholded decisions (ACC/PR/F1). Adding a recommended thresholding recipe could help some users. 

3) Model-family anomalies: certain heavily instruction-optimized models behave atypically under CoDeC (e.g., chat/trace behaviors perturb logits broadly), suggesting architecture/task biases may require special handling.

### Questions
1) Could you provide a small, principled thresholding guide (e.g., validated percentiles on held-out datasets) for users who need binary “contaminated/not” decisions in audits? 

2) How does SCoDeC behave under various context sizes n across different dataset lengths/formats? A brief cost–benefit curve would aid deployment. 

3) For instruction-tuned/chat-optimized models that yield universally high scores, can lightweight decoding settings (e.g., stop tokens, no-think modes) mitigate anomalies?

### Soundness
3

### Presentation
3

### Contribution
3

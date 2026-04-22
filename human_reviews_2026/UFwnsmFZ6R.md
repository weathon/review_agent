# Rethinking Data Curation in LLM Training: Online Reweighting  Offers Better Generalization than Offline Methods

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 6, 4

## Abstract
Data curation is a critical yet under-explored area in large language model (LLM) training. Existing methods, such as data selection and mixing, operate in an offline paradigm, detaching themselves from training. This separation introduces engineering overhead and makes the curation brittle: the entire pipeline must be re-run under model/task shifts. Moreover, offline methods alter data size through hard filtering or resampling, often sacrificing data diversity and harming generalization. We propose to rethink data curation as an online reweighting problem, where sample importance is dynamically adjusted during training via loss weighting rather than static pre-processing. Specifically, we introduce ADAPT (Adaptive Data reweighting for Pretraining and FineTuning), a dynamic online framework that reweights training samples with adaptive per-sample learning rates guided by similarity-based quality signals, without changing the number of training samples. Unlike offline methods that enforce a static data distribution, ADAPT acts as an implicit curriculum learner, progressively shifting focus from coarse-grained patterns to fine-grained semantic distinctions as the model evolves. Experiments on both instruction tuning and large-scale pretraining show that ADAPT consistently outperforms offline selection/mixing and prior online methods, achieving stronger cross-benchmark generalization under equal FLOPs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper propose an online reweight mechanics for LLM training samples. The author conduct various experiments to demonstrate the effectiveness of the algorithm.

### Strengths
1. The paper is easy to read and understand.
2. The idea is clear and easy to follow.

### Weaknesses
1. This paper claims a lot contribution of "online" as stated in the title, but there exist several online reweight algorithms. For example [1] and [2]. The fair comparison should be made between proposed algorithm and other online algorithms. The storytelling is misleading such comparison is absent.
2. The improvement is limited. As we can see in Table 2, ADAPT improves a little comparing against other baselines. For example, ADAPT (38.00) is a little better than RegMix (37.97). The improvement 0.03 might be under standard deviation range.
3. The experimental setting is limited. The author is encouraged to perform more experiments across different model sizes and architectures to test algorithm robustness.


[1] Importance weighting can help large language models self-improve
[2] Take the Bull by the Horns: Hard Sample-Reweighted Continual Training Improves LLM Generalization

### Questions
Please check the weakness part.

### Soundness
2

### Presentation
3

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
This paper proposes ADAPT, a method for selecting influential data, selecting mixtures of data, and reweighting data -- all in one framework. They do this by defining a function $s(\cdot)$ that will assign a score to an input. In data selection, samples are selected by applying a threshold on the scoring function (only data that has a high score will be chosen). In data mixing across the domain level, the scoring function is used to determine how much weight a domain (cluster of samples, perhaps) should be given. They do this by transforming the score with the $exp(\cdot)$ function and then normalize the scores across domains. Finally, in (online) data reweighting, the scoring function is used to determine how much to scale the loss of the sample by during gradient descent. The paper shows competitive performance compared to baselines.

### Strengths
- The problem of unifying online and offline data curation is a timely problem
- The paper is well organized and written
- The experimentation is extensive and thorough

### Weaknesses
- According to Figure 2a, ADAPT-BM25 and ADAPT don't do much better than LESS, with the same number of FLOPs
- According the Figure 2a/2b, with fewer FLOPs, BM25 is able to beat most baselines and ADAPT. Which means that under a tighter FLOPs budget, a simpler method like BM25 could achieve best performance (and it also generalizes to different datasets well)
- I might have missed it, but I could not find any analysis on the qualitative results of data selection, mixing, and reweighting

### Questions
1. In the "Setup" paragraph in Section 3, lines 140-146: what do you mean by "domain"? If "in-domain" and "out-of-domain" are interchangeable with "in-distribution" and "out-of-distribution", why does $D_{train}$ contain in-domain _and_ out-of-domain. Since the model has access to the training set, doesn't that automatically mean that all training data is in-domain? If "domain" is not the same as "distribution", how do you define and determine "domain"?

### Soundness
3

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
3

### Summary
The paper argues that many existing “offline” data-curation pipelines (selection/mixing) are brittle and compute-inefficient when judged end‑to‑end, and proposes reframing curation as online sample reweighting integrated into training. The method, ADAPT, implements per‑sample learning‑rate multipliers via similarity‑based scores to a validation/anchor set, including a variant that uses the model’s own evolving representations (Eq. 7–10). Empirically, ADAPT is claimed to provide a better accuracy‑vs‑FLOPs trade-off than offline selection/mixing and prior online reweighting, on instruction tuning (Llama‑2‑7B with LoRA) and pretraining (TinyLlama‑120M on SlimPajama). Key visuals include the trade-off plots (MMLU/BBH vs FLOPs, Fig. 1 & Fig. 2) and tables showing small but consistent average gains (Tables 1–3)

### Strengths
- Unified view and cost accounting. The paper formalizes selection, mixing, and reweighting under one umbrella and compares them with a total FLOPs accounting that includes pre‑/in‑loop costs (Eqs. 5–6 on p.4; Sec. 3.4). This framing is valuable for the community, where “hidden” preprocessing costs are often ignored

- Simple, generally applicable mechanism. Casting curation as per‑sample learning‑rate scaling (Eq. 7) is elegant and easy to integrate in standard optimizers; using model-state representations (Eq. 9–10) to compute similarity is a sensible way to keep the weighting adaptive as training proceeds (Sec. 5.1–5.3).

- Cross-setting evaluation. The paper runs both instruction tuning and pretraining experiments and explicitly studies in‑domain vs out‑of‑domain generalization (e.g., MMLU‑val → MMLU‑test vs BBH, Fig. 1 & 2; Table 1), which aligns with the paper’s motivation about brittle offline curation

- Interpretable diagnostics. The distributional views (Fig. 3a–b on p.8) help illustrate how weighting changes the effective mixture and similarity landscape across epochs, consistent with an implicit curriculum narrative.

### Weaknesses
1. Compute accounting vs “negligible overhead.” The main text states ADAPT “incurs nearly zero additional overhead” (Sec. 1–2), but Appendix C quantifies ADAPT as 2k × 8N × P × E vs training’s typical 6N per token—i.e., ~33% extra FLOPs for ADAPT relative to a 6N baseline, which is not negligible. This discrepancy should be reconciled with a precise breakdown of where the +2N arises (similarity scoring, extra passes, etc.), and whether the same penalties are charged to other baselines (e.g., BM25 index building).

2. Use of evaluation data as anchors / potential leakage. For pretraining, ADAPT’s anchor/validation pool is built by sampling from the same downstream evaluation benchmarks (e.g., ARC‑C, COPA, MultiRC, etc.; Sec. 6.1, p.6–7). Even if only “validation” items are used, this design can bias models toward those tasks, weakening claims of broad generalization. A variant where anchors come from disjoint corpora (or different task families) is important to rule out target‑task leakage.

3. “100× fewer samples” and compute parity are under‑specified. The paper emphasizes using 100 unique examples but keeps training time nearly identical to R1 (15h37m vs 15h23m on Qwen3‑0.6B; §4) and shows only modest time savings versus AdaRFT (2–11%; Figure 13). It appears SPaRFT reduces unique training items but not the number of optimization steps. This weakens the “minimal resources” narrative and invites a stronger compute‑parity study (equal steps, equal tokens, and equal number of unique items for baselines).

4. On pretraining, ADAPT improves the average by 0.19–0.38 points over Uniform (Tables 2–3, pp.9), and performance is mixed on individual tasks. It’s not evident that these differences are robust: #seeds and statistical tests are not specified for Tables 2–3. For instruction tuning (Table 1, p.8), error bars are shown, but again the number of seeds and test methodology are unclear; several intervals overlap (e.g., ADAPT vs full SFT on BBH). Without significance analysis, claims of superiority are tentative.

5. Sec. 5 references “smooth gating,” normalization, and clipping but does not specify the exact functions, schedules, or hyperparameters (e.g., temperature, clipping thresholds, batch vs global normalization). Reproducibility would benefit from pseudocode and explicit details.

6. Baseline fairness and configuration. For DoReMi and RegMix, the paper states it uses domain weights from Lu et al. (2023) as selection ratios (Sec. 6.1). That sounds closer to using fixed weights than actually running the optimization procedures those methods prescribe, which could understate their performance relative to ADAPT. Please clarify and, ideally, run the baselines as originally intended. The LinUpper configuration and its cap parameter α could heavily impact performance; more detail and a parameter sweep would help substantiate the negative result (Table 2).

### Questions
1. Overhead accounting. Please reconcile the “nearly zero overhead” claim with ADAPT = 8N vs 6N FLOPs per token in Appendix C. What exact computations contribute to the +2N, and are equivalent costs charged to offline baselines (e.g., BM25 indexing, embedding passes)?

2. Anchors vs evaluation data. How much performance drops when anchors are drawn from unseen, disjoint distributions (e.g., natural‑language corpora not overlapping with evaluation tasks)? This is vital to support the generalization narrative.

3. Method details. What are the precise gating/normalization/clipping functions (and hyperparameters), and are weights normalized per‑batch or globally? Could you include pseudocode?

4. Baseline implementations. Did you run DoReMi/RegMix as originally proposed (with mixture‑weight optimization), or did you only apply fixed weights from another paper? If the latter, can you report results from the full algorithms?

5. Ablations. Can you provide ablations on (i) representation pooling (Eq. 9) vs mean pooling, (ii) anchor‑set size/composition, and (iii) choice of similarity metric (BM25 vs embedding vs model‑state cosine)?

### Soundness
2

### Presentation
3

### Contribution
3

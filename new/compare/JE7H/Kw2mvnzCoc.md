---
job_id: c579fb16-b837-4e05-9a5d-f14834c9f275
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: Kw2mvnzCoc.pdf
paper: TSPULSE: Tiny Pre-Trained Models With Disentangled Representations for Rapid Time-Series Analysis
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a self-supervised pre-trained model for time-series, focusing on representation learning, disentanglement across temporal/spectral/semantic spaces, and transfer to multiple downstream tasks, which is squarely within ICLR’s scope.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work in Appendix A.6, Method/Architecture, Experiments, Results, Conclusion) are present and sufficiently detailed. The method is nontrivial, the experiments are extensive (four tasks, many datasets), and there is no obvious fatal methodological flaw or test leakage. The work is clearly positioned as a new pre-training framework plus architecture for TS diagnostics, with quantitative evidence.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no hidden prompts, attempts to influence LLM-based reviewing, or other manipulative content in the main paper text.

---

# Expected Review Outcome:

## Summary

The paper introduces TSPulse, a family of ~1M-parameter pre-trained models for time-series diagnostic tasks (anomaly detection, imputation, classification, similarity search). The core idea is a disentangled masked reconstruction framework that operates jointly in time and frequency domains, producing three distinct embedding “views”: temporal, spectral (FFT), and semantic (register) embeddings, trained with multi-head objectives.

The authors also propose lightweight task-specific “post-hoc fusers” (TSLens for classification and Multi-Head Triangulation for anomaly detection), as well as a hybrid masking strategy for pre-training that mixes block and point-level masks to better match real-world missingness. Extensive experiments on TSB-AD, UEA, LTSF imputation, and a similarity search benchmark claim substantial gains over much larger pre-trained models and strong data-specific baselines, together with detailed efficiency analysis and sensitivity studies of disentanglement.

---

## Strengths

1. **Well-motivated disentanglement across spaces and abstraction levels, with clear architectural mapping.**  
   The paper makes a compelling case that different TS diagnostic tasks need different forms of information: local timing, frequency content, and global semantics. This is reflected concretely in the architecture in **Figure 2**, where:
   - Time patches and FFT patches are encoded separately and concatenated with register tokens.
   - The decoder output is split into three contiguous segments corresponding to time, FFT, and semantic tokens, each tied to specialized heads (full reconstruction vs spectral vs signature/next-point).  
   This is an intuitive and reasonably principled way to implement “soft disentanglement” without overly complex constraints, and the sensitivity analysis (Section 6, Table 2) demonstrates that the resulting embeddings actually behave differently under perturbations.

2. **Strong, broad empirical performance across four tasks and many datasets.**  
   - **Anomaly detection:** On TSB-AD (univariate & multivariate), **Figure 4** and **Table 13** show TSPulse (ZS) beating all 40 methods, including SubPCA, CNN, Moment, Chronos, TimesFM, etc., with +14–16% VUS-PR improvements in zero-shot and +24–26% after self-supervised fine-tuning. The full dataset-wise tables (Tables 14–15) confirm that the gains are not limited to a single dataset.  
   - **Classification:** On 29 UEA datasets, **Figure 5** and **Table 17** show mean accuracy 0.733 vs 0.701 (VQShape), 0.675 (Moment), and 0.634 (UniTS), plus 5–12% gains over contrastive baselines (TS2Vec, T-Rep, etc.), with a model that is an order of magnitude smaller.  
   - **Imputation:** **Figure 6** and **Table 19** report zero-shot MSE ~0.074 under hybrid masking, outperforming Moment (large) by >70% and UniTS (prompt-tuned) by ~50%, and even beating fine-tuned strong supervised models (TimesNet, FEDformer, Non-Stationary Transformer) in some settings.  
   - **Similarity search:** **Figure 7** and **Tables 28–29** show +25–40% absolute gains in PREC@3 / MRR@3 over Moment, and roughly 2× over Chronos, across synthetic and real benchmarks.  
   The breadth and consistency of improvements, especially in strict zero-shot scenarios, is a nontrivial contribution.

3. **Careful, quantitative analysis of the claimed disentanglement properties.**  
   Section 6 and Appendix A.3–A.4 go beyond hand-wavy statements. The distortion metrics in **Equations (1)–(3)** quantify robustness of each embedding type to missingness, noise, and phase shifts. **Table 2** (and Tables 4–7) show:
   - Time embeddings highly sensitive to phase/time shifts (130% distortion), consistent with fine-grained temporal alignment.
   - FFT embeddings less phase-sensitive but more sensitive to masking and noise.
   - Semantic (register) embeddings most robust overall and minimally affected by phase, matching their intended role for search and high-level tasks.  
   Additional PCA plots in **Figures 8–12** dissect sensitivity to amplitude scaling, noise, trend, shape class, and missing data; the semantic embeddings cluster primarily by frequency/shape rather than nuisance factors. This is significantly more thorough than what most “disentangled representation” claims provide.

4. **Hybrid masking addresses a real and under-discussed bias in imputation pre-training.**  
   The hybrid masking scheme (Section 2, “Masking & RevIN” and Section 4.3) is well-motivated: block-only pre-training leads to models that work well only when test-time missing spans match the pre-training pattern. The ablations in **Table 1(c)** and **Table 23** are sharp:
   - Removing dual-space learning degrades ZS imputation by ~8%.
   - Pre-training **without** hybrid masking (block-only) causes a 79% error increase under hybrid evaluation (MSE from 0.074 → 0.354), while slightly improving block-only evaluation.  
   The MAR/MNAR stress tests in **Table 31** further show large robustness margins over Moment in realistic missingness settings. This is a concrete and practically relevant insight.

5. **Task-specific post-hoc fusers are simple yet empirically important.**  
   - TSLens (Section 3.3, **Figure 3-C**) is just an MLP-based feature reducer + linear head over the disentangled embeddings, but **Table 1(b)** shows that replacing it with naive average pooling or max pooling drops mean UEA accuracy by 11–16%.  
   - Multi-Head Triangulation (Section 3.3, **Figure 3-D**) uses separate deviation streams from time, FFT, and prediction heads, and either statistical fusion or head selection. **Table 1(a)** shows that the triangulation strategy provides higher VUS-PR than any individual head, and the tuning-based head selection is backed by TSB-AD’s official tuning split.  
   These are low-complexity modules that nonetheless make clear, measurable differences.

6. **Efficiency claims are backed by concrete numbers and configurations.**  
   **Table 3** shows per-batch GPU/CPU inference times and memory against GPT4TS/OFA, VQShape, Moment (small/base/large), UniTS, Chronos, etc., under a fixed input shape. TSPulse is consistently:
   - 4–56× faster on GPU, 4–29k× on CPU vs various baselines.
   - 2–33× smaller in parameters vs most models (and 40–300+× vs several).  
   In the similarity search section, these efficiency gains are also tied to end-to-end retrieval latency and embedding size; the model is clearly deployable on CPU.

7. **Extensive ablations tying objectives to task performance.**  
   - **Table 1(b)**: classification ablations over 17 UEA datasets systematically quantify the impact of each architectural element: removing short/long embeddings, masking in fine-tuning, identity initialization of channel mixers, channel expansion, TSLens, and dual-space learning.  
   - **Table 22** and **Table 23–24**: imputation ablations show that hybrid pre-training and dual-space learning are not cosmetic; they produce large performance swings.  
   - **Table 26**: similarity-search ablations confirm register embeddings and hybrid pre-training dominate; using only time or FFT embeddings cuts PREC@3 by ~50–70%.  
   These studies strengthen the causal narrative around the design choices.

---

## Weaknesses

1. **“Disentanglement” is purely emergent and lacks an explicit independence or factorization objective.**  
   The model calls its heads/segments “disentangled” because time, FFT, and register embeddings are optimized against different reconstruction/signature tasks (Section 2, “Multi-Objective Heads”), and the sensitivity analysis suggests qualitatively different behaviors. However, there is no explicit regularizer enforcing statistical independence (e.g., adversarial disentanglement, factorized priors, or mutual information penalties) across segments. In particular:
   - The loss is just a weighted sum:  
     \(\mathcal{L} = \lambda_1 \mathcal{L}_{\text{time1}} + \lambda_2 \mathcal{L}_{\text{time2}} + \lambda_3 \mathcal{L}_{\text{fft}} + \lambda_4 \mathcal{L}_{\text{sign}} + \lambda_5 \mathcal{L}_{\text{pred}}\),  
     but no term encourages the three embedding types to be decorrelated, orthogonal, or otherwise disentangled.  
   - The TSMixer backbone fully mixes across time/FFT/register tokens (Figure 2, Block 3), so shared representations can propagate freely.  
   The paper’s empirical evidence for emergent disentanglement is strong, but for a paper that leans heavily on “disentangled” in the title and narrative, it would be useful either to (a) introduce a simple regularizer (e.g., cross-covariance penalties between embedding segments) or (b) explicitly reframe the claim as “specialized views” rather than strict disentanglement.

2. **Some mathematical and notational issues that need cleaning and clearer specification.**  
   - In Appendix A.10, the equation  
     \(\mathbf{X}_m = \mathbf{X}_m+\mathbf{X}_{m+1}+\mathbf{X}_m.\)  
     appears to be a typographical error and is not used elsewhere; this is confusing in a core section explaining FFT extraction.  
   - The masking procedure in Section 2 is not fully formalized: the paper explains qualitatively how hybrid masking mixes full and partial patch masking and how the raw-level mask token \(\mathbf{M} \in \mathbb{R}^{1 \times pl}\) is re-used for individual positions, but the exact probability distribution over mask patterns (e.g., fraction of block vs point masks, sampling strategy within a patch) is left vague. This matters because hybrid masking is claimed as a central innovation driving imputation and MAR/MNAR robustness.  
   - In the definition of \( \mathbf{X}^f_{\text{sign}} \) in Appendix A.10, the softmax-normalized log-magnitude spectrum, the paper does not specify clearly across which axis softmax is applied (per channel across frequency bins is implied but should be explicit) and how this interacts with the FFT Head vs Signature Head dimensions.

3. **Lack of stronger baselines for “lightweight” and “disentangled” time-series representation learning.**  
   While the paper compares against large pre-trained models (Moment, UniTS, Chronos, TimesFM, Lag-Llama) and classic contrastive/autoencoder baselines (TS2Vec, T-Loss, TNC, TS-TCC, etc.), it omits more recent or directly related TS disentanglement work in the **main comparisons**:
   - For example, **TimeDRL (Chang et al., 2024)** is only mentioned in the related-art discussion but not used as a baseline on any classification benchmark where it would be applicable.  
   - Similarly, TF-C (Zhang et al., 2022) and BTSF (Yang & Hong, 2022) explore temporal-frequency fusion (and even disentanglement vs entanglement), but are not quantitatively compared. Given that TSPulse’s main conceptual claim is “disentangled multi-space representation helps downstream tasks”, adding at least one such method to **Table 17** (classification) or the similarity-search evaluation would make the positioning much sharper.  
   At minimum, the authors should clarify computational constraints, input-length compatibility, or other reasons for excluding these baselines.

4. **Head selection for anomaly detection might overfit tuning data and is not compared to equally tuned ensembles.**  
   Multi-Head Triangulation for AD uses a tuning set (with labels) to select the best scoring head among Time, FFT, Pred, and Ensemble, per dataset (Appendix A.11.3, Tables 11–12). This is allowed by TSB-AD’s protocol, but two questions arise:
   - Many baselines on TSB-AD likely used tuning for hyperparameters but not per-dataset model/head selection; so TSPulse might be benefitting from an additional degree of freedom (head choice) that others do not exploit.  
   - The “ensemble” head uses max across streams; if we allowed a small linear or logistic regression combiner tuned on the same validation labels, could that outperform head-selection and better calibrate scores? Currently **Table 1(a)** only reports individual heads vs the simple triangulation scheme, not vs more flexible ensemble methods under equal tuning budgets.  
   Without more controlled comparisons, it is hard to disentangle how much of the AD gain is due to the pre-training itself versus the relatively rich tuning strategy.

5. **Task-specialized pre-training vs generalist models: fairness and practical trade-offs.**  
   TSPulse explicitly pre-trains separate models for each target task, reweighting loss objectives (Section 3.1, Appendix A.9). In contrast, many baselines (Moment, UniTS, GPT4TS) aim to be generalist. While the paper does include a “unified” TSPulse experiment in **Table 30**, several points remain under-explored:
   - The unified TSPulse still uses the imputation/search-optimized pre-training, not a jointly optimized multi-task objective across all heads. This may not fully test how far TSPulse can go as a single general-purpose model.  
   - There is no analysis of pre-training compute vs benefit: pre-training 3–4 specialized 1M models might still be cheaper than one massive 300M model, but this trade-off is not quantified.  
   - Some baselines (UniTS) also support task-specific prompts and heads; a fairer comparison would consider minimal prompt-tuning or head-specialization for them per task, not just a single configuration.  
   None of this invalidates the core results, but the practical implications of “one tiny model per task” vs “one big generalist” deserve deeper, more quantitative discussion.

6. **Limited exploration of multivariate aspects during pre-training.**  
   The model is pre-trained in a univariate fashion (Section 3.1), treating each channel independently, and only introduces channel mixing during fine-tuning via identity-initialized channel mixers. While this simplifies data handling, it raises two concerns:
   - For tasks where inter-variable dynamics are essential (e.g., multivariate AD on TSB-AD-M, multivariate classification like PEMS-SF and MotorImagery in Table 16), there is no evidence that univariate pre-training is competitive with truly multivariate pre-training given the same parameter budget.  
   - The ablation in **Table 1(b)** shows channel expansion and identity-initialized mixers help, but there is no comparison against a model that is pre-trained with channel mixers enabled from the start (even on synthetically padded channels).  
   This is an opportunity: minimal multivariate pre-training might further enhance performance and could be worth testing or at least discussing.

7. **Some experimental design choices could be clarified further to avoid perceived cherry-picking.**  
   A few examples:
   - In imputation, the UniTS baseline is prompt-tuned on 10% data while TSPulse is zero-shot; this is fair in the sense that it favors the baseline, but the hyperparameters for UniTS (and Moment) under the hybrid masking variant are somewhat ad-hoc (Table 25, “Heterogeneous vs Only-Zeros”), and it is not obvious that the best setting for each baseline was exhaustively searched.  
   - In classification, VQShape experiences an OOM on InsectWingbeat; the authors replace it with “the best-performing baseline” for computing mean accuracy (Table 17). This maintains comparability but should be explicitly flagged as such in the main text, since it slightly understates the gap between TSPulse and VQShape in the mean.  
   - For similarity search, only the smallest variants of Moment and Chronos are considered (Figure 7). This is logical for latency and embedding-size parity, but I would like to see at least a brief mention of how larger variants trade off accuracy vs cost relative to TSPulse.

---

## Potentially Missing Related Work

1. **Li, Y., Chen, Z., Zha, D. (2022). “Towards Learning Disentangled Representations for Time Series.”**  
   This work directly studies disentangled representation learning for time series and proposes concrete disentanglement objectives/architectures. It is very close conceptually to TSPulse’s goal of separating distinct generative factors (temporal, spectral, semantic); it should be discussed in Appendix A.6 (unsupervised representation learning for TS diagnostics) and contrasted with TSPulse’s multi-head reconstruction strategy. It would also be useful to compare the explicit disentanglement criteria in Li et al. with the emergent disentanglement measured by the distortion metrics in Section 6.

(Other related works like TimeDRL, TF-C, BTSF are already cited in Appendix A.6.)

---

## Questions

1. **Can you formalize the hybrid masking distribution and its hyperparameters?**  
   For reproducibility and to better understand what the model is learning, please specify:
   - The probability of selecting full-patch vs point-level masking for each patch or sample.  
   - The distribution over masking ratios (per sequence and per patch).  
   - Whether block and point masks are sampled independently per channel or shared.  
   A simple pseudocode or equations would make this much clearer.

2. **Have you tried an explicit decorrelation or mutual information penalty between embedding segments?**  
   Since the backbone can freely mix information between time/FFT/register tokens, it would be informative to know whether you experimented with any loss term encouraging independence, such as minimizing off-diagonal covariance between \(\mathbf{Time}_E\), \(\mathbf{FFT}_E\), and \(\mathbf{Reg}_E\), or using an InfoNCE-style cross-view loss. If so, how did it affect both distortion metrics (Tables 2, 4–7) and downstream tasks?

3. **How sensitive are the results to the number of register tokens \(R\) and their dimension \(D\)?**  
   The semantic embedding dimension is much smaller (e.g., 256 vs 1536) yet drives similarity search and contributes to classification. Have you run ablations varying \(R\) (e.g., 1 vs 4 vs 8 tokens) and/or halving/doubling the register dimensionality? It would be useful to see whether there is a sweet spot where semantic robustness saturates.

4. **Could you compare against at least one explicitly time-frequency or disentangled TS method as a baseline?**  
   For example, implementing TF-C (Zhang et al., 2022) or BTSF (Yang & Hong, 2022) on a subset of UEA datasets or your similarity-search benchmark, with similar parameter counts, would help contextualize TSPulse among more directly related architectures.

5. **For AD, how would performance change if you restricted yourself to a single head chosen a priori?**  
   To separate the benefit of multi-head pre-training from the benefit of per-dataset head selection, it would be helpful to report:
   - VUS-PR for a fixed head (e.g., Time head only) chosen globally, without tuning.  
   - VUS-PR for a linear ensemble of heads trained on tuning data, versus the current best-head selection policy.  
   This would clarify the extent to which TSPulse’s pre-training alone is responsible for gains vs the tuning strategy.

6. **Any evidence that multivariate pre-training would (or would not) help?**  
   Given the univariate pre-training design, it would be insightful to see either a small experiment on synthetic multivariate data or a discussion: do you expect cross-channel temporal patterns to require a very different masking / FFT design? Or can the current framework be extended with minimal cost?

Clear answers and possibly small additional ablations during rebuttal could further strengthen the paper and my confidence in the central claims.

---

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A. The work uses standard public time-series datasets (Monash, LibCity, UEA, UCR, TSB-AD) with no sensitive personal information discussed in the main paper, and proposes generic representation-learning methods.

---

## Soundness Rating

3: good.  
The methodology is technically sound overall, loss functions and workflows are clear, and experiments are extensive and largely well-designed. Some components (disentanglement claim, hybrid masking specification, AD head selection) would benefit from more formalization and additional controlled baselines, but there are no obvious fatal flaws.

## Presentation Rating

3: good.  
The paper is dense but mostly clear, with useful figures (**Figures 1–3** for intuition, **Figure 2** for architecture, **Figures 4–7** for key results). A few notational glitches and typos in the FFT appendix and masking description should be fixed, and related-work positioning versus disentanglement methods could be tightened.

## Contribution Rating

3: good.  
The combination of: (i) a multi-space, multi-abstraction pre-training framework; (ii) strong empirical results across four distinct TS diagnostic tasks at tiny scale; (iii) detailed sensitivity analyses of emergent disentanglement; and (iv) a practically important hybrid masking strategy, is a meaningful contribution to time-series representation learning and lightweight foundation models, even though none of the individual ideas alone is radically new.

## Overall Rating

8: Accept, good paper (poster).  
Despite some missing baselines and the informal nature of the “disentanglement” claim, the work is technically solid, addresses an important problem, and shows strong, consistent empirical improvements with a compellingly efficient architecture. With minor clarifications and a bit more context against closely related disentanglement work, it would be a valuable addition to ICLR.

## Reviewer Confidence

4: confident.  
I am familiar with time-series pre-training, disentangled representations, and the main baselines (Moment, UniTS, Chronos, TimesFM, etc.), and I have read the math, ablations, and appendices carefully. There is some room for authors to clarify implementation details and add baselines, but it is unlikely that I have fundamentally misunderstood the core method.
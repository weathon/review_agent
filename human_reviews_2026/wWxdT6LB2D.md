# Test-Time Matching: Unlocking Compositional Reasoning in Multimodal Models

- Decision: Accept (Poster)
- Scores: 4, 2, 6, 6

## Abstract
Frontier AI models have achieved remarkable progress, yet recent studies suggest they struggle with compositional reasoning, often performing at or below random chance on established benchmarks. We revisit this problem and show that widely used evaluation metrics systematically underestimate model capability. To correct this artifact, we introduce a group matching score that more faithfully evaluates model capability. Moreover, correctness under the new metric can be translated into correctness under existing metrics via a simple overfitting step. This adjustment enables SigLIP-B16 to surpass all previous results and GPT-4.1 to yield the first result surpassing estimated human performance on Winoground. Building on this insight, we propose Test-Time Matching (TTM), an iterative, self-improving algorithm that further bootstraps model performance without any external supervision. TTM delivers additional, non-trivial improvements: for example, TTM enables SigLIP-B16 to surpass GPT-4.1 on MMVP-VLM, establishing a new state of the art. Importantly, TTM remains broadly effective even on benchmarks without metric-induced effects or group structures, achieving relative gains up to 85.7% on challenging datasets such as WhatsUp. Across 16 dataset variants spanning diverse setups, our experiments demonstrate that TTM consistently improves model performance and advances the frontier of compositional reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Test-Time Matching (TTM), a method that enhances compositional reasoning in multimodal models by redefining evaluation metrics (GroupMatch) and applying unsupervised test-time adaptation. While the idea of leveraging matching structures is interesting, the improvements mainly come from metric relaxation rather than genuine reasoning gains. The proposed TTM is conceptually similar to prior test-time adaptation methods, incurs high computational cost, and shows limited practicality beyond small benchmarks.

### Strengths
1. The paper addresses an important problem in multimodal compositional reasoning and highlights biases in existing evaluation metrics.

2. The proposed GroupMatch metric and Test-Time Matching (TTM) framework offer an interesting perspective on improving model performance without additional supervision.

### Weaknesses
1. The paper mainly improves evaluation metrics (GroupMatch) rather than actual model reasoning ability. The performance gains may stem from relaxed scoring rather than genuine compositional understanding.

2. The Test-Time Matching (TTM) algorithm is conceptually similar to existing test-time adaptation methods (e.g., TENT, AdaContrast). The contribution appears incremental rather than fundamentally new.

3. TTM requires iterative matching and fine-tuning during inference, which is computationally expensive and unsuitable for real-world deployment or online systems.

4. The reported improvements are mostly on small datasets (e.g., Winoground). It is unclear whether the method generalizes to larger or more realistic multimodal tasks.

### Questions
1. Since much of the reported improvement comes from the new GroupMatch metric, does the proposed method still yield significant gains under conventional metrics? The dependence on the new metric raises concerns about fair comparison.

2. TTM requires iterative matching and fine-tuning during inference. How does this impact efficiency compared to standard zero-shot evaluation? A runtime analysis would be helpful.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a more relaxed and losser evaluation metric, GroupMatch, for multimodal compositional reasoning and an iterative test-time finetuning method (TTM). Instead of modifying model architectures, the authors show that changing the metric from GroupScore to GroupMatch can boost the performance of existing models. Also, the author claim that TTM further boosts performance through the pseudo-label–based training of model parameters at test time.

**For me, I highly believe this work is lower than the bar of ICLR.** The reasons are one-by-one listed in weaknesses.

### Strengths
1. The authors propose a simple but effective TTM method with impressive empirical gains.
2. The authors claim that the results achieved by the method surpass human performance on Winoground.

### Weaknesses
1. **Lack of clear rationale and rigorousness.** The motivation behind using a looser metric is not fully clear. The authors should give more explaniation for the connection between the relaxed metric GroupMatch and the real reasoning ability of models. Does GroupMatch truly measure reasoning ability, or just use a looser rule to make results look good? I still believe that the standrad strict metric is more rigorous than the GroupMatch desinged in this paer.

2. **Risk of inflation.** Because GroupMatch is easier to satisfy, models may appear much stronger without any actual improvement in reasoning. This could distort comparisons across methods or benchmarks. I indeed do not believe using a looser evaluation metric is absolutely better than the use of a strict metric. 

3. **Lack of theoretical grounding.** The paper claims that GroupMatch “reveals hidden capabilities,” but provides limited formal justification or analysis on why this metric better reflects reasoning.

4. **Lack of theoretical grounding.** The paper could provide more explicit analysis of the runtime/efficiency implications of TTM, especially for large-scale deployments.

5. **Limited robustness analysis.** There is no clear evaluation of how sensitive GroupMatch is to noise, adversarial perturbations, or different group sizes. If the metric is fragile, its reliability as an evaluation tool is questionable.

6. **Unconvincing results.** Since TTM explicitly uses the test data for adaptation, and GroupMatch relaxes the matching requirement. So, the authors should report the performance improvment brought by TTM using the original stric GroupScore, which makes the restults more solid.

7. **Poor practical applicability.** TTM need to finetue the model' parameters on test data, which has worse practical applicability in real-world scenarios, e.g., test data may be scarce and models may be very large (such as GPT-4,5, and Gemini).

### Questions
1. Does the method generalize beyond image-text tasks to other modalities (e.g., audio-text, video-text)?

2. Does GroupMatch truly measure reasoning ability, or just alignment under a softer rule?

3. Can models cheat the metric GroupMatch without improving their underlying reasoning as it is more looser and relaxed?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper revisits the problem of compositional reasoning in multimodal models, which are often reported to perform near random chance on benchmarks such as Winoground. The authors first identify that existing evaluation metrics (notably GroupScore) systematically underestimate model capability. They propose a new metric, the Group Matching Score (GroupMatch), which evaluates the best overall image–caption matching within each group. This new metric reveals hidden competence in both contrastive VLMs (e.g., CLIP, SigLIP) and MLLMs (e.g., GPT-4.1), enabling GPT-4.1 to surpass estimated human performance on Winoground. Building on this, they introduce Test-Time Matching (TTM), an iterative, supervision-free self-training algorithm that converts confident model-induced matchings into pseudo-labels, progressively bootstrapping model performance during inference. TTM yields further non-trivial improvements—e.g., SigLIP-B16 surpasses GPT-4.1 on MMVP-VLM—achieving new state-of-the-art results across 16 benchmark variants, including datasets with and without group structures. In summary, the paper’s key contributions are: (1) redefining evaluation through GroupMatch to uncover hidden compositional ability; (2) proposing TTM for test-time self-bootstrapping without external supervision; and (3) demonstrating broad applicability and new SOTAs across both grouped and non-grouped multimodal reasoning benchmarks

### Strengths
The paper reframe multimodal compositional reasoning evaluation. By identifying that existing metrics systematically underestimate model ability, it introduces the GroupMatch score as a new evaluation paradigm that better reflects true multimodal alignment. Moreover, the proposed Test-Time Matching (TTM) algorithm creatively extends test-time training and self-training principles to multimodal evaluation without external supervision. This combination of metric reformulation and unsupervised adaptation constitutes a distinctly original contribution in the field. The derivation of probabilistic properties of existing and proposed metrics offers analytical clarity on why previous benchmarks yield misleadingly low scores. The empirical evaluations are comprehensive—covering 16 datasets and multiple model families (CLIP, SigLIP, GPT-4.1)—and are backed by detailed ablations, sensitivity analyses, and comparisons. The results are consistent and reproducible, indicating high experimental quality.

### Weaknesses
1. The SimpleMatch and TTM procedures explicitly overfit to test data using pseudo-labels derived from the model’s own predictions. Although the paper positions this as “test-time adaptation,” it effectively tunes on test distributions, which may blur the boundary between evaluation and training. The authors should clarify the implications of this for benchmark comparability and fairness, and discuss possible countermeasures such as validation-based stopping criteria or out-of-distribution robustness checks.

2. Although the paper covers 16 dataset variants, most are derived from similar group-based compositional reasoning benchmarks (Winoground, MMVP-VLM, Colorswap, SugarCrepe, Whatsup). It remains unclear whether TTM generalizes to broader multimodal tasks such as retrieval, captioning, or visual reasoning benchmarks like ARO or GQA. Testing beyond compositional group setups would strengthen claims about general applicability.

3. While the paper presents sensitivity studies on threshold decay, it does not deeply analyze the types of errors propagated during pseudo-labeling or how inaccurate matchings affect subsequent iterations. An empirical study on pseudo-label noise, or visualization of incorrect matchings and their impact, could make the bootstrapping dynamics more interpretable and trustworthy.

4. The iterative self-training procedure (ten iterations, multiple epochs each) likely increases inference and adaptation time significantly. The paper would benefit from reporting computational overheads, memory costs, and convergence behavior, as well as exploring lightweight or one-shot alternatives to make TTM practical for real-world applications.

5. The paper would benefit from a more formal characterization of when and why TTM converges, or how pseudo-label noise affects performance over iterations. Connections to existing theories in semi-supervised or test-time training (e.g., Sun et al., 2020; Gandelsman et al., 2022; Akyürek et al., 2024) could strengthen the methodological depth.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper argues that multimodal models' poor performance on compositional reasoning benchmarks is partly due to flawed evaluation metrics that underestimate their true capabilities. The authors first introduce a new evaluation metric, the "group matching score" (GroupMatch), which evaluates the best overall matching within a group and uncovers substantial "hidden competence" in existing models. Building on this, they propose Test-Time Matching (TTM), an iterative, supervision-free self-training algorithm that bootstraps model performance by finetuning on high-confidence pseudo-labels generated at test time. This approach achieves new state-of-the-art results, notably enabling GPT-4.1 to surpass estimated human performance on Winoground and SigLIP-B16 with TTM to outperform GPT-4.1 on MMVP-VLM.

### Strengths
1. One strength of the paper is its insightful critique of existing evaluation metrics, such as GroupScore, which it convincingly argues systematically underestimate model capabilities in compositional reasoning. The introduction of the "GroupMatch" score provides a simple, intuitive, and more effective alternative that reveals significant "hidden competence" in existing models.

2. The proposed Test-Time Matching (TTM) algorithm is an effective supervision-free method that iteratively bootstraps model performance. This self-training approach, which uses high-confidence pseudo-labels at test time, provides a practical way to unlock the latent abilities identified by the new metric.

3. The paper presents extensive and strong empirical results across 16 dataset variants, achieving new state-of-the-art (SOTA) performance. Notably, the SimpleMatch method allows GPT-4.1 to surpass estimated human performance on Winoground for the first time, and TTM enables a smaller model (SigLIP-B16) to outperform GPT-4.1 on MMVP-VLM.

### Weaknesses
1. The TTM algorithm relies on iterative finetuning at test time, which is computationally expensive and time-consuming. Discussion about the trade-off between compute and performance is needed. For example, compare a small model with more iterations, and a large model with fewer iterations.

2. The performance of TTM appears sensitive to the choice of new hyperparameters, particularly the threshold scheduling used for pseudo-label selection. The paper notes that the initial threshold is "dataset- and model-dependent" and requires careful setting, which may make the method difficult to apply robustly to new datasets or models without costly tuning.

3. GPT-4.1 is closed-sourced. It is necessary to conduct experiments on an open-sourced MLLM.

### Questions
Please see the weakness

### Soundness
3

### Presentation
3

### Contribution
3

# Token-level Data Selection for Safe LLM Fine-tuning

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
Fine-tuning large language models (LLMs) on custom datasets has become a standard approach for adapting these models to specific domains and applications. However, recent studies have shown that such fine-tuning can lead to significant degradation in the model's safety. Existing defense methods operate at the sample level and often suffer from an unsatisfactory trade-off between safety and utility. To address this limitation, we perform a systematic token-level diagnosis of safety degradation during fine-tuning. Based on this, we propose token-level data selection for safe LLM fine-tuning (TOSS), a novel framework that quantifies the safety risk of each token by measuring the loss difference between a safety-degraded model and a utility-oriented model. This token-level granularity enables accurate identification and removal of unsafe tokens, thereby preserving valuable task-specific information. In addition, we introduce a progressive refinement strategy, TOSS-Pro, which iteratively enhances the safety-degraded model's ability to identify unsafe tokens. Extensive experiments demonstrate that our approach robustly safeguards LLMs during fine-tuning while achieving superior downstream task performance, significantly outperforming existing sample-level defense methods. Our code is available at https://github.com/Polly-LYP/TOSS.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel framework for safe LLM fine-tuning called Token-level data Selection for Safe LLM fine-tuning (TOSS), which directly addresses the critical issue of safety degradation during domain-specific model adaptation.
The core of TOSS is a unique mechanism that quantifies the safety risk of each token within the custom dataset. This is achieved by measuring the loss difference between a safety-degraded model (trained on harmful data) and a utility-oriented model (trained on high-quality utility data).
By employing this fine-grained, token-level data selection instead of conventional coarse-grained (sample-level) filtering, the authors achieve a more efficient utilization of the training data. The resulting cleaned dataset allows for a more effective instruction-tuning process, leading to a superior trade-off that significantly enhances model safety.

### Strengths
- Novelty and Performance: The paper proposes a simple yet highly effective token-level data selection method (TOSS/TOSS-Pro). The experimental results demonstrate its superiority, achieving SOTA performance across three distinct safety benchmarks.

- Clarity and Methodology: The presentation of the work is notably clear. The authors effectively structure the paper by including preliminary experiments (diagnostic analysis) which systematically build the rationale for their main design, guiding the reader through the necessity of the fine-grained, token-level approach.

### Weaknesses
- Generalization to Out-of-Distribution (OOD) Scenarios: The paper primarily focuses on the safety issue during the post-training or continuous fine-tuning phase. Since the two reference models (safety-degraded and utility-oriented) heavily rely on in-domain data for training, the generalizability of TOSS/TOSS-Pro to Out-of-Distribution (OOD) safety scenarios is questionable. For instance, if the reference models are trained using financial safety-related data, safety concerns might still persist in OOD domains such as medical safety or cybersecurity. The paper needs to discuss the framework's robustness or limitations in transferring across different safety categories.

- Dependence on Reference Model Training Data: The proposed token-level data selection method is highly dependent on the quality and specificity of the reference model training data. However, the paper lacks detailed information regarding the composition, source, and size of the harmful and utility reference datasets ($\mathcal{D}^{h}$ and $\mathcal{D}^{u}$), which are crucial for reproducibility and understanding the framework's effectiveness.

- Potential for Response Disruption and Utility Evaluation: When masking tokens with high token-level scores (safety risk) in the response label during the safe training phase, it is possible that this leads to discontinuous or ill-formed generated responses. The paper should provide a more comprehensive evaluation of the model's general instruction-following utility after TOSS is applied, using widely accepted benchmarks like AlpacaEval and MT-Bench (in addition to the SLIMORCA utility benchmark used), to ensure that the safe fine-tuning process does not cause model degradation or "model collapse."

- Lack of Detail on AI Safety Evaluation Metrics: The paper lacks a detailed introduction and justification for the specific AI safety evaluation metrics used. While the paper mentions safety benchmarks like HEx-PHI and HH, from my understanding, it seems that the description in Appendix C uses evaluation paradigms similar to AlpacaEval.(i.e., LLM-as-a-Judge). This approach is known to have limitations and potential biases in safety evaluation.

### Questions
1. Considering that the paper's training objective utilizes the **next token prediction pattern** (causal language modeling), a core concern arises regarding the handling of masked tokens. Specifically, if token $y_i$ is identified as a high-risk (harmful) token and is subsequently masked out (i.e., $m_{i}=0$) so that it does not contribute to the loss, the context for the prediction of the next token $y_{i+1}$ still relies on $y_i$ being present in the input sequence (as show in **Line 229 and 286**).This means that the model $f_{\theta}$ is still exposed to the harmful token $y_i$ during the prediction of $y_{i+1}$, which could potentially introduce uncertainty or residual harmful signal into the learned weights, thereby compromising the intended safety enhancement.
Does the paper's current handling method (simply setting the loss weight $m_{i}=0$) adequately mitigate the risk of exposing the model to harmful context for subsequent token predictions? 

2. Are there better optimization strategies that could be explored to fully sever the harmful influence of $y_i$ on the training of $y_{i+1}$? For example, techniques like replacing the high-risk token $y_i$ with a neutral token (e.g., a special [SAFE] or [MASK] token) or completely removing the span of harmful tokens from the input sequence.

3. Distinction from Existing Token-Level Work: We have recently seen several works focusing on token-level filtering or selection, many of which utilize the Shannon entropy $\log P(x)$ (or maximum likelihood loss) derived from the sequence modeling objective. Could the authors explicitly articulate the key differences between the proposed TOSS/TOSS-Pro framework (which uses a loss-difference metric between two specialized reference models) and these contemporary token-level selection methods? Specifically, why is the dual-reference model approach necessary and superior to simpler methods based solely on a single model's loss or entropy (e.g. safety-degraded reference model)?

4. Potential Negative Impact of Token Masking: The paper employs a token-level masking or filtering technique on the training data. Does this token-level masking or filtering of the response labels during the fine-tuning process introduce undesirable side effects on the learned model distribution? Since the model is trained with discontinuous targets (due to masked tokens), is there a risk of degrading the model's fluency or coherence in generation, especially for non-harmful, utility-oriented responses? The paper should provide a dedicated discussion or empirical evidence on this potential trade-off.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper argues that the root cause of safety degradation during LLM fine-tuning lies at the token level, a finer granularity than the conventional sample level. To address this, the authors propose TOSS and its progressive variant, TOSS-Pro, which aim to identify and mask high-risk tokens to preserve the model's safety alignment.

### Strengths
1. To my knowledge, this is the first work to propose a token-level data selection framework specifically for safe LLM fine-tuning. This fine-grained approach convincingly addresses a key limitation of coarse-grained, sample-level methods, which often discard valuable, task-relevant information.

2. The motivation is well-supported by a clear diagnostic analysis. The KL divergence analysis across token positions provides compelling empirical evidence for the paper's central hypothesis that safety-degrading signals are localized and not uniformly distributed, thus justifying the need for a token-level solution

### Weaknesses
1. The method's effectiveness is heavily dependent on the quality of the initial reference models, which introduces significant computational/data overhead and a potential "bootstrapping paradox". The framework requires two pre-trained reference models (a safety-degraded model and a utility-oriented model), which demands substantial extra resources and datasets. More critically, there is a circular dependency: to clean a custom dataset, one needs reference models trained on datasets that are assumed to be well-profiled. While the authors propose TOSS-Pro to iteratively refine the safety-degraded model, this does not fully resolve the "garbage-in, garbage-out" problem. If the initial reference model is of poor quality, the iterative process may fail to effectively identify harmful tokens. The paper lacks a sensitivity analysis on how the quality of the initial reference datasets impacts the final performance.

2. The proposed scoring function risks over-filtering useful tokens in specialized domains where the line between "harmful" and "technical" is blurry. For example, in tasks like chemical synthesis or cybersecurity, tokens related to potentially dangerous but scientifically valid concepts could be incorrectly assigned high-risk scores and masked. The paper's evaluation of utility is confined to a general-purpose instruction-following dataset (SLIMORCA). To demonstrate robustness, the method must be evaluated on more complex and professional domains such as code generation (e.g., LiveCodeBench), mathematical reasoning (e.g., AIME), and scientific QA. Furthermore, the paper does not report the impact on the model's general capabilities, which should be measured on standard benchmarks like MMLU. 

3. The paper's comparison is limited to other data-centric, training-time methods and overlooks other important defense paradigms. To justify the significant computational cost of TOSS, it is crucial to compare it against less expensive alternatives, such as test-time intervention methods (e.g., Circuit Breakers) or model-editing techniques (e.g., Safety Arithmetic, SafetyLock). Without this comparison, it is unclear whether the performance gains offered by TOSS are substantial enough to warrant its high resource demands.

### Questions
See in the Weaknesses.

### Soundness
1

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
This paper proposes TOSS, a token-level data selection framework that addresses safety degradation during LLM fine-tuning by identifying and removing only unsafe tokens rather than entire samples. Using two reference models (safety-degraded and utility-oriented), they score each token based on loss differences to partially discard harmful tokens while preserving task-relevant information.

### Strengths
The paper shifts the focus from sample-level to token-level selection and provides compelling evidence. The KL-divergence analysis (Figure 2) shows that safety-degrading signals are concentrated in specific tokens rather than entire samples.

### Weaknesses
- Evaluation over-relies on win rate (relative preference) and fails to capture absolute safety; e.g., a model can “win” 88% yet still emit harmful content.
    - Other safety metrics are missing: Attack Success Rate (ASR), harmful-content generation rate, and false-refusal (over-safety) rate are neither reported nor analyzed.

### Questions
- Q1: Figure 4 (left) indicates that global ranking outperforms local ranking; could you provide an intuition/analogy for why, and any supporting statistics (e.g., harmful-token distributions or discard fractions)?
- Q2: Have you conducted a win-rate comparison between SEAL and TOSS?
- Q3: Could you explain Figure 5? If the ratio is 1, are all tokens discarded?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Token-level data Selection for Safe LLM fine-tuning (TOSS), a novel framework designed to mitigate safety degradation in Large Language Models (LLMs) during fine-tuning on custom datasets. The core innovation is moving from coarse-grained sample-level data selection, which suffers from an unsatisfactory safety-utility trade-off, to a fine-grained token-level approach. TOSS quantifies the safety risk of each token by calculating a loss-difference metric between a safety-degraded model (an expert in unsafe patterns) and a utility-oriented model (an expert in task utility). Tokens with high risk scores are selectively masked (discarded) from the custom dataset before fine-tuning, thereby preserving valuable task-specific information. The paper further introduces TOSS-Pro, a progressive refinement strategy that iteratively updates the safety-degraded model using the most informative harmful samples, which leads to more accurate identification of unsafe tokens and enhanced safeguarding performance. Extensive experiments validate that TOSS and TOSS-Pro achieve a superior trade-off between safety and utility compared to existing sample-level defense methods like SEAL.

### Strengths
*Originality: The paper makes a highly original contribution by proposing the first token-level data selection framework specifically for safeguarding LLM fine-tuning. The fundamental insight that "the unit of safety degradation is not the sample, but the token" is strongly motivated by a systematic token-level diagnosis, distinguishing this work significantly from prior sample-level methods (e.g., SEAL).
*Quality & Clarity: The proposed methodology, TOSS, is technically sound and clearly articulated. The loss-difference metric is an elegant, intuitive mechanism for quantifying the joint safety-risk and utility-alignment of a token. The introduction of the S(y_{i,j}^{cus})decomposition (Eq. 2) provides excellent clarity on the complementary roles of the two reference models. Furthermore, the empirical evidence, including the main results (Table 1) and comprehensive ablation studies (Figure 4, Tables 3 & 4), strongly supports the claims and validates the effectiveness of the token-level paradigm shift.
*Significance: The work addresses a critical vulnerability in the burgeoning practice of LLM fine-tuning and customization. The superior performance over state-of-the-art baselines, particularly the up to 30% higher win rate on safety benchmarks and 11% on utility benchmarks compared to SEAL on Llama-3-8B-Instruct (Table 1), demonstrates a significant advance in achieving a favorable safety-utility trade-off. The strong transferability result (Table 2) suggests a practical and scalable deployment advantage.

### Weaknesses
*Computational Cost of Reference Model Training: The core TOSS framework relies on training two full reference models ($f_{\theta^h}$  and $f_{\theta^u}$  ) via SFT, which can be computationally expensive, particularly for much larger LLMs (e.g., Llama-70B). While LoRA fine-tuning is used (Appendix B), the overall process—training two models and then performing token assessment on the entire custom dataset—is significantly more intensive than sample-level filtering that often relies on a single, lighter ranking model (like SEAL's bi-level optimization ranker). The authors should discuss the time complexity and computational overhead, perhaps in relation to the gain, or suggest strategies to reduce this cost (e.g., using smaller reference models or knowledge distillation).
*Hyperparameter Sensitivity of d (Discarding Ratio): While Figure 5 shows robustness across different discarding ratios d, the main results (Table 1) fix d=0.1. The results for d=0.1 show a substantial utility improvement for TOSS (68.37%) over SEAL (57.41%) on Llama-3-8B-Instruct. However, the curves in Figure 5 show that the choice of d is crucial for maximizing utility, which first increases and then decreases. This highlights that d is a critical, potentially fragile, hyperparameter for optimization. A more robust method for automatically or adaptively setting the optimal discarding ratio d (or per-sample local thresholds) would greatly strengthen the framework and reduce deployment complexity.
*Clarity on Progressive Refinement Sample Selection: In TOSS-Pro, the authors select the k most informative harmful samples D_t^s  by finding samples that contain the top-ranked tokens. This is a sample-level selection mechanism built on a token-level score. The paper should elaborate on how a sample is defined to "contain" a top-ranked token in the set of selected samples D_t^s . Does this mean any token in the sample is in the top set, or is there a minimum number/proportion of top tokens required? Clarifying this selection logic is important for reproducibility and understanding how the "higher-quality supervision" is precisely curated

### Questions
1. The core diagnosis (Figure 2) shows the largest △KL shift in the initial few tokens, but also spikes later (e.g., token 7). Your naive baseline of masking the first five tokens performed poorly on utility. Could you provide a qualitative analysis or visualized examples (similar to Appendix F, but for the naive mask) illustrating which essential utility tokens are sacrificed by the fixed masking compared to the selective masking of TOSS? This would powerfully support the central thesis that selective masking preserves utility within the critical initial positions.
2. The token score formulation is S(y)=L^u (y)−L^h (y). This gives equal weight to utility-related risk (high L^u) and safety-related alignment (low L^h). Have the authors experimented with a weighted loss difference metric? For instance, S′(y)=αL^u(y)−βL^h(y), where α and β could be hyperparameters, or perhaps learned through a bi-level optimization akin to SEAL? This could allow for a more deliberate control over the safety-utility trade-off curve.
3. For TOSS-Pro, the safety-degraded model is progressively updated. Why is the utility-oriented model $f_{\theta^u}$  kept fixed throughout the T iterations of TOSS-Pro (Algorithm 1, line 4, line 8)? Since the goal is the safety-utility trade-off, continually refining the utility model using the non-selected (safe/high-utility) tokens could potentially provide an even stronger utility signal, perhaps further enhancing the trade-off. What is the rationale for fixing $f_{\theta^u}$ ?

### Soundness
4

### Presentation
4

### Contribution
4

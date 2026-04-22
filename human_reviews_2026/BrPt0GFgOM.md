# Dual-objective Language Models: Training Efficiency Without Overfitting

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
This paper combines autoregressive and masked-diffusion training objectives without any architectural modifications, resulting in flexible language models that outperform single-objective models. Autoregressive modeling has been a popular approach, partly because of its training efficiency; however, that comes at the cost of sensitivity to overfitting. On the other hand, masked-diffusion models are less efficient to train while being more resilient to overfitting. In this work, we demonstrate that dual-objective training achieves the best of both worlds. To derive the optimal balance between both objectives, we train and evaluate 50 language models under varying levels of data repetition. We show that it is optimal to combine both objectives under all evaluated settings and that the optimal balance is similar whether targeting autoregressive or masked-diffusion downstream performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Dual Language Models (Dual-LMs), which jointly train a single transformer with both autoregressive (AR) and masked-diffusion (MD) objectives. The key motivation is to balance sample efficiency (of AR training) and overfitting resilience (of MD training) without changing the model architecture. The authors conduct a large-scale empirical study on 50 language models to support the method's effectiveness.

### Strengths
1. Clear motivation and timely problem.
The work addresses a relevant issue as large-scale language model training increasingly encounters limited high-quality data and potential overfitting under repeated exposure.
2. Practical recommendations
The derived empirical guidelines (e.g., small MD component under regular regimes, stronger MD ratio under high repetition) are actionable for practitioners designing large-scale LLM training recipes.

### Weaknesses
1. Limited contribution in standard settings.
When trained with only 1× data repetition, Dual-LMs provide no significant improvement over pure autoregressive baselines. This weakens the overall claim of universal benefit; the method is primarily helpful under extreme data-reuse scenarios
2. Practical rarity of extreme repetition.
The key advantage appears only under 128× repetition, which is rarely used in modern large-scale LLM training. In realistic compute-optimal regimes (1–8× repetition), the dual-objective gains are negligible.
3. Learning rate confound.
For very high repetition, a smaller learning rate is typically required to maintain stability. Since the experiments maintain a fixed LR schedule, it is unclear whether the observed benefits of Dual-LMs come from the objective mixing itself or from suboptimal tuning in baselines.
4. Method scope and positioning.
Overall, the approach seems more suitable as a regularization mechanism for repeated training rather than a fundamentally new paradigm for LLM optimization. The contribution is empirical rather than conceptual, and its practical importance outside data-scarce settings is limited.

### Questions
1. The paper’s results suggest that the dual-objective helps mainly under extreme data repetition. What prevents it from providing gains in standard (1–8×) training regimes? Is it fundamentally tied to overfitting mitigation rather than general sample efficiency improvement?

### Soundness
2

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
3

### Summary
This paper studies the trade-off between sample efficiency and overfitting resilience in language modeling by combining autoregressive and masked-diffusion objectives in a single transformer architecture, without architectural modifications. The authors conduct an extensive empirical study, training 50 models across varying data repetition regimes, to uncover the optimal balance between the two objectives. They show that the dual-objective approach consistently outperforms single-objective baselines in both autoregressive and masked-diffusion downstream tasks, providing practical guidance on loss balancing and implications for prefix language modeling.

### Strengths
- The authors train and evaluate 50 language model configurations under a wide range of data repetition regimes, allowing for robust conclusions on objective balancing.
- The dual-objective method is introduced without architectural changes, maximizing practical usability.
- The paper distills its complex results into two practical, easy-to-understand guidelines for training models in regular and data-constrained regimes.

### Weaknesses
- The core idea of combining AR and masked objectives in a single decoder-only model is not entirely new and builds heavily on previous work, particularly GPT-BERT, which the authors acknowledge. The primary contribution is more of an extensive empirical investigation rather than a fundamentally new training paradigm.
- While Section 6 discusses prior methods such as GPT-BERT and AntLM, the paper does not include empirical comparisons with these or other recent approaches (e.g. Block Diffusion [1], AR-Diffusion [2]) that address the same AR–diffusion trade-off. Including recent works as baselines under the same training setup would provide a clearer and fairer assessment of the proposed method’s advantages.
- The paper assumes that the findings generalize to larger models, but this remains speculative, as all experiments are limited to the 470M scale. Given the unpredictability of scaling behavior, it is unclear whether the observed trends would hold for models beyond 1B parameters.
- While Equations (2) and (4) define the individual objectives, the paper offers little theoretical explanation of how their joint optimization works. The rationale for specific weighting choices is empirical, and potential gradient interference or complementarity between the two losses is not analyzed.

[1] Arriola, Marianne et al. “Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models.” 2025
[2] Wu, Tong et al. “AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation.” 2023

### Questions
- The paper builds on established ideas such as GPT-BERT and CM3, with the main contribution appearing to be the systematic analysis of data repetition effects. Could the authors clarify the specific technical or algorithmic innovations introduced by the proposed dual-objective framework beyond these prior methods?
- Would the authors consider adding or discussing comparisons with related works (e.g. Block Diffusion and AR-Diffusion), which also address the AR–diffusion trade-off?
- Can the authors provide evidence—empirical or analytical—that the optimal AR–MD ratios and the 16-repetition guideline (Remark 2) remain valid for larger models (≥ 1B parameters)?
- Can the authors provide a more rigorous theoretical basis or empirical ablation for their objective weighting strategy?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This work studies Dual LM, which combines the auto aggressive next token prediction and masked language modeling into a single model. To be specific, authors are investigating the potential to train one single LLM with two different training objectives. And the finding is we can achieve a better trade off between compute efficiency and data efficiency by adjusting the ratio of two training objectives.

### Strengths
1) Very clear presentation. The lessons and insights are very presented so that readers can understand the problem well.
2) Studying the data efficiency is a very important problem, for both large and small models, especially considering we are overtraining models for better quality and cheaper serving cost.
3) The sweep on training objective ratio is informative. This can help to design better training schedule targeting on a specific data/compute efficiency.

### Weaknesses
1) The definition of sample-efficiency is confusing. I think it would be better to rephrase it as "compute-efficiency". The sample efficiency makes people confused whether the sample here means "unique token efficiency". 
2) Minor: I understand the research is usually for long term purpose, but to be very honest, the conclusion we achieved in this work is we only need dual LM for very aggressive repeats. This is anyway informative but not something we have to use now.

### Questions
NA

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper aims to introduce a unified training objective for language modeling that jointly optimizes autoregressive (AR) and masked-diffusion losses within a single transformer, while requiring no architectural modification. 
By co-training a single transformer model on both losses, DLM leverages AR’s rapid convergence while regularizing with the diffusion objective to prevent overfitting. Extensive experiments across diverse benchmarks demonstrate that this hybrid objective consistently enhances performance over single-objective baselines.

### Strengths
This paper is well written and easy to follow. The narrative is coherent, and the topic is both timely and important for the current stage of large language model development.

The proposed DLM objective is conceptually simple and practically appealing. It requires no architectural changes and includes GPU-efficient adaptations, making the method easy to implement and broadly applicable. This design choice enhances the practical relevance of the work.

The empirical evaluation appears systematic and comprehensive. The experiments are carefully structured, covering a wide range of data regimes and objective ratios, which strengthens the credibility of the results and the generality of the conclusions.

Although the work is primarily empirical, it offers new insights that meaningfully advance understanding in the community.

### Weaknesses
### The link to diffusion language model is a bit weak 

It is unclear to me whether the “diffusion mode” in the proposed objective truly corresponds to a standard masked diffusion language model, as to my knowledge, no published work implements diffusion language model using a decoder. 
Conceptually, the proposed objective appears closer to introducing stochastic regularization into standard autoregressive training—adding noise to the input sequence to improve generalization and reduce overfitting. 
Moreover, it remains ambiguous whether the resulting model supports parallel decoding, a defining property of diffusion-based LMs. 
While the paper claims that the model “can generalize to prefix LM,” it does not explicitly demonstrate diffusion-style generation. A clearer explanation of how the diffusion time-conditioning interacts with the next-token prediction objective would strengthen the technical grounding.


### Lack of analytical justificaiton. 

The paper’s conclusions are derived almost entirely from empirical evidence. While the experiments are thorough and convincing, additional theoretical or analytical insights would improve the work’s depth. For example, identifying sufficient conditions (some toy cases) under which the dual-objective training is guaranteed to outperform either individual objective would help explain why the method works beyond observation. 


### Incomplete discussion of training dynamics.
The paper focuses primarily on zero-shot, multiple-choice benchmarks but provides limited discussion of training behavior. Analyses of the loss curves (for both AR and diffusion modes), convergence patterns, and stability would give a more complete understanding of how the dual objective influences optimization and generative quality.

### Questions
* Could you elaborate more on how closely does the proposed “diffusion mode” correspond to a standard masked-diffusion language model? For instance, how is the diffusion time variable integrated into the next-token prediction objective, is it sampled per token or globally for the sequence, and how does this choice affect model behavior? How does the new objective support parallel decoding or multi-token prediction? 

* How stable is the training process under the dual objective? I am wondering what would the loss curves look like and whether there are any significant tradeoffs between the two modes during training?

* How does this work relate to a recent work [1] that proposed a modified decoder architecture to do any-order next-token prediction? 

[1] Any-Order GPT as Masked Diffusion Model: Decoupling Formulation and Architecture, arxiv preprint 2506.19935

### Soundness
2

### Presentation
3

### Contribution
3

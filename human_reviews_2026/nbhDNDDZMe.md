# Learning to Refine: Self-Refinement of Parallel Reasoning in LLMs

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 6, 2

## Abstract
To further enhance the ability of Large Language Models (LLMs) to solve complex, multi-step reasoning problems, test-time scaling (TTS) methods have gained widespread attention. Existing approaches such as Best-of-N and majority voting are limited as their performance depends on the quality of candidate responses, making them unable to produce a correct solution when all candidates are incorrect. Introducing an additional model to select the best response also incurs significant deployment costs. To this end, we introduce Generative Self-Refinement (GSR), a novel parallel test-time scaling framework where a single and unified model first generates a set of candidate responses in parallel and then performs self-refinement to synthesize a new superior solution based on a prompt consisting of the problem and these candidates. However, LLMs struggle to perform refinement effectively when prompted directly. Therefore, we design a hybrid training pipeline by jointly optimizing for two complementary objectives, solving problems directly and refining candidate responses. Experimental results demonstrate that our method achieves state-of-the-art performance across five mathematical benchmarks. We further show that this learned self-refinement skill is a model-agnostic enhancement, robust across different model scales and generalizing to out-of-distribution reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduce GSR, a novel approach that trains a model to aggregate multiple response generated in parallel by training with a hybrid training pipeline with a teacher model and a student model. The resulted model, GSR-7B, is evaluated on 5 different challenging math benchmarks. A detailed discussion is provided to show the robustness of the approach.

### Strengths
- The paper is overall easy-to-follow. The approach, while relatively simple, is very intuitive and reasonable.
- The experiments are comprehensive over multiple challenging math datasets. The baselines include multiple different reward models, and the detailed discussion shows the generalization capability of the proposed method.
- Compute budget are considered by reporting maj@5.

### Weaknesses
- Given GSR is a training-based algorithm, it is unclear whether directly training the underlying model on the same dataset with the same compute budget can lead to the same benefit.
- I find the idea of self-aggregation is relevant to a few recent work [1, 2, 3, 4]. While [4] is relatively new, I think [1, 2, 3] should be discussed and included as additional baselines.
- While in Sec. 4.5, additional results on different models are provided, those results are still after training. It is unclear whether the trained model, e.g., GSR-7B, can be directly combined with Qwen2.5-14B-Instruct.
- The models used are relatively old. I would appreciate it if the authors can include some more recent models released in this year, e.g., Deepseek-R1 series, or Qwen3 series, or GPT-oss series which include a more solid RL pipeilne during training to show the effectiveness of the approach on SOTA reasoning models.

Minor typo: In table 9, it probably should be MATH500, not MAHT500.

[1]. Yang X, An Y, Liu H, et al. Multiverse: Your Language Models Secretly Decide How to Parallelize and Merge Generation[J]. arXiv preprint arXiv:2506.09991, 2025.

[2]. Li Z, Feng X, Cai Y, et al. Llms can generate a better answer by aggregating their own responses[J]. arXiv preprint arXiv:2503.04104, 2025.

[3]. Yang C, Wang X, Lu Y, et al. Large language models as optimizers[C]//The Twelfth International Conference on Learning Representations. 2023.

[4]. Zhao W, Aggarwal P, Saha S, et al. The majority is not always right: Rl training for solution aggregation[J]. arXiv preprint arXiv:2509.06870, 2025.

### Questions
Please refer to the weakness section for general concerns. Additionally, here are some questions that involves clarity:
- Is the selfRef@4 in Sec. 4.3 on GSR or on the original model? Can you report the results of the alternative setting?
- In table 1, the caption says additional Pass@1 from  Yang et al. (2025c) and QwenTeam (2024) is reported, but I cannot find them anywhere.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Generative Self-Refinement (GSR), a parallel test-time scaling method that enables a single LLM to generate multiple candidate solutions and then self-refine them into a superior answer. A hybrid training pipeline jointly optimizes direct-solving and refinement via teacher-student distillation on math data.

### Strengths
1. The paper is written clearly;
2. The overall training pipeline is relatively simple.

### Weaknesses
1. The paper explores Self-Refinement but lacks detailed discussion of closely related works such as SELF-REFINE[1], SCORE [2], ARIES [3], ReVISE [4] and SynPO [5]：
   * SELF-REFINE also focuses on refining a model’s own outputs—what distinguishes GSR from SELF-REFINE? 
   * SCORE investigates self-correction in mathematical reasoning—what is the conceptual difference between self-refinement and self-correction? 
   * ARIES similarly studies self-refinement, and GSR’s SFT loss function appears quite similar to that of ARIES. What are the key distinctions between GSR and ARIES? 
   * ReVISE and SynPO also explores refinement through verification or refiner-based mechanisms, yet the paper provides little discussion of these relevant studies.
2. The proposed approach essentially distills a smaller model from a stronger teacher model, which makes it highly dependent on teacher quality. The paper lacks analysis of this dependency.
3. Figure 4 shows that as the number of candidates increases, the performance of Self-Refinement becomes worse than Majority Voting. This suggests that the model’s self-refinement ability may be largely forced by distillation; once the number of candidates deviates from that seen during training, performance stagnates or even degrades, indicating poor generalization over candidate count.
4. The authors could further explore the model’s refinement behavior, for example:
   - Is Self-Refinement always beneficial, or does it sometimes turn correct answers into incorrect ones? What are the respective proportions of “correct → wrong” and “wrong → correct”?
   - Can the proposed strategy perform iterative self-refinement—refining its own answers multiple times—or is it limited to a single refinement pass over the candidates?
5. As a decoding strategy, the paper lacks fine-grained ablation studies, such as analyzing how parameters like temperature and top-p affect the performance of Majority Voting and Self-Refinement.
6. The paper also lacks direct comparisons with related methods such as SCORE [2], ARIES [3], and ReVISE [4].



[1] Self-Refine: Iterative Refinement with Self-Feedback. NeurIPS 2023.

[2] Training Language Models to Self-Correct via Reinforcement Learning. ICLR 2025.

[3] ARIES: Stimulating Self-Refinement of Large Language Models with and for Iterative Preference Optimization. ICLR 2025 Workshop.

[4] ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification. ICML 2025.

[5] Self-Boosting Large Language Models with Synthetic Preference Data ICLR 2025.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Generative Self-Refinement (GSR), a novel parallel test-time scaling framework designed to enhance Large Language Models (LLMs) in solving complex, multi-step reasoning problems. Unlike existing test-time scaling methods such as Best-of-N or majority voting, which are limited by the quality of candidate responses and cannot produce a correct solution if all candidates are flawed, GSR empowers a single, unified LLM to both generate diverse candidate responses in parallel and then perform self-refinement to synthesize a superior solution. This refinement is achieved by prompting the model with the original problem and its self-generated candidates, enabling it to diagnose flaws and selectively leverage insights, even from erroneous solutions, to construct a correct answer.

Overall, this paper shows a promising way of teaching the model to do self-refine for test-time scaling instead of designing complex pipelines, which could be a valuable insight and paradigm for the community.

### Strengths
1.  **Significant Conceptual Novelty in Self-Correction:** The core strength of GSR lies in its potential to **correct solutions even when all initial candidates are flawed**, representing a significant conceptual advance over selection-based methods like majority voting. This self-correction mechanism pushes the boundary of model capability and introduces a promising new paradigm for internal consistency checking.
    
2.  **Clarity and Technical Soundness:** The motivation for the Generative Self-Refinement method is clearly articulated. The proposed methodology is technically sound, and the paper is well-structured and easy to follow, making the complex topic accessible.
    
3.  **Extensive Experimental Validation:** The experimental validation is extensive and comprehensive. The authors thoroughly investigate the impact of the hybrid Supervised Fine-Tuning (SFT) dataset on the model's refinement ability, demonstrate the robust scaling benefits across different model sizes, and analyze the accuracy evolution as the number of candidates is increased. This thoroughness is commendable.

### Weaknesses
- **Baseline selection need to be improved:** The primary weakness lies in the authors compare the GSR with pure test-time scaling (TTS) methods as baseline, which is a bit unfair. Given that **GSR requires dedicated Supervised Fine-Tuning (SFT) on a specialized dataset** (collected following Section 3.3) to unlock the self-refinement skill, it operates as a _training technique_ rather than a generalized, plug-and-play TTS method. This distinction is critical and should be clarified. Furthermore, the experimental results (e.g., Qwen2.5-7B-Instruct on several mathematical benchmarks) show that the proposed method (`selfRef@4`) often performs comparably to or _worse_ than the simpler baseline of majority voting (`maj@4`) without the proposed SFT process. This lack of consistent, significant uplift, combined with the prerequisite training, undermines its positioning as a superior TTS method. The authors must either justify this classification or **compare GSR against other methods that explicitly train internal reasoning or reflection abilities** (e.g., Chain-of-Thought approaches with explicit verification steps).
- Lack of Sensitivity Analysis to Candidate Diversity and Quality: The efficacy of both the self-refinement and majority voting components is inherently tied to the **quality and diversity of the initial candidate pool**. The current analysis lacks a systematic investigation into the sensitivity of GSR performance against these critical metrics. A thorough ablation study is warranted to explore how GSR behaves when candidates exhibit low diversity (i.e., multiple identical, but incorrect, responses) or low aggregate quality. Providing stronger evidence for its robustness beyond a simple majority selection process is necessary.

### Minor Comments
- Subscripts should be wrapped in \text{} when it is not a variable. For example, $\mathcal{D}_{\text{direct}}$ and $\mathcal{D}_{\text{selfR}}$ instead of $\mathcal{D}_{direct}$ and $\mathcal{D}_{selfR}$.

### Questions
- How do you guarantee that the teacher model has the ability of drawning correct answers from the candidates proposed by the students?
- The accuracy saturates as the number of candidate increases. This contradicts the goal of test-time "scaling" methods, where we expect performance gains from increased computational resources. Have the authors investigated the scalability of the Supervised Fine-Tuning (SFT) dataset size?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a parallel test time scaling framework labeled Generative Self-Refinement, which improves the reasoning of a large language model by creating parallel traces and iteratively improving them using a verification pipeline. This refinement approach, along with the proposed hybrid training solution, improves downstream reasoning benchmarks by significant margin.

### Strengths
- Strong empirical results from the base model, but likely due to knowledge distillation and not self refinement.

### Weaknesses
- The method introduced by the paper is defined as self-refinement, however the only self-refinement part is the first set in Section 3.2, where the model itself is prompted to become a critic of its own generations. Subsequently, the proposed approach turns into a distillation problem, where a teacher model is used to provide additional signal. I fail to understand how this can be deemed as "self-refinement".
- From the experiment details, it is clear the proposed approach is distillation of QwQ-32B model into Qwen2.5-7B-Instruct. However, the baselines does not reflect the performance of QwQ-32B (from Table 1). Interestingly, QwQ-32B significantly outperforms the proposed method on AIME24 (79.5 vs 66.0 reported in the paper)
- Therefore, all reported results can just be taken to be a distillation effort from QwQ-32B, leaving the proposed title and method misleading as the improvements are not coming from self-refinement.
- If the authors intend to treat this as a distillation work, then significant rewriting of the draft is required, along with pure distillation based baselines.

### Questions
- Please clarify where self-refinement is being useful in this setup.

### Soundness
1

### Presentation
2

### Contribution
1

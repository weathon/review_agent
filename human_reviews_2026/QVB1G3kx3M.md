# Reinforcement Learning for Better Verbalized Confidence in Long-Form Generation

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 6

## Abstract
Hallucination remains a major challenge for the safe and trustworthy deployment of large language models (LLMs) in factual content generation. Prior work has explored confidence estimation as an effective approach to hallucination detection, but often relies on post-hoc self-consistency methods that require computationally expensive sampling. Verbalized confidence offers a more efficient alternative, but existing approaches are largely limited to short-form question answering (QA) tasks and do not generalize well to open-ended generation. In this paper, we propose LoVeC (Long-form Verbalized Confidence), an on-the-fly verbalized confidence estimation method for long-form generation. Specifically, we use reinforcement learning (RL) to train LLMs to append numerical confidence scores to each generated statement, serving as a direct and interpretable signal of the factuality of generation. We introduce two novel evaluation settings, free-form tagging and iterative tagging, to assess different verbalized confidence estimation methods. Experiments on three long-form QA datasets show that our RL-trained models achieve better calibration and generalize robustly across domains. Also, our method is highly efficient, being 20x faster than traditional self-consistency methods while achieving better calibration.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes LoVeC, an RL–based framework that enables LLMs to generate confidence scores alongside long-form factual statements in one-time generation. It introduces two new evaluation settings, free-form tagging and iterative tagging, to assess confidence calibration in long-form text. Experiments show that LoVeC generalizes robustly across domains and achieves better calibration than traditional self-consistency methods as well as other baselines.

### Strengths
1. This paper introduces an RL-based method for generating and calibrating confidence scores in long-form factual text using LLMs.  Extensive experiments demonstrate the effectiveness of the proposed approach across both free-form and iterative tagging settings.

2. The authors thoroughly analyze the proposed method, as well as its baselines, and perform ablation studies on each component, providing a clearer understanding of the model’s design and effectiveness.

### Weaknesses
1. The presentation of the paper needs improvement. For example, in line 258, the authors mention using additional subordinate rewards (e.g., informativeness and format rewards) in Appendix B, yet the appendix does not provide clear details about them. While $r^{\text{correct}}$ is defined as the total factuality score judged by the reward model, the specific details and prompts of these subordinate rewards, which are crucial for the success of the RL algorithm, remain unclear.

2. The authors do not provide intermediate validation or analysis of model behavior during RL training, such as how confidence assignments evolve over time. Since the reward formulation in Eq. (6) appears potentially unstable for online RL training, including intermediate evaluations, such as tracking reward trends or showcasing intermediate model outputs for correctness and confidence, would strengthen the paper’s persuasiveness.

### Questions
My questions mainly concern Weakness 2, specifically the intermediate policy behaviors during online RL training. Providing additional analysis in this area could help address some of my concerns.

### Soundness
2

### Presentation
2

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
This work proposes a relatively novel setting, in which the model must directly decode its sentence-wise confidence in long-form QA in a verbalized fashion.
The method, LoVeC, uses either DPO or GRPO to perform online or offline fine tuning of the models to assess own uncertainty.
The authors devise a specialised reward that optimizes its calibration through RL. 
This resulted in improved calibration on selected benchmarks.

### Strengths
The novel setting provides fine grained verbalized confidence estimates and may even be useful in situations where only the output of the model is available without any additional information. RL use for calibration is an interesting approach and the optimization objectives seem somewhat innovative. The authors perform many ablations to determine the best way to fine tune model to produce verbalized confidence scores.

### Weaknesses
0. Typos (not affecting my assessment, just heads up):
    1. Line 461: double full stop.
1. Evaluation:
    1. The evaluation with fact extraction and fact checking pipeline is complex. Only QA setting is considered, but possibly the paper would benefit from evaluation on synthetic problems or somewhat more verifiable domains (see e.g. [4]). 
    2. (major) Several generally accepted strong uncertainty estimation methods (i.e. Perplexity or GNLL of a segment [1][2] or intenal states based methods, i.e. [3]) have not been considered. These methods can be easily applied in this setting, since log probabilities are usually available. 
    3. Somehow "Vanilla" prompting approach is consistently better than Baseline Methods and LUQ on Windhallu. From the description Vanilla and Verb-Conf should be roughly identical. Are the same models used everywhere throughout evaluation?
    4. To follow up the previous point, it appears that the model for LoVeC might be fine tuned directly on WildHallu. Would this be unfair to other methods?

### References:
1. Aichberger, L., Schweighofer, K. & Hochreiter, S. Rethinking Uncertainty Estimation in Natural Language Generation. Preprint at https://doi.org/10.48550/arXiv.2412.15176 (2024).
2. Fadeeva, E. et al. LM-Polygraph: Uncertainty Estimation for Language Models. Preprint at https://doi.org/10.48550/arXiv.2311.07383 (2023).
3. Kossen, J. et al. Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs. Preprint at https://doi.org/10.48550/arXiv.2406.15927 (2024).
4. Ielanskyi, M., Schweighofer, K., Aichberger, L. & Hochreiter, S. Addressing Pitfalls in the Evaluation of Uncertainty Estimation Methods for Natural Language Generation. Preprint at https://doi.org/10.48550/arXiv.2510.02279 (2025).

### Questions
1. Could one calibrate, e.g. NLL of segments of interest directly with this approach instead of producing verabalized confidence?
2. How does the reward function deal with invalid confidence assignment (e.g. in Tab.3 11,12,... appear)? 
3. How does ablation in paragraph at line 452 relate to the known effect of judge LMs preference for own outputs?
4. Would the method be good at assessing prompt perturbation? I.e. the predicted confidences should be lower for randomly perturbed prompts.

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
4

### Summary
This paper proposes LoVeC (Long-form Verbalized Confidence). It is a reinforcement learning framework that trains language models to generate numerical confidence scores alongside long-form factual text. Using on-policy (GRPO) and off-policy (DPO) variants, the model learns to align its self-reported confidence with factual correctness, which avoids costly post-hoc sampling. The authors also introduce two evaluation settings: 1. free-form tagging (confidence generated inline) and 2. iterative tagging (confidence assigned to fixed text). The proposed method is faster than existing works.

### Strengths
- The paper is well-organized, easy to follow, and read
- Uncertainty estimation for long-form generation is an important topic to explore.
- The proposed method is efficient, and this aspect is supported by experimental evidence as well.

### Weaknesses
- I find the main contribution of the proposal limited. Using existing RL algorithms to train models to generate confidence scores per sentence lacks originality. This is a combination of sentence-level uncertainty estimation and RL fine-tuning for verbalized confidence, which both exist in the literature. While the method is well-executed and may yield practical benefits, it does not introduce a fundamentally new algorithmic component or theoretical insight beyond this combination.
- The only supervised method in the benchmark is the proposed method, which gives a significant advantage. It has been previously shown that fine-tuning the model for p(true) and verbalized confidence improves performance. I think fine-tuned versions of these methods are essential. 
- Does the proposed method yield calibrated confidence scores when we do sampling? Analysis of this is important since temperature sampling is a common practice.

### Questions
- Currently, the reward is computed by aggregating all confidence scores in the generation. And all confidence tokens receive the same reward. For instance, the confidence score of a sentence might perfectly match the ground truth; however can still receive a negative or a positive reward due to other sentences. Would a more granular reward assignment approach improve the calibration level? I think exploring this direction both improves the contribution/originality of the paper and makes it more interesting.
- During iterative tagging evaluation, the fixed text is generated by the confidence estimaton model, right?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces LoveC, a method to make LLMs verbalize their confidence (0–10) sentence-by-sentence during long-form generation, trained with reinforcement learning so that the stated confidence aligns with factual correctness. It targets two common gaps: (i) prior long-form confidence methods are mostly post-hoc and expensive (multi-sample self-consistency, extra claim-extractor models), and (ii) most verbalized confidence work is short-form. LoveC outputs statements and confidence in one decoding pass and is evaluated in two settings: Free-form Tagging (model generates answer + confidence) and Iterative Tagging (given a fixed answer, the model tags each sentence’s confidence). The authors experiment with two variants: LoveC GRPO (on-policy RL) and LoveC DPO (off-policy RL). Experiments on WildHallu, Bios, and PopQA show strong calibration improvements.

### Strengths
- Simple and effective approach: LoveC applies a straightforward idea by training the model to verbalize numeric confidence inline, achieving strong calibration 

- Strong empirical performance: Both LoveC GRPO and DPO consistently outperform prior methods across multiple datasets and metrics, showing that the approach is robust and generalizable.

- Efficient and practical: The method produces confidence scores quickly during generation, making it efficient 

- Clear and well-written paper: The writing is clear and well-structured and well explained


Overall, this is a promising paper and I'm happy to increase scores if my concerns are adressed

### Weaknesses
- The paper would benefit from additional ablations exploring alternative reward formulations or scaling choices to show how each component contributes to performance.

- It would be interesting to see whether combining LoveC with self-consistency or other sampling-based calibration methods could further improve confidence estimation.

### Questions
- In Table 2 how does the literature sota LUQ perform at par/worse than vanilla baselines?

### Soundness
3

### Presentation
3

### Contribution
3

# EgoExo-Con: Exploring View-Invariant Video Temporal Understanding

- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Can Video-LLMs achieve consistent temporal understanding when videos capture the same event from different viewpoints? To study this, we introduce EgoExo-Con (Consistency), a benchmark of comprehensively synchronized egocentric and exocentric video pairs with human-refined queries in natural language. EgoExo-Con emphasizes two temporal understanding tasks: Temporal Verification and Temporal Grounding. It evaluates not only correctness but consistency across viewpoints. Our analysis reveals two critical limitations of existing Video-LLMs: (1) models often fail to maintain consistency, with results far worse than their single-view performances. (2) When naively finetuned with synchronized videos of both viewpoints, the models show improved consistency but often underperform those trained on a single view. For improvements, we propose View-GRPO, a novel reinforcement learning framework that effectively strengthens view-specific temporal reasoning while encouraging consistent comprehension across viewpoints. Our method demonstrates its superiority over naive SFT and GRPO, especially for improving cross-view consistency. All resources will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the ability of Video-LLMs to maintain consistent temporal understanding of events captured from different viewpoints. The authors introduce EgoExo-Con, a new benchmark consisting of synchronized egocentric and exocentric video pairs with refined natural language queries for temporal verification and grounding tasks. Through extensive experiments, they reveal that current SOTA models struggle significantly with cross-view consistency, often achieving scores that are much lower than their single-view performance. The paper also demonstrates that naively fine-tuning models on a mix of both viewpoints is insufficient and can even degrade performance. To address these shortcomings, the authors propose View-GRPO, a reinforcement learning framework that encourages models to develop view-specific reasoning chains while aligning their final temporal predictions, showing improved consistency over standard fine-tuning methods.

### Strengths
- EgoExo-Con benchmark is a significant contribution. The authors have been sourcing data from diverse datasets and performing careful filtering and human-backed refinement of queries to ensure they are unambiguous and visible from both perspectives.
- The paper provides a comprehensive evaluation of a wide range of Video-LLMs. The key finding that cross-view consistency scores are "barely over half their single-view performance" and that naive multi-view training fails to improve consistency, is a crucial insight that highlights a fundamental weakness in current architectures and training paradigms.
- The proposed View-GRPO method offers a promising direction for improving consistency.

### Weaknesses
- The effectiveness of the proposed View-GRPO method is demonstrated only on the Qwen2.5-VL model family. While the results are positive, application to wider range of models would be necessary to make a stronger claim about the generalizability and robustness of the approach.
- Authors showed that the choice of judge model impacts performance. However, the issue of reliability and bias in judge is left for future work. Given its central role in the method's success, a more thorough analysis or ablation study (e.g., removing the reasoning reward) is wanted.
- The View-GRPO is an incremental approach built on top of GRPO. The novelty lies in its application to the cross-view consistency problem and the design of the reward function.

### Questions
This paper presents a valuable and timely contribution by defining and benchmarking the problem of view-invariant temporal understanding. The EgoExo-Con dataset and the accompanying analysis are strong points that will benefit the community.
The proposed View-GRPO method, while seems promising, has weaknesses in presentation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper studies the cross-view consistency in predictions of video models.

* First, the authors propose a benchmark (EgoExo-Con) of 491 synchronised videos and 3,178 temporal-bounded queries. While video models perform well in individual views, their predictions are not consistent across views.
* It is shown that naive SFT on a mix of ego-exo video data does not improve this consistency likely due to conflicting priors.
* Finally, the authors propose View-GRPO, an RL based approch (along with a training dataset View30K) to enhance temporal reasoning while encouraging view-invariant video comprehension. This outperforms SFT and standard GRPO. View-GRPO includes three reward signals:
    1. Format reward (for answer structuring)
    2. Accuracy: irrespective of view, the final answer consistency is encouraged by this reward
    3. Reasoning reward: the model is encouraged to output reasoning traces similar to those of GPT5 - this balances cross-view temporal reasoning as well as certain view-specific details.

### Strengths
1. **Useful problem and benchmark.** The proposed problem on cross-view consistency is interesting and has several practical applications like learning from human demonstrations. The ExoEgo-Con benchmark, although not as big in size, is thoroughly constructed and is useful in measuring this consistency.

2. **Strong baseline.** The proposed View-GRPO is simple, well-formulated and shows strong performance on the proposed benchmark.

3. **Presentation.** The paper is very well written and was easy to understand.

### Weaknesses
1. Instead of enforcing cross-view consistency only through the final answer, the View-GRPO method relies on matching reasoning traces for cross-view consistency. This means that the consistency is enforced via the language space and not directly via visual correspondence. In some narrow cases this can be a limitation for examples where it is hard to describe certain visual elements in language.

2. There is still a lot of room for improvement in terms of consistency scores (Tab 3). While this highlights strength of the benchmark, it could strengthen the paper if some discussion is included on what the authors project as potential ways to improve performance (more ego data in pre-training? entirely new algorithms/architectures?).

3. (Minor) There is some overlap with prior work [1] in terms of problem formulation. However, it is not published and can be considered concurrent. Nevertheless, it may be nice to include a deeper discussion on the differences.

4. Some discussion on whether the proposed ViewGRPO affects performance on other standard video benchmarks is desirable. I am not sure if it is LoRA or full finetuning. In case of latter, such evaluation is much needed.

[1] EgoExoBench: A Benchmark for First- and Third-person View Video Understanding in MLLMs
Yuping He, Yifei Huang, Guo Chen, Baoqi Pei, Jilan Xu, Tong Lu, Jiangmiao Pang

### Questions
1. Audio is view-invariant. Have you considered including audio in the mix and using Qwen-Omni like models?

2. Is there a way to automatically quantify if a given query for a video is exo-friendly, ego-friendly or both-friendly? For example, "a person laughing" is likely only exo-friendly and "a person holding something tight" is ego-friendly (clearly visible in ego view but perhaps not visible in exo view).

3. Another type of consistency could be to pass both ego and exo views together and ask if they depict the same action or ask to match coresspondences. Have you considered something like that?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces evaluation benchmark EgoExo-Con to evaluate cross-view video temporal understanding. The authors propose View-GRPO and construct View30K, to explicitly enhance temporal reasoning.

### Strengths
1. The paper is clearly written. 
2. The proposed EgoExo-Con eval set can be useful for this area of research.
3. The proposed reinforced approach enhanced view-invariant comprehension in video-LLMs.

### Weaknesses
1. The proposed evaluation set EgoExo-Con only contains 491 items, which is pretty small and is hard to say weather it is simply finding a hard set for the current Video LLMs.
2. The close-sourced and human performance are reported on a randomly sampled subset, which may be sensitive to sample selection and cannot be fairly compared with open-sourced models and proposed model.
3. In Figure 1 (b), I don't think it is appropriate to expect model understanding "put a knife" from the provided exo video. Because the object of interest is too small. Even for human, it is hard to identify that action. The top example in Figure 3 also makes me confused about how could human even be able to understand the person is opening a cabinet door. 
4. The evaluation is pretty limited, only on the proposed set. And compared Table 1 and Table 2, the proposed method is not better than previous models, e.g. TimeChat-VT.

### Questions
1. Can you somehow apply your View-GRPO approach with existing model on tasks like Learning Fine-grained View-Invariant Representations from Unpaired Ego-Exo Videos via Temporal Alignment? It is also tackling the similar problem.
2. How many video frames do you take for training? The training time seems too long --"8 xA100 GPUs and requires over 1 day for the 3B model and 2 days for the 7B model".

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces EgoExo-Con, a benchmark to evaluate the capability of VideoLLMs in understanding temporal consistency across ego- and exocentric viewpoints of the same events. The benchmark focuses on two tasks, including temporal verification (binary QA) and temporal grounding. The authors found that the existing VideoLLMs achieve good single-view performance while showing poor cross-view consistency. To this end, they propose View-GRPO, an RL-based fine-tuning framework, extending Group Relative Policy Optimization with an additional reasoning reward and a curated dataset to enhance cross-view temporal reasoning. Empirical results on EgoExo-Con show that View-GRPO improves consistency over standard SFT and GRPO baselines.

### Strengths
**[S1] Writing and motivation**
- The paper is well-written and easy to follow. The motivation is clearly presented.

**[S2] Experiments**
- The paper covers extensive range of experiments, including evaluation both open- and closed-source models, reporting detailed per-subset results (CharadesEgo, LEMMA, EgoExo-4D), and fine-tuning analyses.

### Weaknesses
**[W1] Dataset and task clarification**
- EgoExo-Con combines pre-existing datasets, so its domain diversity still depends on those sources. The paper claims “comprehensive” coverage, but 491 pairs is small compared to modern multimodal benchmarks.
- Evaluating temporal consistency across viewpoints is critical in view-invariant video understanding. Traditionally, cross-view temporal consistency has been evaluated through cross-view phase progression or Kendall's $\tau$. Additionally, cross-view frame or clip retrieval may also measure the capability of temporal consistency of the models. Compared to previous evaluation tasks, temporal verification may provide a limited insight, as it measures whether the model identifies an event as the same or not. The necessity of temporal verification should be clearly presented in the paper.
- In sum, the contribution of the introduced benchmark may not be significant.

**[W2] View-GRPO**
- View-GRPO itself seems not inherently specific to cross-view reasoning. In other words, rewards corresponding to each view can be optimized separately, and co-optimization across viewpoints is not guaranteed.

**Minor issues**
- Repeated entries in reference (e.g., Feng et al., 2025a/b and Grauman et al., 2024a/b)
- Typo: L77 their sing-view
- Figure 6 should be moved before Figure 7 (or swap the order of figures)

### Questions
- There are substantial performance gaps between closed-source models and open-source models, as shown in Table 1. What makes closed-source models stronger? This is just a question to know the authors' opinion.
- Please see weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

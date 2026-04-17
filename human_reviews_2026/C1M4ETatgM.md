# Vision-SR1: Self-Rewarding Vision-Language Model via Reasoning Decomposition and Multi-Reward Policy Optimization

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Vision-Language Models (VLMs) often suffer from visual hallucinations -- generating things that are not consistent with visual inputs -- and language shortcuts, where they skip the visual part and just rely on text priors. These issues arise because most post-training methods for VLMs rely on simple verifiable answer matching and supervise only final outputs, leaving intermediate visual reasoning without explicit guidance. As a result, VLMs receive sparse visual signals and often learn to prioritize language-based reasoning over visual perception. To mitigate this, some existing methods add visual supervision using human annotations or distilled labels from external large models. However, human annotations are labor-intensive and costly, and external signals can introduce high latency cost.

In this paper, we introduce Vision-SR1, a three-stage self-rewarding reinforcement learning method that improves visual reasoning without relying on external visual supervision. Vision-SR1 decomposes VLM reasoning into two components: visual reasoning and language reasoning, where the model is first prompted to produce self-contained visual descriptions sufficient to answer the question without referring back to the input image, before jointly optimizing both visual and language reasoning through our multi-reward loss objective. To validate this self-containment, the same VLM model is re-prompted to perform language reasoning using only the generated visual reasoning as input to compute visual reward.  The final reward is computed through a decoupled reward-advantage framework, where visual reward and language reasoning reward each have their advantages, log probabilities, and KL divergence calculated separately. This decoupling enables more fine-grained reward computation by preventing the entanglement of heterogeneous reward signals. Our experiments show that Vision-SR1 improves visual reasoning, mitigates visual hallucinations, and reduces reliance on language shortcuts across diverse vision-language tasks, while being more efficient than methods that rely on external visual reward models, which require additional GPUs to host. In contrast, Vision-SR1 introduces no extra GPU overhead beyond that of standard training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Vision-SR1, which decomposes VLM reasoning into visual perception and language reasoning, then self-verifies whether the perception alone is sufficient to answer the question, providing a self-visual reward without external supervision to reduce hallucinations and language shortcuts.

### Strengths
- Clear and well-written.
- Method is effective across many tasks with thorough experimental validation.
- Provides intuitive theoretical analysis supporting the design choices.
- Raises several valuable, thought-provoking questions for the community.

### Weaknesses
- Visual perceptions that are sufficient to answer the question without referring back the input image: is it possible (some visual information is hard to express in language)? Is it necessary (visual reasoning might be understood as a carrier of latent space reasoning, used to aggregate and evolve visual information and implicit thinking via attention)?
- “If the correct answer is derived, a self-visual reward is assigned.” Which part of generation is this visual reward used to encourage in the paper? I am a bit confused. Can this visual reward be assigned solely to the visual perception part in the first rollout? What would the effect be?
- Only Qwen-2.5-VL is trained; how about other model series? Adding them would be more convincing.
- Is the Language Shortcut Rate metric reasonable? First, “If the evaluator can reproduce the correct ground-truth answer using only this information, the generated visual reasoning is deemed self-contained.” How many cases are there where the evaluator reproduces the wrong answer but the generated visual reasoning is actually self-contained? It would be better to have a more detailed human analysis (on a small amount of data is fine).
- Curious about the method’s performance on spatial reasoning tasks, because I think many perceptions and reasoning in 3D space are hard to express in natural language.
- Typo: an extra comma at line 067.

### Questions
- Additional discussion: Whether the generated visual reasoning contains all information needed to answer the question—Is it possible, and is it necessary?

- Additional experiments:
  - If possible, try assigning the visual reward solely to the visual perception part in the first rollout.
  - If time permits, train models beyond the Qwen-2.5-VL series.
  - If time permits, evaluate Vision-SR1 on spatial reasoning tasks such as VSI-Bench [1], MMSI-Bench [2], and OmniSpatial [3].

- Additional human analysis: In the Language Shortcut Rate evaluation, how many cases are there where the evaluator reproduces the wrong answer while the generated visual reasoning is actually self-contained? It would be better to include a more detailed human analysis (a small amount of data is sufficient).


[1] Yang, Jihan, et al. "Thinking in space: How multimodal large language models see, remember, and recall spaces." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

[2] Yang, Sihan, et al. "MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence." arXiv preprint arXiv:2505.23764 (2025).

[3] Jia, Mengdi, et al. "OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models." arXiv preprint arXiv:2506.03135 (2025).

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
4

### Summary
This paper proposes Vision-SR1, a self-reward reinforcement learning (Self-RL) framework designed to address visual hallucinations and language shortcuts in VLMs. It decomposes reasoning into two stages: visual perception and language reasoning. The model first generates a self-contained perceptual description (c), then re-prompts itself and relies solely on c to answer the question, thereby generating a self-derived visual reward. The method employs a multi-reward strategy for optimization, computing decoupled losses for both visual and answer rewards, effectively enhancing visual reasoning capability while reducing the LSR.

### Strengths
(1) Vision-SR1 allows VLMs to self-verify whether their visual perception is self-contained and sufficient to answer a given question. This effectively provides dense and adaptive reward signals for intermediate visual reasoning steps, avoiding the high costs, biases, and reward hacking risks associated with external human annotations or large model distillation.
(2) Advantage functions and loss terms are computed separately for visual and answer rewards, effectively decoupling the training signals. This ensures balanced reinforcement of both visual perception and language reasoning, while also avoiding the signal sparsity and entanglement caused by the traditional approach of summing rewards.

### Weaknesses
(1) Vision-SR1 uses the model itself as a verifier, evaluating the quality of visual perception by re-prompting the same-policy VLM. This may lead to circular dependencies and inherent biases: the model could learn to generate perceptions that "convince itself" rather than perceptions that truly align with the image.
(2) By re-prompting the same-policy VLM to assess visual perception quality, the method relies on two rollouts during training, which may effectively double the training cost.

### Questions
The authors mention that self-rewarding may lead to reward hacking. Could the authors quantitatively analyze the intrinsic bias of the reward model in Vision-SR1? Furthermore, would it be possible to introduce lightweight external verification (e.g., a frozen CLIP or BLIP module) to cross-check the true visual consistency of the self-reward, thereby mitigating the circular dependency issue inherent in self-rewarding?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose an improvement to vision-R1, decoupling the visual perception description generation part and the answer generation part for reward, to avoid LVLMs overly relying on language reasoning shortcuts. The method is clear, the experiments are solid, and the comparison with vision-R1 is clear.

### Strengths
1. The writing is clear, the method description is clear and specific, very straightforward, and the experiments are solid.

### Weaknesses
1. The only concern is novelty, as the idea of first generating image description and then generating answer based on the description has been explored before. [1]
2. Requires inference twice, which would be time-consuming and not infrastructure-friendly.

[1] Multimodal Chain-of-Thought Reasoning in Language Models

### Questions
1. Have the authors analyzed how the model's attention changes after this training, and whether it affects its distribution on visual tokens?

2. How do the authors ensure that the generated image description is self-contained? Since the model can see both the question and the image, it could potentially guess the answer and then generate an image description where the description already contains answer-related information.

3. How do the authors guarantee that decoupling the visual perception stage and the answer generation stage can avoid language shortcuts? From a training perspective, in the second stage, after removing the image input, wouldn't the model be more prone to relying on language shortcuts? So why does using this two-stage decoupling method mitigate language shortcuts and visual hallucination? Why can't a single-stage approach achieve this? Is there a theoretical explanation? For example, changes in attention patterns over visual tokens?

4. How do the authors address the infrastructure inefficiency issue for this two-stage training approach?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Vision-SR1, a self-rewarding reinforcement learning framework that decomposes vision-language reasoning into visual perception (i.e., visual descriptions) and language reasoning stages. The method introduces Multi-Reward Policy Optimization, which separately computes and combines visual perception rewards and answer accuracy rewards, avoiding entangled learning signals. Besides, Vision-SR1 generates self-contained visual perceptions and verifies them using the model itself, improving grounding and reducing hallucination. Experiments on 47K examples show gains across visual reasoning, math, and hallucination benchmarks, outperforming Vision-R1 and related baselines.

### Strengths
I believe the studied direction is important -- overcoming the VLMs overly relying on learned priors, particularly text priors. Also, the math equation is descried clearly in the paper. 

Empirical results on a couple of comprehensive benchmarks are good, demonstrating improvements. It mostly contains general visual understanding and multimodal mathematical reasoning. Some ablation studies are included as well. It also curated Vision-SR1-47K from some open source VLM benchmarks (table 1), while I am unsure if data may pollute the test set or not, since the performance of finetuning over on this data improve the performance quite a lot --- Vision-R1 47K data (fair comparisons).

### Weaknesses
From my perspective, the proposed two-stage pipeline is a bit ad-hoc due to it explicitly enforces there could be two stages, where the first stage generate some visual descriptions, and then the second stage is doing the reasoning. Such pipeline is expectedly not the ultimate goal for VLMs. It is basically a captioning models + a LLM --> this is not something we want to achieve for VLMs. Such behavior should emerge in VLM inference process, not enforced by constructing a small scale datasets with specific setting ⟨visual reasoning⟩ c ⟨/visual reasoning⟩.

For evaluation,I would recommend to include some benchmarks directly motivated by language shortcuts and text priors, such as Probing Visual Language Priors in VLMs and Winoground: Probing vision and language models for visio-linguistic compositionality. 

Also, extend to larger scale of models, such as 72B, will make the whole evaluation solid. Extending the experimental results to other base-model beyond Qwen-2.5-VL is also helpful.

### Questions
Could the author consider the experimental questions shown above, particularly I curious how the finetuned model perform in the datasets which directly motivated by VLM overly relying on language priors? I will consider the response and other reviewers' comments to adjust my final score, but plz feel free to skip any experiments you believe is unreasonable. 

Regarding the two stage things, I will definitely discuss with other reviewers and AC.

### Soundness
2

### Presentation
3

### Contribution
2

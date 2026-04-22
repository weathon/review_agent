# Leveraging MLLMs for Zero-Shot Action Recognition: Concise, Discriminative and anti-Hallucination Prompting

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 4, 6

## Abstract
Leveraging the capabilities of large language models (LLMs), multi-modal LLMs (MLLMs) show great promise for zero-shot action recognition (ZSAR). However, current MLLM-based approaches often struggle to accurately locate the right action from many, largely due to issues such as lengthy, vague prompts and hallucinated outputs. In this paper, we introduce CDantiHalP concise, discriminative and anti-hallucination prompting), a novel LLM-driven approach to enhance MLLM performance in ZSAR. CDantiHalP is a training-free, post-refinement method designed to improve recognition accuracy for any baseline model. It consists of two core components: (1) concise, discriminative prompting to effectively distinguish confused action pairs, and (2) logic-contradictory hallucination detection (LogCHalD) to identify and mitigate hallucinations. Rather than relying on MLLMs to select from a broad set of labels, CDantiHalP leverages their strength in pairwise comparison of specific concepts. The use of concise-discriminative prompts highlights the distinguishing features between confused actions, guiding MLLMs to focus on critical differences while remaining alert to potential hallucinations. The LogCHalD framework further enhances response reliability by using a logic-contradictory strategy to detect hallucinated responses for each confused action pair. During inference, CDantiHalP assesses hallucination risk and emphasizes consistency across MLLM outputs to mitigate the impact of hallucinations. Extensive experiments demonstrate that CDantiHalP achieves state-of-the-art performance on various ZSAR datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the problem of zero-shot action recognition (ZSAR) using multi-modal large language models (MLLMs) without additional training data. The authors identify two key limitations of current MLLM-based ZSAR methods: 
1) Ineffective prompt design: lengthy or vague prompts that reduce discriminative power.
2) Hallucination:  incorrect or inconsistent predictions due to over-reliance on priors or noisy video frames.

the paper proposes CDantiHalP which is a training-free, post-refinement framework composed of:
1) Concise, Discriminative Prompting (PromptCD): Automatically generated prompts contrast confused action pairs (e.g., throw discus vs. throw hammer) rather than having the model choose from an entire label list.Prompts are tagged as [image]/[video] depending on whether static or dynamic cues are most relevant.
2) Logic-Contradictory Hallucination Detection (LogCHalD): a logic-contradictory prompt (PromptLC) to test the consistency of MLLM responses. If the model gives the same answer to contradictory queries, hallucination is detected.
3) Recognition Refinement: LLM reasoning (PromptRR) to reconcile responses and refine final predictions, prioritizing consistent, low-risk outputs.

Empirical results on UCF101, HMDB51, and Kinetics-600 show that CDantiHalP achieves state-of-the-art zero-shot performance and improves existing baselines (e.g., TEAR, XCLIP, ActionCLIP).

### Strengths
1) The paper includes comprehensive experiments and ablations validate every component (prompt design, hallucination mitigation, frame/video tagging, hyperparameters). The paper also evaluates multiple MLLM backbones (LLaVA-NeXT-Video, VideoLLaMA2, VILA) and demonstrates cross-model robustness.
2) CDantiHalP offers a practical, scalable enhancement for ZSAR pipelines without additional training cost. The logic-contradictory hallucination detection (LogCHalD) is a creative, lightweight mechanism for assessing hallucination risk without retraining.

### Weaknesses
1) The approach relies on a co-occurrence-based “confused pair dictionary” generated from baseline model outputs. This might limit applicability to new datasets or unseen classes where no prior co-occurrence statistics exist. A discussion of adaptive or online pair construction would be valuable.
2) For LogCHalD, quantitative metrics for hallucination detection accuracy are not explicitly reported. A confusion matrix or correlation between hallucination risk and performance degradation would make the claim stronger.
3) Since multiple MLLM queries are made per video (PromptCD and PromptLC, sometimes per frame), runtime cost could be substantial. The paper should include timing or computational analysis.

### Questions
1) How sensitive is CDantiHalP to the specific LLM used (e.g., GPT-3.5 vs. open-source LLMs)? Can smaller models (e.g., Vicuna, Mistral) reproduce the same quality prompts?
2) Could the dictionary be built dynamically during inference, e.g., via nearest-neighbor semantic similarity rather than fixed co-occurrence?  How does performance degrade if the dictionary is incomplete or noisy?
3) Have the authors evaluated LogCHalD’s ability to correctly identify hallucinations (precision/recall)? Does the threshold  θ=0.8 generalize across datasets, or was it tuned per dataset?

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
3

### Summary
The proposed method is a training-free, post-refinement pipeline for Zero-Shot Action Recognition that converts a multi-class problem into pairwise discrimination between likely-confused classes. It uses an LLM to auto-generate a concise, discriminative prompt (PromptCD) and a logic-contradictory prompt (PromptLC) to probe hallucinations (LogCHalD). At inference, it refines a baseline model’s top-2 using these prompts and a consistency rule.

### Strengths
1. Instead of  long, label-list prompts (token-heavy) it transforms to binary, feature-targeted questions which reduces prompt length/noise.
2.  The method  integrates with multiple MLLMs  and yields consistent gains—useful in low-resource or deployment-constrained settings.
3. Exhaustive ablation study.

### Weaknesses
1.  The confused-pair dictionary D is built by “randomly sampling a subset of test videos,” collecting top-2 co-occurrences, and keeping pairs above a certain λ. Ideally it would good to derive D from train/auxiliary data or a held-out validation set and freeze it before testing. It appears that the current settings is a test-aware meta-tuning of the prompt inventory.
2.  It seems that LogCHalD will work when objects are mutually exclusive within a clip, but can fail when both objects can co-occur (multi-person scenes, equipment in the background).
3. An adaptive or learned threshold (or cost-sensitive policy) would be preferable instead of the static threshold, especially given class imbalance and varying yes-bias across models. Since Different MLLMs may have different bias profiles., Also r’s distribution will shift across datasets.
4. The method requires multiple MLLM passes.

### Questions
Can you think of an easy solution for the adaptive threshold?  Also did you purposely exclude Something Something Dataset considering its long temporal dependence ?

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
3

### Summary
This paper introduces CDantiHalP, a training-free, post-refinement framework designed to improve zero-shot action recognition (ZSAR) using multimodal large language models (MLLMs). The core idea is to replace vague, lengthy prompts with concise and discriminative ones (PromptCD) that focus on distinguishing confused action pairs. Furthermore, a logic-contradictory hallucination detection module (LogCHalD) is proposed to verify the consistency of responses and identify hallucinated outputs. Experiments conducted on UCF101, HMDB51, and Kinetics-600 using VILA, LLaVA-NeXT-Video, and VideoLLaMA2 demonstrate consistent performance gains across datasets. The method achieves state-of-the-art accuracy among training-free approaches and shows strong generalization across models and baselines.

### Strengths
1. The proposed Concise-Discriminative Prompting effectively exploits the comparative reasoning strength of MLLMs, turning a complex multi-label selection problem into pairwise action discrimination.

2. The Logic-Contradictory Hallucination Detection (LogCHalD) is a clever design that uses contradictory prompts to test the internal logical consistency of model outputs, providing an interpretable way to identify hallucinations.

3. The framework requires no fine-tuning or additional data, and can refine predictions from any existing ZSAR baseline, showing strong portability.

4.  The paper presents solid empirical evaluations with ablations, qualitative visualizations, and cross-model generalization studies.

### Weaknesses
1. While the paper reports impressive results on VILA (Lin et al., 2024), LLaVA-NeXT-Video (Zhang et al., 2024), and VideoLLaMA2 (Cheng et al., 2024), these models may no longer be at the frontier of video understanding. It would be valuable to test newer and stronger MLLMs, such as Qwen3-VL or InternVL2/3, which exhibit fewer hallucinations and stronger temporal modeling. Evaluating CDantiHalP on these modern models would better validate its robustness and contributions to the field.

2.  The current pipeline heavily relies on GPT-3.5 to generate Concise-Discriminative (PromptCD) and Logic-Contradictory (PromptLC) prompts, as well as to verify hallucination conflicts.
   This dependency increases inference-time latency and cost.
   The paper could discuss whether these components can be generated or approximated by smaller, open-source LLMs (e.g., 72B or 34B in scale) to reduce overhead without degrading performance.
   Besides, the verify step may potentially be done by rule-based methods without relying on large models since it involves checking for logical contradictions. Discussing such alternatives would enhance practicality.

3.  Although the method is “training-free,” the number of GPT and MLLM calls per sample may be high. Quantitative analysis of latency, API cost, and runtime complexity may be missing.

4.  The experiments are limited to three standard action recognition datasets. Evaluating on real-world long video datasets (e.g., ActivityNet-QA, Something-Something V2) or multimodal reasoning benchmarks may further establish generality.

### Questions
1. Add experiments on modern and stronger models to strengthen the claim of general applicability.

2. Explore cost-reduction strategies using smaller open-source models for generating and verifying prompts, or precomputing a reusable prompt library to reduce the GPT query burden.

3. Provide detailed efficiency analysis by reporting average computation time and API costs for prompt generation and hallucination detection, comparing them against simpler baselines.

4. Extend evaluation beyond ZSAR by testing whether CDantiHalP generalizes to video question answering or temporal reasoning tasks, which may expand its applicability and impact.

### Soundness
3

### Presentation
3

### Contribution
3

## Human Reviewer 1

### Summary
This paper proposes to induce or enhance chain-of-thought reasoning of VLMs, with an application on Referring Expression Comprehension  (REC). The training recipe is standard practice, i.e., SFT+RL (GRPO).

### Strengths
1. Using RL for Referring Expression Comprehension is under-explored, beyond early efforts ([a-b] in Weaknesses below).
2. The improvements from SFT/RL post-training look encouraging, although not totally convincing (See Weaknesses 2 and 3).

### Weaknesses
1. Citations to previous REC + RL works are absent. For example, [a-b]. Instead, the paper only cites generic VLM works with RL, in which REC is only a subtask.
2. In Table 3, Rex-Thinker-CoT and Rex-Thinker-GRPO perform worse than QwenVL-2.5-7B (the base model of Rex-Thinker), which seems to be a sign of catastrophic forgetting due to post-training. The authors should find a way to mitigate this.
3. The main experimental results (Tables 2 and 4) are on one dataset only, the HumanRef. 
4. (Minor) "Symbiotic approach" is an unnatural framing. Such two-stage pipelines are commonly used, and people usually don't call them "symbiotic approaches".

[a] iterative shrinking for referring expression grounding using deep reinforcement learning. CVPR 2021.

[b] One for all: One-stage referring expression comprehension with dynamic reasoning. Neurocomputing 2023.

### Questions
1.  In the example in figure 3, the detector detects "person", which is straightforward. Can the model detect targets with negations, e.g. "non-persons"? It seems not obvious and may be challenging.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper studies referring expression grounding with explicit, verifiable reasoning rather than direct box regression. The task is conducted as retrieval over candidate boxes from an open-vocabulary detector, followed by a plan–act–summarize chain of thought that can also abstain when the target is absent. Training is two-stage: supervised fine-tuning on curated CoT traces, then GRPO reinforcement with an F1-based detection reward, a format reward, and KL regularization. Experiments on HumanRef and out-of-domain RefCOCOg show consistent gains in precision, recall, and F1.

### Strengths
The plan–act–summarize CoT exposes intermediate reasoning tied to concrete boxes. This improves debuggability and reduces hallucination risk. It also enables a principled “no target” refusal. 
This paper implements the SFT-then-RL framework on the REC with reasoning for MLLMs. 
The reward combines F1 for grounded detection with a lightweight format constraint. This directly optimizes what the benchmark cares about.

### Weaknesses
The paper does not report even small-scale human analysis/evaluation of the GPT-4o–generated chain-of-thought data. Quality control relies mainly on some rule-based functions like answer-conditioned prompts and automatic consistency filtering (keeping only samples whose final prediction matches ground truth). This may introduce bias in the framework and lacks inter-annotator checks as GPT-4o is not the most advanced model and the data is generated data. As a result, the reliability and transferability of these might be limited. 

In another side, the contribution is incremental relative to prior retrieval-based referring and grounded CoT work[1,2]. The methodological addition is an F1-aligned RL reward with strict IoU matching. From the experiments the performance beyond existing approaches appears small margin. 

[1] ChatRex: Taming Multimodal LLM for Joint Perception and Understanding. [2] ARGUS: Vision-Centric Reasoning with Grounded Chain-of-Thought, CVPR 2025

### Questions
It might be better for the author to conduct a human analysis/evaluation of random samples of the GPT-4o–generated chains of thought.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper introduces Rex-Thinker, a novel framework for the task of Referring Expression Comprehension (REC). Diverging from traditional methods that directly predict bounding boxes, the authors reformulate the task as an explicit and interpretable Chain-of-Thought (CoT) reasoning process. Rex-Thinker employs a symbiotic architecture that first utilizes an open-vocabulary object detector to generate candidate object proposals ("box hints"). Subsequently, a Multimodal Large Language Model (MLLM) performs step-by-step reasoning over these candidates to evaluate their alignment with the given language description.

### Strengths
1. The authors tackle the task of Grounded Object Referring from a novel perspective (Chain-of-Thought reasoning), providing a new, interpretable approach.
2. The authors have constructed a high-quality dataset that can facilitate the development of the research community.
3. The paper is well-written and clearly organized.

### Weaknesses
1. The methodology seems largely built on recent "R1-like RL" and "think-with-images" paradigm, which lacks novelty.
2. The paper lacks validation for the annotations generated by GPT-4o. Given that commercial models have been shown to have issues (e.g., hallucination), a manual review and evaluation of the annotated data is necessary to ensure its quality.
3. The paper lacks comparisons with the recent "Think-with-image" paradigm, e.g., Deepeyes, Pixel-Reasoner, and GRIT. Considering that Rex-Thinker also employs a two-stage post-training paradigm, comparing it with these methods would be crucial for better clarifying the authors' contributions.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
An overly simplified summary of the paper: the authors propose using explicit chain-of-thought (CoT) reasoning with predefined steps (plan, action, summarize) for the object-referring task (i.e., detecting the exact object(s) referred to by a natural language question). To train models to follow these steps the authors use GPT-4o to annotate an existing object-referring dataset, HumanRef (Jiang et al. (2025b)), with their reasoning traces, creating the HumanRef-CoT dataset. They then use those annotations to post-train a model (SFT for cold-start, followed by GRPO) to produce the Rex-Thinker model. The evaluation is split into in-domain (HumanRef) and out-of-domain (RefCOCOg (Mao et al., 2016)) benchmarks: Rex-Thinker outperforms other models on the in-domain dataset and is comparable to them on the out-of-domain set.

### Strengths
Overall, the paper presents an interesting idea and gives enough explanation and detail to follow it (the appendix is particularly helpful for things omitted in the main paper due to space). The evaluation covers both in- and out-of-domain datasets to demonstrate the model’s effectiveness, and the authors include additional experiments exploring different aspects of the approach.

### Weaknesses
There are, however, three main issues that justify my score. 
First, the paper doesn’t convincingly demonstrate the quality of the generated dataset, HumanRef-CoT. There is no targeted evaluation of the quality or usefulness of the reasonings added on top of HumanRef — no human evaluation or deeper analysis — which is surprising since the dataset is presented as one of the paper’s main contributions. My concern is amplified by the authors’ own note (line 239) that GPT-4o sometimes produces wrong answers, which is problematic given that the ground truth is available in the input (Figure 2). 
Second, the out-of-domain results for Rex-Thinker limit the generalizability of the proposed idea — the gains seem largely in-domain. 
Third, the evaluation could dig deeper into which instance types Rex-Thinker fails on and which it improves; for example, the “Interaction” column in Table 2 seems like a good candidate for further discussion. With the current results it’s hard to draw clear conclusions about the model’s strengths and limitations.

### Questions
For the results in Tables 2 and 3, it’s unclear how the numbers for the other models were obtained. What exact setup was used for those baselines? For example, was SFT done using ground-truth reasoning or not? Please clarify the evaluation/setup for each compared model (I may have missed this in the appendix so please point me to the correct section if that is the case).

Minor typo: line 332, “Blod” → “bold.”

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
4
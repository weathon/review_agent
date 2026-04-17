# BoundRL: Efficient Structured Text Segmentation through Reinforced Boundary Generation

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 4

## Abstract
As structured texts become increasingly complex across diverse domains -- from technical reports to generative AI prompts -- the need for text segmentation into semantically meaningful components becomes critical. Such texts often contain elements beyond plain language, including tables, code snippets, and placeholders, which conventional sentence- or paragraph-level segmentation methods cannot handle effectively. To address this challenge, we propose BoundRL, a novel and efficient approach that jointly performs token-level text segmentation and label prediction for long structured texts. Instead of generating complete contents for each segment, it generates only a sequence of starting tokens and reconstructs the complete contents by locating these tokens within the original texts, thereby reducing inference costs by orders of magnitude and minimizing hallucination. To adapt the model for the output format, BoundRL performs reinforcement learning with verifiable rewards (RLVR) with a specifically designed reward that jointly optimizes document reconstruction fidelity and semantic alignment. To mitigate entropy collapse, it further constructs intermediate candidates by systematically perturbing a fraction of generated sequences of segments to create stepping stones toward higher-quality solutions. To demonstrate BoundRL's effectiveness on particularly challenging structured texts, we focus evaluation on complex prompts used for LLM applications. Experiments show that BoundRL enables small language models (1.7B parameters) to outperform few-shot prompting of much larger models. Moreover, RLVR with our designed reward yields significant improvements over supervised fine-tuning, and incorporating intermediate candidates further improves both performance and generalization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose BoundRL, a theoretically grounded framework for efficient reinforcement learning that leverages structural bounds derived from Markov decision process (MDP) decomposability. Instead of learning value functions directly, BoundRL constrains the policy search space using upper and lower bounds on the true value function, defined by state abstraction and partitioned Bellman operators. The core theoretical claim (Theorem 4.3 and 4.5) proves that the approximation error of BoundRL is tightly bounded by the granularity of the state partition, providing guarantees on both sample efficiency and performance suboptimality. Experiments on MuJoCo and gridworld tasks show that BoundRL can outperform classical algorithms such as PPO and DQN, particularly in structured or decomposable environments.

### Strengths
Instead of regenerating every segment, the system outputs only segment-start tokens and labels, then reconstructs spans by locating those starts in the input. This cuts output from O(|d|) to roughly O(n) and lowers hallucination risk, which is a neat, practical reframing for structured documents. The paper spells out the locate-and-reconstruct process and the leftmost-occurrence rule to preserve order. 

The RLVR reward mixes a reconstruction-ratio term with exact-match F1 and a character-level F1; the final reward is r = ρ_rec × (EM + F1_char)/2, so improvements must both recover the document and align semantically. They further tackle entropy collapse by constructing intermediate candidates via single-step shorten/extend/label-perturb operations and only keep them when they strictly improve reward. Empirically, this yields consistent gains over SFT and matches or surpasses alternative RL baselines.

### Weaknesses
Most training/evaluation is on LLM prompts, with a large synthetic split (14,732 train prompts) and a relatively small real-world split (197 prompts). While results on the Langchain set are encouraging, the out-of-domain test size is modest, so claims about generality to other structured texts (e.g., legal or technical docs) remain tentative. 


  The method reconstructs segments by locating generated start-token sequences; if either boundary cannot be found, the segment is discarded. This simple policy is easy to implement but may fail under noisy formatting or repeated phrases where the “unique start” assumption breaks down. The paper itself enforces unique start sequences during SFT to improve robustness, hinting at the necessity of this constraint.


   RLVR training uses m = 4 rollouts per input at temperature 1.2 and only 25% of the training data for cost reasons, then decodes at temperature 0. These choices are reasonable, but the paper offers limited sensitivity studies (beyond a higher-temp variant) on how r’s weighting, rollout count, or selection threshold k affect stability and final quality. The selection rule admits intermediate candidates only when r_gain > 0, with model-specific k, but lacks a theoretical or principled schedule.

### Questions
1. How robust is the boundary-generation scheme when the same start token sequence appears multiple times or when whitespace/formatting gets normalized (e.g., in code blocks)? The paper enforces unique start sequences during SFT, but what happens in the wild without that guarantee, and can the locator incorporate fuzzy matching to reduce discards?

2. The selection rule accepts an intermediate candidate only if it yields a positive reward gain and up to k per batch. Could a curriculum or confidence-weighted scheme (e.g., variable k or a margin-based r_gain threshold) further reduce entropy collapse and improve generalization, especially on the small real-world split?

### Soundness
3

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
3

### Summary
The paper proposes a model for text segmentation. The approach is standard SFT+RL (GRPO), but the paper emphasizes (1) the novelty of predicting only the boundaries to reduce cost, and (2) careful reward design with data augmentation to prevent entropy collapse. The paper also introduces its own text segmentation dataset and shows that the proposed approach performs competitively.

### Strengths
- The model is simple. 
- The reward design (i.e., verifiability of reconstruction) is interesting and requires some ingenuity. Manually augmenting the search space is empirically sensible.
- The paper contributes a new dataset.

### Weaknesses
- Token-level boundary prediction with LLMs is a topic that's been explored in previous works, and not as novel as the paper makes it sound. E.g., [1] generates only the mentions in whole documents for coreference resolution, which can be viewed as a harder version of text segmentation with no label annotation. While combining this with careful RL in the specific context of text segmentation is of course novel, the paper should do a more general acknowledgement of this broad approach to this particular problem.
- It seems a bit odd that the results are only on the proposed new dataset. Shouldn't the paper include standard existing benchmarks for the task?
- It's a little hard to tell from Table 3 that the baselines represent the state-of-the-art. RL-PLUS is the only meaningful third-party baseline, and others are just in-house ablations, it seems? For someone who doesn't have a background on recent text seg research, it's not clear how strong the performance win is.

[1] Seq2seq is All You Need for Coreference Resolution (Zhang et al., 2023)

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper applies a variant of RLVR to boundary detection in language model prompts. It shows that its method of only generating tokens that indicate boundaries is faster than using a prompt-based approach.

### Strengths
In looking at text segmentation, the paper picks up a topic that nowadays receives less attention and may by many already (falsely) considered as solved. It focused on issues with blindly applying LLMs to the problem and points to an alternative that improves over that approach.

### Weaknesses
At a high level, this paper feels like an engineering project that was overly quickly turned into the format of an academic paper. I am lacking rigor in baselines and proper justification for why many of the choices that were made.  

**Why RL**

This paper presents an RLVR hammer and then turns text segmentation into a nail. The paper does not discuss why it is necessary to use RL for this approach. Even after reading it, I am left wondering if it is necessary to treat this as a text generation problem and why I should care about this approach at all. The related work at minimum should outline this argument.

**Lack of Baselines and Datasets**

The probably strongest weakness of this paper is that it evaluates prompt text segmentation via a very limited set of experiments. There are a myriad classic text segmentation benchmarks with strong baselines for scientific papers, corporate filings, emails, and many others. There is no justification given beyond "prompts are complex". 

6.2 is discussing how efficient BoundRL is but it is generating a significant 119 tokens when a classic sequence tagging approach would not have to generate any of this. While the NER baseline is discussed as scoring lower and achieving only fragmented, short segments, this is in disagreement with the entire body of literature. I strongly suspect a faulty implementation. For, example, the NER baseline uses an autoregressive model. This has historically not worked as well as dedicated sequence labeling setups. But, due to the lack of baselines, it is impossible to assess this. 

Even if no classic models are explored, I would have expected baselines based on LayoutLM and similar models aimed to improve document understanding.

**Lack of humans in the process**

The paper relies to a large degree on synthetic data. The only test on human-created prompts is via the langchain hub. But this dataset is not described or analyzed and there is no human evaluation of the results. Given that the langchain hub has templates, it will be biased in many ways. In addition, the distribution over topics is very top-heavy with a focus on certain types of agents over others.

### Questions
n/a

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a method for text segmentation of structured text that reduces the computational load on models as well as improving performance. The authors frame the problem as boundary identification and task a model to identify where various roles start and end throughout the text, assigning these segments the appropriate label. They then use these labels and segments to segment the entire text. They observe that by using this method they are able to improve the efficiency of text segmentation. They also observe that their method is able to segment text more accurately than few-shot prompted Claude models.

### Strengths
The problem is well described, and the paper is clearly written. The problem is relevant. The method improves significantly in terms of efficiency and does improve performance when compared to Claude models.

### Weaknesses
1. While the improvements show in table 3 are significant, the method improves over Claude inconsistently across metrics. It would be good to clarify why this might be and offer some explanation as to which metrics are the most important.

2. While there are lots of experimental results comparing different settings of BoundRL and different fine-tuning paradigms, and there are experimental results comparing to few-shot prompting of Claude, the performance is not compared to methods from prior work. It would be good to see how both performance and efficiency compares to these methods.

### Questions
1. Do you have any insights for the differences in metric improvement in Table 3?

2. Why do you choose few-shot prompting of Claude to compare against vs other prior work?

### Soundness
2

### Presentation
3

### Contribution
3

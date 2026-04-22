# Is On-Policy Data always the Best Choice for Direct Preference Optimization-Based LM Alignment?

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
The alignment of language models (LMs) with human preferences is critical for building reliable AI systems. The problem is typically framed as optimizing an LM policy to maximize the expected reward that reflects human preferences. Recently, Direct Preference Optimization (DPO) was proposed as an LM alignment method that directly optimizes the policy from static preference data, and further improved by incorporating on-policy sampling (i.e., preference candidates generated during the training loop) for better LM alignment. However, we show on-policy data is not always optimal, with systematic effectiveness difference emerging between static and on-policy preference candidates. For example, on-policy data can result in a  $3\times$ effectiveness compared with static data for Llama-3, and a $0.4\times$ effectiveness for Zephyr. To explain the phenomenon, we propose the alignment stage assumption, which divides the alignment process into two distinct stages: the preference injection stage, which benefits from diverse data, and the preference fine-tuning stage, which favors high-quality data. Through theoretical and empirical analysis, we characterize these stages and propose an effective algorithm to identify the boundaries between them. We perform experiments on $5$ models (Llama, Zephyr, Phi-2, Qwen, Pythia) and 
$2$ alignment methods (DPO, SLiC-HF) to show the generalizability of alignment stage assumption and boundary measurement.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates whether on-policy data is always the optimal choice for DPO-based language model alignment. The authors show that the effectiveness of on-policy data varies significantly across different model architectures and scales. To explain this, they propose an alignment stage assumption that divides the alignment process into two distinct phases: the preference injection stage, where diverse data is more beneficial, and the preference fine-tuning stage, where high-quality data is preferred. The authors further propose an algorithm to identify the boundary between these two stages.

### Strengths
- The paper provides the important observation that on-policy data can sometimes degrade alignment performance.

- The authors empirically demonstrate that the effectiveness of on-policy versus off-policy data varies across model scales and architectures, offering nuanced insights into LM alignment dynamics.

- The notion of preference injection highlights the mid-training importance of data diversity.

### Weaknesses
- Some theoretical sections (especially §5.2–§5.3) are mathematically dense and could benefit from more intuitive explanations, visualizations, or pseudocode examples.

- The boundary measurement algorithm is under-emphasized. It would be better if it were moved from the appendix to the main text.

- The paper could include more comparative experiments across different model families in the **main body** to validate the proposed assumption and algorithm further.

### Questions
- In Table 2, is the diversity of PC_on higher than that of PC_llama?

- Could you briefly summarize the purpose and contribution of Section 5.1?

- Could the boundary measurement algorithm be used to adaptively schedule on-policy and off-policy sampling in large-scale industrial alignment pipelines? I found that the computational complexity of the proposed algorithm appears relatively high.

- The paper mainly evaluates on AlpacaEval 2.0 (length-controlled) — have you tested whether similar trends hold on MT-Bench, Arena-Hard, or other evaluation suites?

- In practice, high-quality data often has low diversity. Have you explored data blending strategies to balance these two properties?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates when and why on-policy preference data (generated during training) improves or harms language model alignment compared to static preference data. While previous work such as DPO and SLiC-HF assumes that on-policy sampling always benefits alignment, the authors empirically find that this is not universally true, where performance varies systematically across models (e.g., Llama-3, Zephyr) and alignment stages. To explain these findings, they introduce the alignment stage assumption, distinguishing between a preference injection stage that benefits from data diversity and a preference fine-tuning stage that benefits from data quality. They provide both theoretical analysis and empirical evidence supporting this two-stage perspective, along with an algorithm to identify the boundary between stages. Experiments across multiple model families and alignment methods demonstrate the generality and practical usefulness of this framework.

### Strengths
1.  The writing is clear and easy to follow.

2. Research on the problem of using on-policy data for DPO is interesting.

3. The authors have great literature review for related works.

### Weaknesses
1. I have some concerns about models to measure diversity. As mentioned in the paper, the authors use Zephyr to measure diversity. To further confirm this conclusion, it's better for authors to see whether this diversity patterns are similar for different models.

2. This method can be used to recognize the training stage of LLMs. Does this method have great potential to make DPO training more efficient (like convergence speed)?

3. I find that the authors conduct some experiments on DPO and SLiC-HF. However, some of iterative DPO algorithms which use on-policy data have different loss objectives as DPO or SLiC-HF. Does this method still work on that iterative DPO algorithm like [1]?

[1] Self-play preference optimization for language model alignment.

### Questions
Please refer to weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors finds that On-policy data for alignment is not always optimal and attempts to explain it with dividing the alignment stage into a "preference injection" stage which requires data coverage (diversity) and a preference fine-tuning stages (which requires high quality data). The authors verify their findings on different models and alignment algorithms (DPO, SLiC-HF).

### Strengths
1. The paper investigates the learning dynamics of preference learning, which is understudied. The findings are novel and could be useful to practitioners who are working on human preference learning. 

2. The paper is well written with comprehensive experiments demonstrating the generalizability of their findings. The finding that off-policy DPO followed by on-policy DPO on two iterations provides a simple recipe for people to try out. 

3. The discussion between on- v.s. off-policy data could be extended to many other settings that involves Language Model post-training.

### Weaknesses
1. It seems that people are not really excited about DPO / RLHF anymore. For example, the latest open-source frontier models (Qwen3, GLM 4.5, Kimi-K2...) only adopts a RL process using rubric rewards and verifiable rewards on math / coding tasks. This is a good and interesting paper, it is just that I don't know how much impact would a DPO paper make in 2025. It would be better if the authors can show  a curriculum of on- and off-policy data also generalizes to RLVR experiments.

2. The findings are somewhat validated on, in my understanding, weak models. (e.g. Zephyr-7B and Phi-2-2.7B). I don't know if the base model is stronger (e.g. Qwen3 series) how much the findings would still hold. It seems to me that the on-policy data is not good simply because that some of the reference model were not good enough to produce reasonable outputs, so we need to reply on off-policy good demonstrations to perform alignment.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The submission proposes an qualitative breakdown of the preference optimization into the preference injection and preference fine-tuning stage, based on varied effectiveness of on- and off-policy preference data usage on different models. Authors validate the proposed breakdown on multiple models and datasets, and propose a quantitative boundary area measurement to determine the preference stage a given model and preference dataset pair is in.

### Strengths
- Taking inititative on the varied results we see in the preference optimization literature is good to see. The breakdown of the preference optimization at least makes sense qualitatively, if not theoretically
- The boundary area measurement provides a single quantitative metric that really distills the core of the paper
- The research questions are validated on multiple model families

### Weaknesses
- While the boundary area measurement is a good starting point, I still think the proposition given in the paper lacks practical application. The boundary area measurement is easy to calculate in retrospect, after we have the models and all versions of the preference data, but doesn't seem so easy when decisions need to be made on-the-fly.

### Questions
- What are some practical applications of the propositions made in the paper, when it comes to deciding between on- and off-policy data for preference optimization?

### Soundness
2

### Presentation
3

### Contribution
3

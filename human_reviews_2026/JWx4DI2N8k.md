# LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning

- Avg Score: 6.00
- Decision: Accept (Oral)
- Scores: 8, 6, 4, 6

## Abstract
Ultra-long generation by large language models (LLMs) is a widely demanded scenario, yet it remains a significant challenge due to their maximum generation length limit and overall quality degradation as sequence length increases. Previous approaches, exemplified by LongWriter, typically rely on ''teaching'', which involves supervised fine-tuning (SFT) on synthetic long-form outputs. However, this strategy heavily depends on synthetic SFT data, which is difficult and costly to construct, often lacks coherence and consistency, and tends to be overly artificial and structurally monotonous. In this work, we propose an incentivization-based approach that, starting entirely from scratch and without relying on any annotated or synthetic data, leverages reinforcement learning (RL) to foster the emergence of ultra-long, high-quality text generation capabilities in LLMs. We perform RL training starting from a base model, similar to R1-Zero, guiding it to engage in reasoning that facilitates planning and refinement during the writing process. To support this, we employ specialized reward models that steer the LLM towards improved length control, writing quality, and structural formatting. Experimental evaluations show that our LongWriter-Zero model, trained from Qwen2.5-32B, consistently outperforms traditional SFT methods on long-form writing tasks, achieving state-of-the-art results across all metrics on WritingBench and Arena-Write, and even surpassing 100B+ models such as DeepSeek R1 and Qwen3-235B.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces LongWriter-Zero by building a framework to generating ultra-long text by applying RL to a base LM. 

Different from the SFT approach from previous works, this paper proposes a RL approach with three main contributions 1) A dedicated reward model designed to balance length, writing quality and structural format. 2) The integration of a thinking step. 3) continual pre-training on writing-centric data would significantly boost the performance on top of the RL-based training. 

The resulting model proves the effectiveness of the RL framework by demonstrating to achieve SOTA performance on WritingBench and good ELO score, surpassing its SFT counterpart model and open-source/proprietary models in long text generation.

### Strengths
* The paper successfully built a pure RL-driven approach for an open-ended long-text generation task, moving beyond the prevalent paradigm of SFT on synthetic data, shed light on a topic with large headroom.
* The paper is well-written to systematically  perform ablation on the importance of continual pre-training and the "think" step, providing valuable architectural insights. 
* LongWriter-Zero demonstrates SOTA results on multiple challenging benchmarks, validating the overall effectiveness of the proposed framework.

### Weaknesses
* It's a bit unclear comparison: 
   * If the comparison is between a basic SFT and the RL GRPO setup, it's no surprise the GRPO RL setup has more advantage; 
   * If the comparison is between RL and the SOTA approaches (which often has DPO/RL steps after a basic SFT), Table 1 is the only section discussion about evidence. 

* The paper is lacking details on how to build teh preference dataset for "Writing RM", e.g. data origin, size, diversity and human rating rubrics, etc. Given the importance of reward signals in RL, the missing info makes the major weakness on reproducibility and the trust in the results. 

* The paper mentioned removal of KL penality term (beta = 0, line 138). For the open-ended question like long-form writing, it's not justified   to use this setup. The KL term is an important regularizer mitigating reward hacking and the authors mentioned the repetition-drive reward hacking as a key limitation. The decision to remove KL penalty needs more thoughts.

### Questions
Questions: 
* Figure 2 and Figure 3 shows different story for "Base-think": Figure 3 shows a high ELO at the very beginning (before 150steps) but significantly worse in "Writing RM", does that mean an misalignment between Writing RM and ELO ratings? or writing quality is less important in deciding who win?

* Figure 6 mentioned thinking token length plateaus because writing task has a ceiling. Could this plateau be an artifacts of the experiment design? For example, (a) could it be caused by the context length? (b) The format RM implicitly pensalizing overly long thinking setup? 

* Why remove KL-Penalty? details seen in the weakness section

Suggestions on Writing:
* Fig. 2 (right) needs more explanation on "Mean Non-Overlong Generation Length". It's hard to understand. And Green curve has different behavior than the other two methods. Need more explanation. 
* Use "Direct-Answer" and "NoThink" in different places but presenting the same meaning, a bit confusing.
* nit: "-Continual Pretrain" in Table 1 may be misunderstood as dash line rather than "minus". Same for "-Thinking".

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper describes a new approach to generating quality long-form text, LongWriter-Zero, aiming to improve over standard LLM generation baselines and the small number of works that have explicitly tackled longform generation (LongWriter, Suri).  The approach uses three components, framing their use using three research questions (RQs): RL with appropriate reward model aspects (length, quality, structural format and redundancy), test-time scaling, and continual pretraining.  There is an evaluation on three benchmarks (the existing WritingBench, Arena-Write defined in this paper, and a human-in-the-loop win-rate evaluation), as well as more detailed investigations like e.g. assessments of the importance of the three core components of LongWriter-Zero.

### Strengths
* The three components of LongWriter-Zero are intuitive, even if not necessarily obvious, contributing some originality.  I thought the argument for RQ2, on the use of test-time scaling – that it had mostly been used for e.g. code and math problems, and that its use was less obvious for long-form generation – was sensible and interesting, and worth investigating.

* The model seems to perform strongly, even against much larger models.  Having three different types of evaluation was also a strength.  Ablation analyses were also informative.

* This is an important aspect of LLMs to look at, and there isn't too much work in this space yet, making this a useful contribution.

### Weaknesses
* The evaluation was rather different from the LongWriter paper by Bai et al, which is really the primary baseline, as framed by e.g. the paper abstract and the related work.  LongBench-Write isn’t mentioned at all in the present paper.  While WritingBench does in some ways supersede LongBench-Write (with that paper discussing the relationship of the new benchmark to the prior LongBench-Write), I still would have expected to see an evaluation on LongBench-Write here for comparability.

* Relatedly, the critic used for the first evaluation probably warrants some discussion.  It of course makes sense to use the critic that comes with WritingBench, but there’s no commentary on or investigation of possible bias in the evaluation of Qwen models wrt non-Qwen models by a Qwen critic.  (Bai et al noted this for their GPT-4o judge for LongWriter.)  This also factors into the results in Table 1, where the DeepSeek-R1 is probably close enough that I wonder whether the improvement is real or a result of bias.

* Looking elsewhere in Table 1 results, the commentary in Sec 3.2 just observes that LongWriter-Zero outperforms all models, but it only outperforms Qwen3-235B-A22B by a quite small 0.01 overall.  (I’d also note that the Qwen3-235B-A22B score for EN should be bolded as equal highest given the one decimal place result.)  While this is still noteworthy, given that LongWriter-Zero is a much smaller model, I think it should be remarked upon.

* Relatedly, do you have any thoughts about why LongWriter-Zero and Qwen3-235B-A22B are so close in the Table 1 results by the critic but there is a much more substantive difference in Elo ratings (discussed in Sec 3.2)?

### Questions
Please see above.

### Soundness
3

### Presentation
4

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
While ultra-long generation is a desired capability in large language models, it often requires fine-tuning on SFT data with expensive data which may also lack quality. This paper instead motivates and introduces LongWriter Zero, an RL paradigm to transform a base language model into one that can generate long and high-quality text - without written data. They provide intuitive reward design, an introduce a new benchmark to evaluate different LLMs against each other (Arena-Write). They also perform helpful ablations demonstrating the efficacy in the continual pre-training case.

### Strengths
* The paper is written well in general
* This solves a key, timely problem: how do we encourage more length without access to expensive SFT data? 
* There are comprehensive details of prompts and experimental design which seem well-justified, including a thorough appendix.
* The reward design is justified and includes multiple components
* The proposed model LongWriter-Zero (32B scale) outperforms a variety of methods across a few benchmarks

### Weaknesses
* The results are shown on only one model trained with the introduced paradigm. It would be nice to test if this holds across other models, namely 1) models not Qwen and 2) perhaps a smaller size Qwen (ie, does this still work at smaller scale)
* While the approach is nice, replacing costly SFT data, the process intuitively seems prone to length reward hacking, since the reward is based on the averaged advantage over length equally with format and writing quality, a repetitive string may overwhelm the rest of the rewards by being extremely long, leading to reward hacking with long and non-fluent texts. This may be solved by weighting the other factors more, or in a different way.
* Related to the above, it is unclear how robust the writing reward model, the proxy for quality, performs in complex and long scenarios. Though it is mentioned that it "capture holistic writing quality, including fluency, coherence, and helpfulness", there are other more complex factors like factuality which are hard to control for.
* There should be more baselines. While it is nice to compare to existing other models, there needs to be more comparison of this length method to other approaches, such as those that leverage synthetic data, etc (ie, LongWriter on 32B scale)
* Minor: ArenaWrite seems small - only 100 examples
* There are a couple formatting errors, for example, a citation link on the bottom of Page 6 spanning over the page break
* Minor: Sections RQ1, RQ2, and RQ3 seem a bit disconnected, could be transitioned better.

### Questions
* Why did you need to create and evalute with Arena-Write, ie, what is wrong with existing benchmarks for this?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces LongWriter-Zero, an RL framework using GRPO to train LLMs for ultra-long text generation from scratch, bypassing synthetic SFT data. It uses composite rewards for length control, writing quality, and format, plus <think> prompts for reasoning and continual pre-training on writing data. The authors test the approach on WritingBench, Arena-Write, and human win-rate evals. Results achieve SOTA: 8.69/10 average on WritingBench, 1447 Elo on Arena-Write, and win rates >62% vs. 100B+ models like DeepSeek-R1.

Strengths:
- The pure RL approach fosters emergent long-form capabilities without data quality biases from synthetic SFT, enabling innovation beyond teacher model limits.

Weaknesses:
- Using heuristic rewards naturally entails reward hacking. The effectiveness of the proposed method is based on how well the heuristics of the reward models align with the actual metrics that they care about. The paper does not show that LongWriter-Zero is immune from reward hacking. This suggests that the generalization of the approach might be limited.
- Composite reward balancing risks suboptimal trade-offs, where normalizing advantages could dilute focus on critical aspects like coherence in ultra-long sequences.

### Strengths
- The pure RL approach fosters emergent long-form capabilities without data quality biases from synthetic SFT, enabling innovation beyond teacher model limits.

### Weaknesses
- Using heuristic rewards naturally entails reward hacking. The effectiveness of the proposed method is based on how well the heuristics of the reward models align with the actual metrics that they care about. The paper does not show that LongWriter-Zero is immune from reward hacking. This suggests that the generalization of the approach might be limited.
- Composite reward balancing risks suboptimal trade-offs, where normalizing advantages could dilute focus on critical aspects like coherence in ultra-long sequences.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

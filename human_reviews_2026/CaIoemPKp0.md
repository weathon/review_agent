# VisualThinker: First ever R1-Zero's Aha Moment on just a 2B non-SFT Model

- Avg Score: 3.60
- Decision: Reject
- Scores: 8, 4, 2, 2, 2

## Abstract
Recently, DeepSeek-R1 demonstrated how reinforcement learning with simple rule-based reward can enable autonomous development of complex reasoning in large language models, characterized by the "aha moment", in which the model manifest self-reflection and increased response length during training. However, attempts to extend this success to multimodal reasoning often failed to reproduce these key characteristics. In this paper, we present the first successful replication of these emergent characteristics for multimodal reasoning on only a non-SFT 2B model. Starting with Qwen2-VL-2B and applying reinforcement learning directly on the SAT dataset, our model achieves 59.47% accuracy on CVBench, outperforming the base model by approximately ~30% and exceeding SFT setting by ~2%. By further incorporating a small amount of cold-start data, we achieved 70.58% accuracy on CVBench, a performance surpass GPT-4o-mini. In addition, we observed that applying RL to instruct models often leads to trivial and low-diversity reasoning trajectories and presents our insights and attempts to understand and mitigate this issue.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Following the success of DeepSeek-R1 in natural language reasoning, several studies have attempted to extend its reinforcement learning–based reasoning paradigm to multi-modal domains. However, prior efforts have struggled to reproduce the distinctive “aha moment” behavior, where multi-step reasoning leads to sudden insight and problem resolution. This paper is the first to successfully elicit such “aha moment” in vision-language reasoning, resulting in a substantial performance improvement over existing open-source multi-modal reasoning models. Remarkably, the proposed 2B-parameter vision-language model (VLM) achieves performance comparable to proprietary models such as GPT-4o, despite being trained solely with reinforcement learning (RL) without supervised fine-tuning. The paper further provides a comprehensive analysis of RL training behaviors in both the base and instruction-tuned versions of the multi-modal model, offering valuable insights into how reinforcement learning shapes reasoning ability across modalities.

### Strengths
* The paper presents an interesting and scientifically meaningful finding for advancing multi-modal reasoning. Notably, the observation that reinforcement learning (RL) on a base multi-modal model (rather than an SFT model) yields better performance, and can elicit an “aha moment” phenomenon, is both novel and thought-provoking.
* The paper is well-written, clearly structured, and accessible, making it understandable even to readers who are not experts in multi-modal reasoning.
* The authors perform diverse and insightful analyses to support their claims, including extensions to instruction-tuned models (and discussions of their failures), quantitative evaluation of “aha moments,” and multiple ablation studies that enhance the empirical depth of the work.

### Weaknesses
* The evaluation is somewhat less rigorous. The proposed method is tested only on a single, relatively small 2B-parameter vision-language model, and lacks comparison against stronger open-source non-reasoning VLMs.
* The “aha moment” is a qualitative byproduct of reasoning rather than as a mechanism linked to predictive performance. A more rigorous analysis should clarify whether this phenomenon directly contributes to improved task accuracy.
* The technical novelty appears incremental, as the main methodological difference from DeepSeek-R1 lies in applying RL to a non-SFT model. No new techniques are introduced that specifically address visual processing or visual reasoning challenges.

### Questions
1. In Figure 4, the instruct model’s improved performance on multi-modal benchmarks suggests that RL training also benefits the instruction-tuned variant. However, this might imply that the benchmarks themselves are not complex enough to necessitate multi-step reasoning. Do the RL-trained instruct models also show inferior performance on all three vision-centric benchmarks, as compared to the base model?
2. Are the vision-centric benchmarks employed in the paper sufficiently complex to demand genuine reasoning? Although the paper describes them as involving spatial relationships, object counting, depth ordering, and relative distance, it is unclear whether these tasks truly require multi-step reasoning. In contrast, natural-language reasoning tasks (e.g., mathematical or logical inference) inherently involve sequential reasoning steps, while many of the visual tasks might be solvable using simpler perception-based models such as depth estimation or semantic segmentation.
3. Could there be techniques specific to visual perception that would enhance the model’s ability to learn visual reasoning patterns? In its current form, the approach appears to be a direct adaptation of DeepSeek-R1 to vision-language settings without introducing mechanisms uniquely suited to visual reasoning.

### Soundness
3

### Presentation
4

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
# Problem:

How to reproduce the characteristics of DeepSeek-R1’s ‘aha moment’, towit increased response length and self-reflection-evidencing intermediate token generation during RLVR training, in multimodal reasoning context?

# Contributions:

This paper proposes a replication recipe of the ‘aha moment’ for multimodal reasoning on only a non-SFT 2B-parameters model. This recipe consists of a cold-start fine-tuning on curated reasoning trajectories before applying RLVR, in order to alleviate on RL instabilities.

Experimental evidences use a SAT dataset for training and show transfer-learning improvements on CVBench, to the extent of surpassing GPT-4o-mini’s performance.

This paper also provides insights about applying RLVR to instruct-fine-tuned models, which reveals trivial and low-diversity reasoning trajectories.

### Strengths
# Strengths:

## Quality:

SQ1: I appreciate the insights from Table 1 and related text.

SQ2: The ablation study on the effect of temperature is a really insightful investigation. I would welcome the authors to add more along that line, if space allows.

## Clarity:

SC1: I appreciate the background information (e.g. Table 1, RL algorithm) that helps make the reading experience self-contained.

## Originality:

cf. weaknesses for trade-offs discussions…

## Significance:

cf. weaknesses for trade-offs discussions…

### Weaknesses
# Weaknesses:

## Quality:

WQ1: While Figure 3’s experiment possibly contains some confounders that could be worth discussing, I think the main issue with this Figure and related analysis is the lack of statistical information: is it the mean that is represented, or the median? Over how many samples per points? Are the samples all the same prompt but with different random seeds or is it over different prompts but with the same random seed (I suspect the latter)?

I think this experiment has a great potential, especially if some other self-reflection reasoning-indicative keywords could be added (e.g. ‘but’, ‘however’, ‘therefore’, …), but the current state of it falls short of being robust or statistically trustworthy. I hope the authors can address this in there revisions.

WQ2: The current paper pushes a narrative on a recipe to get non-SFT VL models to acquire self-reflection-typed reasoning skills, but it only experiments with one single base model. I think it would strengthen the narrative and provide more evidence to the claim if the authors could show that their proposed recipe also works with one or two other non-SFT base models.

## Clarity:

WC1: I think it would ease the reading experience further if some visual examples from the datasets and benchmarks used could be presented in the main paper or appendix, in order for the reader to get a better sense of what kind of transfer learning is obtained from SAT to CVBench, especially given the visual modality being of interest.

## Originality:

WO1: Using Table 1, it seems that the only differences between the current paper and R1-Multimodal-Journey are the facts that the latter (i) did not elicit increasing response length dynamics and (ii) started from an Instruct-fine-tuned model as base model. Regarding (ii), the fact that the current paper shows how to get a non-SFT 2B VL model to acquire self-reflection-typed reasoning is important to the community. However, I fail to see value for the community in (i). Thus, it might be worth for the narrative of the paper to reduce focus on (i) and maybe increase focus on the non-SFT aspects. I would advise the authors to start by adding an extra row to Table 1 that specifies whether fine-tuning is considered or not, to highlight more explicitly that aspect.

WO2: Following WQ2: I think the paper would have a greater impact if it could improve its external validity by, for instance, showing results on applying the recipe to another non-SFT model, possibly of a different family, or with different number of parameters.

## Significance:

WS1: None of the results in the paper report statistics that could enable statistical significance evaluation. I would advise the authors to perform experiments on randomised seeds (>5) and report error of the mean  as much as possible, starting with Figure 2 and 3 (e.g. mean=line+std.error=shaded area) and Table 2 (even if past papers failed to provide such statistics - it is never too late to make it right).

Indeed, as it stands, it is unclear whether, for instance, the current results are not the consequence of a lucky, unintentional cherry-picking of the right (default) random seed.

WS2: Moreover, running some statistical significance test on e.g. Table 3 and Table 4 would strengthen the claims made.

### Questions
cf. weaknesses above.

### Soundness
2

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
3

### Summary
The paper proposes VisualThinker, a GRPO-based RL recipe to elicit R1-style emergent reasoning in a 2B, non-SFT multimodal model. The authors train on the static subset of SAT with a rule-based reward , reporting two training dynamics that co-evolve with accuracy, the “insight moment” and increasing reaction length, and achieving superior results.

### Strengths
1. Demonstrates the emergence of “aha moment” behaviors and longer reasoning chains in a multimodal RL setup, which has rarely been shown before.
2. Achieves consistent performance improvements on spatial reasoning benchmarks, suggesting the method is empirically effective and reproducible.

### Weaknesses
1. The paper treats the “aha moment” and response length increase as key indicators or prerequisites of valid R1 replication, yet seems provides no causal proof that these phenomena are necessary or sufficient for performance gains. Prior works (e.g., DeepSeek-R1, Nature 2025; OpenAI o1 tech report) describe “aha” and longer reasoning as correlated trends, not as defining properties. The current framing seems overgeneralizes these correlations without referencing evidence.
2. The “aha moment” is measured only via frequency of tokens like wait/again and average response length, which are more like style-dependent. I have concerns about their validity as genuine signals of reasoning.
3. The training pipeline is largely a direct application of GRPO with rule-based rewards; the only new ingredient is tracking emergent phenomena. I find it difficult to interpret this as methodological innovation.
4. The experiment primarily focuses on three benchmarks—CVBench, BLINK, and VSR—all emphasizing spatial reasoning capabilities. It is recommended to incorporate textual-visual QA or out-of-distribution tasks to demonstrate broader generalization abilities.
5. The paper shows successful cases but no independent assessment is conducted on the “quality of the reasoning process.” For example, are all long responses logically coherent? Are there instances where the “correct answer is paired with flawed reasoning”?

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents VisualThinker, which the authors claim to be the first successful replication of “aha moment” in the context of multimodal reasoning. By leveraging GRPO, VisualThinker achieves remarkable results in visual reasoning, starting from Qwen2-VL-2B as the base model. The ablation studies further show the relation between SFT and RL for multimodal reasoning.

### Strengths
1.	This paper is well-structured, well-written and easy to follow.
2.	This paper demonstrates the effectives of GRPO in visual reasoning, with approximate 30% improvement over the base model.
3.	The discussion section provides food for thought, although some of them are not that novel in the context of visual reasoning.

### Weaknesses
1.	The authors claim that VisualThinker is the first successful replication of “aha moment” in the context of multimodal reasoning. However, previous works such as [1] have already reported similar reproductions.
2.	The conclusion (e.g. freezing vision encoder, instruct models) are not well-supported, as VisualThinker only uses Qwen2-VL- 2B as the base model. A larger or newer base model is encouraged to be experimented with, and the difference between them will be an interesting topic to be discovered.
3.	The related work and references are outdated. More recent works are encouraged to be included and discussed.
4.	Similarly, it is suggested that the authors compare their approach with more recent baselines.  

[1] Huang, Wenxuan, et al. "Vision-r1: Incentivizing reasoning capability in multimodal large language models." arXiv preprint arXiv:2503.06749 (2025).

### Questions
Please refer to the **Weakness** part. I am looking forward to the authors’ response, and may reconsider this paper based on that.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Simple rule-based rewards have been found to be sufficient to drive complex reasoning capabilities, sometimes referred to as an “aha moment.” In this work, the authors show that for a smaller (2B) vision-language model, it is also possible to elicit this capability. The authors demonstrate this by applying GRPO with a rule-based reward function and use curated examples for initialization over a non-SFT (supervised finetuning) model. They find that on various Vision-language benchmarks (CVBench, BLINK, VSR), the RL approach greatly outperforms SFT, claiming to be the first successful replication of an “aha moment” for a multimodal (VL) model.

### Strengths
* Empirically, VisualThinker R1(-Zero) is a best-in-class model on the visual reasoning benchmark, significantly outperforming even much larger/closed models. This model itself is a significant artifact of contribution.
* Additional analysis on why it is better to RL over non-instruct models and fair comparisons/ablations to resolve questions around instruct model (including negative results), coldstart data, and sampling temperature.

### Weaknesses
* There are several confusing parts around the presentation of the work, not just in grammar/typos or writing (like overclaiming – VL reasoning exists but in much larger models, so claiming “first” is conditioned on model size) but also in scientific hypothesis and experiment structure (see questions).
* Methodologically, there is relatively little novelty in method or experiment design. The main observation is that it is better to RL over non-instruct base models for VL than instruct base models (at the 2B size), but otherwise the recipes for training all the models are standard.
* Ultimately, one nice conclusion I wish we could make is that “if you want VL reasoning, RL the non-instruct model instead of the instruct model." However, we cannot actually make that conclusion because we only have evidence of one model (Qwen VL 2B), and it may not generalize more broadly. Rather than the discussion on length rewards, cold-start, or temperature ablations, it would be more interesting to see if applied to other models of this size (or slightly larger/smaller), whether it is better to RL on SFT or pre-SFT.

### Questions
* The main issue I have is that “Aha moment” is never formally defined so it was confusing to follow. Is it purely related to response length or a separate dimension to it? Or related to accuracy? The caption of Figure 1 says “emergence of self-reflection,” how exactly is that measured? It isn’t as important in the DeepSeek R1 paper because they were describing an observation. This work aims to show that such observation exists, which requires some sort of definition.
* The main result is slightly counterintuitive, as most GRPO/SFT is performed on top of instruct models. Why is VL at 2B different from other datasets or model sizes?
* For figure 3: what does this graph look like for the other models/baselines? Since you can’t get the mid-training checkpoints for public models, what is the frequency at the end for those?
* Figure 4: which graph shows “freezing … vision component”?

### Soundness
2

### Presentation
2

### Contribution
2

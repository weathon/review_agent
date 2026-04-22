# ExPO-HM: Learning to Explain-then-Detect for Hateful Meme Detection

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 6, 8, 6

## Abstract
Hateful memes have emerged as a particularly challenging form of online abuse, motivating the development of automated detection systems. Most prior approaches rely on direct detection, producing only binary predictions. Such models fail to provide the context and explanations that real-world moderation requires. Recent Explain-then-Detect approaches, using Chain-of-Thought prompting or LMM agents, perform worse than simple SFT baselines, and even advanced post-training methods such as GRPO fail to close the gap. Our analysis identifies two key issues of such systems: important policy-relevant cues such as targets and attack types are not hypothesized by the model as a likely explanation; and the binary reward signal is insufficient to guide reasoning. To address these challenges, we propose ExPO-HM (Explain-then-Detect Policy Optimization for Hateful Memes), inspired by the training and evaluation process of human annotators. ExPO-HM combines SFT warmup, GRPO with curriculum learning, and Conditional Decision Entropy (CDE) as both metric and reward for reasoning quality. Across three hateful meme benchmarks, ExPO-HM achieves state-of-the-art performance on binary detection, fine-grained classification, and reasoning quality, with up to 15\% and 17\% F1 improvement over the GRPO and DPO baselines, respectively. By moving hateful meme detection from simple binary alarms to explanation-driven detection, ExPO-HM provides accurate, interpretable, and actionable moderation support. Code available at: https://github.com/JingbiaoMei/ExPO-HM

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes ExPO-HM, an explain-then-detect framework for hateful meme detection. It addresses two challenges in prior explain-then-detect systems: missing policy-relevant cues in explanations and weak binary rewards that do not guide reasoning. Among three datasets (HatefulMemes, MAMI, PrideMM), ExPO-HM reports state-of-the-art performance on classification.

### Strengths
1. The change from direct detection to explain-then-detect in hateful memes is valuable for real-world moderation: explanations provide context, policy linkage, and operational guidance.
2. Extensive experiments are conducted on three public datasets to demonstrate its performance.

### Weaknesses
1. The proposed method shows heavy reliance on LLM-as-judge for rationale quality. It's reasonable to introduce different LLMs to evaluate the effectiveness of the proposed three components. Do we need to make new policies for different platforms?
2. Failure cases are not reported. 
3. The CDE reward depends on the model’s own decision distribution; while penalizing confident errors helps, more experiments on calibration and robustness would strengthen confidence.
4. Limited discussion on explanation faithfulness. Does the rationale actually reflect features used for the decision?

### Questions
Do you have any human evaluations of rationale faithfulness beyond LLM-as-judge?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces EXPO-HM, a method for fine-tuning multimodal local LMs (LMMs) to classify meme images as hateful or not in an “explain-then-predict“ manner, producing both an explanation and a prediction. The method involves two training stages: (1) a warmup SFT stage (SFT-PM) which tunes the LMM to predict the fine-grained image class using a set moderation guideline as additional; (2) a GRPO with curriculum learning (GRPO-CL) stage using a reward function that punishes (A) departure from an explain-then-predict output format and (B) inaccuracy with respect to the ground truth label. The “curriculum“ aspect pertains to the label used--the GRPO-CL process starts out using fine-grained ground truth labels before transitioning to a mix of fine-grained and binary labels. Finally, a variant of GRPO-CL is explored using conditional decision entropy (CDE) CDE as an additional reward signal, which measures the entropy of the label decision conditioned on the produced explanation.

The proposed method is compared to a number of baselines across three hateful meme datasets, including non-explain-then-predict baselines, previously published models, and explain-then-predict models using SFT, DPO, and standard GRPO. EXPO-HM is found to consistently outperform all baselines and prior models on all three datasets. API based models are not compared due to a high rate of rejection of the hateful meme inputs. An ablation study is performed to assess the marginal impact of each part of the training (SFT-PM, GRPO-CL, and the CDE reward component), which finds that all three provide marginal utility. An additional analysis finds that SFT-PM the most effective of a number of possible SFT warmup stages.

### Strengths
- **Important task:** The paper covers an important and not particularly solved task, that of hateful meme detection.
    
- **Comprehensive baselines:** The paper uses a very comprehensive set of baselines, including both prior work, direct detection baselines, and explain-then-detect baselines using a variety of levels of training including SFT alone, DPO, and standard GRPO. The paper provides a reasonable justification for not including API-based models such as the OpenAI o-series. I can’t think of anything that is missing.
    
- **Good results:** The proposed method outperforms all baselines and prior models. The improvement is a little incremental over the previous state of the art (+~1% F1), but it is consistent across all three datasets and it uses a very different method (no retrieval augmentation).
    
- **Good ablation studies:** The ablation studies do a nice job of validating the individual pieces of the EXPO-HM method, which might otherwise seem a little arbitrary and over-engineered.

### Weaknesses
**Minor** **spelling mistakes:** “taionale”, “subjectuive“ line186

### Questions
No particular questions

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
This study propose a method, which is referred to as Explain-then-Detect Policy Optimization, for hateful meme classification and explanation generation, which is mainly utilising the idea of how manual annotation processes are conducted. It demonstrate that combining SFT-warmup, GRPO with curriculum learning and conditional decision entropy improves the performance for both detection and reasoning.

### Strengths
- Focused on the limitation of current CoT based approaches, which underperforms SFT based model. 
- The proposed approach inspired by how "human annotators are trained for the annotation" is well motivated. 
- Experimental results on different meme datasets to show the effectiveness of the proposed approach.
- Task addressed - binary, fine-grained and reasoning
- Ablation studies highlights the importances of the different components of the proposed  approach.
- To ensure reproducibility authors plans to release augmented data and code upon publication.

### Weaknesses
- While LLM-as-a judge for explanation/reasoning evaluation is becoming widely used approach, however, they are limited in many setups. Related to this, prompt plays a role for judge and reproducibility if often challenging. 
-  Hyperparameters (e.g., CDE) can be sensitive to the dataset, how are they set for different datasets?

Typos/Grammatical issues
- L130: Specify what CXHXW refers to
- L187: scarce taionale corpora
- 213: "study policy guidelines" -> study annotation guidelines?
- 236: "reward In" please check, there might be fullstop or colon.

### Questions
- Is it possible to provide details fine-grained results for attack types and targets? 
- How exactly were the policy manuals built from the annotation guideline? For examples there is not detailed annotation guideline for hateful memes.

### Soundness
3

### Presentation
3

### Contribution
3

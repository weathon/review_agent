# Training Language Models to Self-Correct via Reinforcement Learning

- Avg Score: 8.00
- Decision: Accept (Oral)
- Scores: 8, 8, 8, 8

## Abstract
Self-correction is a highly desirable capability of large language models (LLMs), yet it has consistently been found to be largely ineffective in modern LLMs.  Current methods for training self-correction typically depend on either multiple models, a more advanced model, or additional forms of supervision. To address these shortcomings, we develop a multi-turn online reinforcement learning (RL) approach, SCoRe, that significantly improves an LLM's self-correction ability using entirely self-generated data. To build SCoRe, we first show that variants of supervised fine-tuning (SFT) on offline model-generated correction traces are insufficient for instilling self-correction behavior. In particular, we observe that training via SFT either suffers from a distribution mismatch between the training data and the model's own responses or implicitly prefers only a certain mode of correction behavior that is often not effective at test time. SCoRe addresses these challenges by training under the model's own distribution of self-generated correction traces and using appropriate regularization to steer the learning process into learning a self-correction strategy that is effective at test time as opposed to simply fitting high-reward responses for a given prompt. This regularization prescribes running a first phase of RL on a base model to generate a policy initialization that is less susceptible to collapse  and then using a reward bonus to amplify self-correction during training.  When applied to Gemini 1.0 Pro and 1.5 Flash models, we find that SCoRe achieves state-of-the-art self-correction performance, improving the base models' self-correction by 15.6% and 9.1% respectively on the MATH and HumanEval benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces ScoRe, a multi-turn online reinforcement learning (RL) approach aimed at enabling self-correction in model responses. Through this method, the model learns to identify and correct its own responses, improving overall performance. The authors first analyze existing methods, identifying two main factors that currently limit self-correction capabilities. Based on this analysis, the proposed method employs a two-stage online RL training process with specific optimization objectives to reduce distribution shift and behavior collapse. As a result, the method outperforms other baselines.

### Strengths
1. The self-correction setting in this paper is more realistic, increasing its applicability to real-world scenarios.
2. The paper provides an insightful analysis of prior self-correction methods, identifying two key factors that limit effectiveness: distribution shift and behavior collapse. This analysis offers valuable insights that can inspire future research.
3. ScoRe demonstrates superior performance compared to baseline methods. Additionally, all experiments were conducted on Gemini, a robust baseline model, which further validates the approach.

### Weaknesses
1. In Stage 2, the authors use reward shaping to prevent the model from collapsing to a non-self-correcting solution. The chosen hyperparameter $\alpha$ , $\alpha >1$, could potentially result in the overall sign of $r(y1,y*)$ being negative, which may lead the model to introduce minor errors in the first step to leave room for self-correction in subsequent steps.
2. I believe the primary purpose of using self-correction is to achieve better performance. To support this goal, more baseline comparisons should be introduced. For instance, self-improvement methods like REST-EM, which involve only a single round of generation without self-correction during inference, rely solely on the model itself and should be comparable.

### Questions
Please refer to the weaknesses part.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors present a technique, ScoRe, for teaching LLMs to self-correct. The technique uses two RL training phases: in the first, the model is trained to give correct second-turn answers without giving different first-turn answers than the base model; in the second, the model is trained on a reward signal that incentivizes correct answers in both turns as well as improvement from the first turn to the second. SCoRe is motivated via a fine-grained analysis of the failures modes of prior SFT-based techniques for teaching self-correction.

### Strengths
1. The paper's analysis of the failure modes for prior SFT-based methods is very insightful, with the authors making use of edit-distance-based metric and an analysis of train-test differences to understand why prior methods fail to learn self-correction or fail to generalize out-of-distribution.
2. The results appear relatively strong, with SCoRe substantially outperforming prior methods on the evaluations presented. 
3. The presentation is overall relatively clear.

### Weaknesses
1. Certain choices in the technique don't appear to be "as simple as possible," and the text doesn't consistently do a good job of motivating these choices. (See questions.)
2. I would like to see these results compared to the very simple baseline of RL directly against the final answer, but with the self-correction prompt inserted after the first turn.

### Questions
1. As I understand things, the goal in phase I is to teach the model to self-correct given answers from the base model. The natural way to do this would be to input first turns sampled from the base model and RL the model to give accurate second turns. Instead, this paper has the model generate both turns, with an RL training signal that rewards the model for high second-turn accuracy and with a large KL penalty against the base model for the first turn. This seems quite overcomplicated—am I misunderstanding something?
2. In phase 2, if I am understanding correctly, the reward is {first-turn correctness} + {second-turn correcteness} - {KL against the base model} + alpha * {second-turn reward - first turn reward}  where alpha > 1. If so, then this effectively gives a large reward for second-turn correctness while actively penalizing first-term correctness. Is this intended? If so, why should this be better than just directly training on second-turn directness only?
3. The authors claim that a policy which tries to give its best-guess attempt in turn 1 followed by no self-correction should *generalize worse to new problems* than self-correction policies, but don't substantiate this claim with theoretical arguments or empirical findings. Why should this be true?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces SCoRe, a novel multi-turn RL method to enhance the self-correction ability of LLMs. The SCoRe improves LLMs' performance in correcting their mistakes without needing extra external supervision. Compared to supervised fine-tuning (SFT), which struggles with distribution shift and behavior collapse, SCoRe utilizes multi-turn RL with regularization strategies, achieving good accuracy gains on MATH and HumanEval benchmarks.

### Strengths
1.  The work identifies and studies two limitations of existing self-correction methods: distribution shift and behavior collapse.

2. The work proposes a novel and original multi-turn RL method.  The method's significance lies in its potential to address key limitations of existing approaches.

3. The quality of empirical analysis is good by showing improvements in self-correction metrics on established datasets and ablation studies on various components of the proposed method.

4. The work is presented with commendable clarity, including detailed explanations of the algorithm and experimental setup, making it accessible to readers.

### Weaknesses
1. No experiments are conducted with open-source models such as the Llama series.

2. The models are trained for only two attempts, leaving the scalability of the proposed method to additional attempts uncertain.

### Questions
For the MATH benchmark, why is a portion of the test data used for training? Could this make the evaluation less comprehensive?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents SCoRe, a novel reinforcement learning (RL) approach to enhance self-correction in large language models (LLMs). Unlike previous methods that rely on multiple models or external feedback, SCoRe employs a multi-turn online RL mechanism using self-generated data. This two-stage process begins with training on a base model's correction traces to avoid behavior collapse, followed by multi-turn RL with reward shaping to promote effective self-correction. SCoRe is the first approach to attain positive self-correction result, surpassing traditional fine-tuning and prompting-based methods on math and code benchmarks.

### Strengths
1. First approach for making self-correction really work.
2. Very solid experiments and ablation studies along with in-depth analysis providing insights for achieving inference time scaling like OpenAI's o1 series.

### Weaknesses
1. This work conduct experiments on private Gemini series which is hard to reproduce, it would be beneficial to include experiments on open-source models (Llama 3).
2. This work only explore 3 datasets (HumanEval, MBPP and MATH) on code and math. It would be better to introduce more datasets of varying difficulty levels (e.g. AIME).
3. Also, it would be better to conduct experiments on a broader range of diverse subject (e.g. Physics, Chemistry).

### Questions
1. Sequential self-correct introduces a dependency on previous answers, so it would be best to compare its real inference time with parallel attempts.

### Soundness
4

### Presentation
4

### Contribution
4

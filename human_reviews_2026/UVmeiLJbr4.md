# Beneath the Surface: Exposing and Mitigating Surface Learning in Large Language Models

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
As Large Language Models (LLMs) continue to evolve, assessing their genuine comprehension of underlying knowledge is crucial to ensure the reliability in real-world applications. To evaluate what LLMs learn, we first introduce ME-Test suite, including Mathematical and English grammar examinations, where each question is equipped with relevant knowledge to guide the model. Building upon this, we construct a sequence of questions with increasing difficulty based on Cognitive Load theory, enabling the model to perform continuous problem-solving using the dialogue history. Through a comprehensive evaluation, we uncover a phenomenon of **Surface Learning** behavior on LLMs similar to student learning behavior in Education Psychology. The behavior indicates that although the models seem to know the formulas and strategies required to solve specific types of problems on the surface, they do not truly comprehend the essence of these concepts, resulting in surface-level short-term benefits rather than in-depth learning. Further to mitigate surface learning behavior of LLMs, we propose a long-term strategy for both training-free and post-training scenarios. In training-free scenario, inspired by Self-Concept theory, LLMs are prompted with goal-setting and planning beforehand as well as feedback afterward to improve the ability in reasoning process. To better activate the underlying knowledge during the post-training process, we propose behavior correction strategy to re-rank samples based on the designed self-cognition indicators of LLMs. This strategy prevents models from relying on easy-to-find paradigms to maximize rewards or minimize losses in the initial training stage, rather than undertaking actual reasoning. Extensive experiments of Supervised and Reinforcement Fine-Tuning (SFT, RFT) conducted on LLMs demonstrate the effectiveness of the strategy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates surface learning behavior in Large Language Models (LLMs). The authors design clever experiments by providing correct/incorrect formulas or concepts alongside mathematical reasoning and English grammar questions. They identify three types of surface learning: (1) rote learning—models memorize formulas without understanding the underlying concepts, (2) ignoring background knowledge—providing relevant or irrelevant formulas doesn't necessarily improve or harm performance, and (3) focusing on answer patterns—models learn spurious correlations between concepts and solutions. To address this, the authors propose a long-term strategy including self-concept planning (training-free) and behavior correction (post-training).

### Strengths
1. Experimental design: The idea of testing model understanding by providing correct/incorrect formulas is innovative and insightful. This controlled setup effectively reveals whether models utilize the formulas or concept.
2. Well-written with good motivation: The paper uses concrete examples to illustrate surface learning phenomena. The narrative is clear, and the problem is well-motivated. This is a worthwhile research direction.
3. Comprehensive evaluation: The paper evaluates multiple open-source, closed-source, and reasoning models, providing broad coverage.

### Weaknesses
1. Incomplete Problem Framing: 
  - This paper aims to address surface learning—where models possess the relevant knowledge but fail to apply it correctly. However, the experiments only use multiple-choice and true/false questions. In these question types, models can obtain correct rewards through incorrect reasoning, a phenomenon known as shortcut learning or reward hacking. Yet surface learning should not be limited to shortcut learning alone. For example, in open-ended math problems where the solution space is vast and models cannot simply guess the correct answer, there still exists the scenario where models know the formulas but cannot solve the problems. We suggest incorporating open-ended question experiments to capture a more complete picture of surface learning beyond shortcut learning.
2. Missing critical implementation details:
  - The multi-turn dialogue implementation is not explained in either the main text or the appendix, undermining the credibility of Section 5.2.2's conclusions
  - Inverse-BC is not clearly defined. If it simply inverts the reward ranking within groups, this baseline seems both odd and poorly motivated
    
3. Counter-intuitive reward design: The appendix mentions increasing rewards for incorrect outputs, claiming this "indicates the model doesn't use answer paradigms to steal scores." This logic is questionable:
  - The model may be attempting to game the reward but failing
  - RL methods like GRPO require accurate rewards; rewarding incorrect outputs is highly counter-intuitive
  - This claim needs much more analysis.
    
4. Fragile conclusions:
  - Figures 6 and 7 show tiny differences (0.01-0.02 or smaller) between methods. Given GRPO's inherent training variance, these differences are not convincing enough
  - Conclusion 3 is based on non-F (providing the wrong formula) hurting performance on English questions, but non-F also performs worst on math Add problems. The relationship between F/non-F and performance needs more systematic analysis
    
5. No downstream validation: Experiments are limited to the constructed ME-Test suite. There's no validation on standard benchmarks (MATH, GSM8K, etc.). It's unclear whether these toy experiment findings generalize to real applications.

Minor Weaknesses

6. Key details buried in appendix: The reward indicator I is only explained in the appendix. Core methodological details should be clearly stated in the main text.
  
7. Scattered conclusions: The three surface learning behaviors and their mitigation strategies show inconsistent effectiveness across settings, lacking a unified explanatory framework.

### Questions
see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies surface learning in LLMs. They define it as cases where models recall formulas or patterns but fail to deeply understand and apply them. The authors introduce ME-Test that tests an LLM with progressive difficulty and exposure to controlled formula. They show that on this task, the model uses surface level heuristics. To mitigate, they propose 2 strategies: prompting for self-Concept planning and post-training  with a behavior-correction sampling strategy. Through extensive experiments on LLMs, they show that surface level learning can be mitigated in LLMs.

### Strengths
The paper clearly identifies and formalizes surface learning  in LLMs. By proposing a controlled testing framework, the authors show that LLMs simply recall symbolic rules but fail to apply them robustly to the task at hand. To mitigate, the authors propose 2 intuitive strategies, following human pedagogy. First, they propose to prompt the model to Self-Concept Planning, that asks the model to generate a plan before solving the question. The second strategy aims to modify sampling in post-training such that the model doesn't learn surface level features. Overall, the authors show a timely evaluation of LLMs and propose careful mitigation strategies to solve the problem.

### Weaknesses
The primary concern is equating surface level learning with the degradation in performance at more difficult questions. The failure of the model could be because of different reasons, e.g. hallucinations when generating longer answers. How would the authors disentangle issues with long form generation with surface level learning? The authors can present the average response length across the 3 settings and try to disentangle length to strengthen their argument.

Second, instead of simply asking the model about which concepts are relevant in prompt style mitigation, could the authors ask the model to reason about all concepts, and reason about irrelevant concepts as well. Some of $2$-way competition could improve the performance of prompting strategies.

How much does the structure of prompt affect the experiment results? For example, do the authors observe major differences in performance, depending on how the formulas are ordered in the prompt?

### Questions
Please check above for my questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies a phenomenon the authors call surface learning: LLMs achieving high accuracy on benchmarks by exploiting superficial correlations rather than engaging in deeper reasoning. They propose ME-Test, a probing framework that pairs original questions with minimally-edited variants where surface cues are altered but core semantics remain. Performance drops under these variants are used to quantify surface learning. To mitigate the issue, the paper introduces two interventions: thinking-path correction (prompting models to revise or reflect on earlier reasoning) and behavior correction (explicitly penalizing shallow patterns during generation). Experiments on GSM8K, StrategyQA, and BBH tasks show large performance drops under ME-Test for several open and closed models; the proposed mitigations yield partial recovery.

### Strengths
1. ME-Test is simple, reproducible, and architecture-agnostic: minimal edits to input help separate form-matching from semantic reasoning.  

2. The performance gaps are systematically measured and provide clear empirical evidence of shortcut reliance.

3. Mitigation strategies are lightweight (prompt-level) and do not require retraining.

### Weaknesses
1. The concept of “surface learning” is not clearly distinguished from existing notions like pattern matching, or spurious correlations. The paper introduces a new term but does not establish conceptual novelty.

2. ME-Test edits sometimes oversimplify meaning preservation, and no human validation is provided to confirm that all minimal edits preserve problem semantics fully.

3. Mitigation methods resemble prompt engineering and chain-of-thought reflection; they are not fundamentally novel, and their effectiveness varies across tasks.

4. Improvements after mitigation are moderate and do not eliminate the gap; no analysis is provided on failure cases or when mitigation backfires.

5. No examination of whether ME-Test correlates with real-world robustness or downstream utility; unclear if this is just another adversarial dataset.

### Questions
1. How is “surface learning” fundamentally different from shortcut learning or pattern exploitation described extensively in prior literature?

2. How do you ensure that ME-Test edits do not unintentionally change semantics or difficulty? Was human validation performed?

3. Does ME-Test correlate with other robustness metrics or real user-facing failures?

4. Can the mitigation methods cause degradation in standard accuracy or introduce verbosity without improving reasoning quality?

5. Do models with explicit training for chain-of-thought (e.g., R1, DeepSeek-R1) still show the same pattern?

### Soundness
2

### Presentation
3

### Contribution
2

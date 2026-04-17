# How well can LLMs provide planning feedback in grounded environments?

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Learning to plan in grounded environments typically requires carefully designed reward functions or high-quality annotated demonstrations. Recent works show that pretrained foundation models, such as large language models (LLMs) and vision language models (VLMs), capture background knowledge helpful for planning, which reduces the amount of reward design and demonstrations needed for policy learning. We evaluate how well LLMs and VLMs provide feedback across symbolic, language, and continuous control environments. We consider prominent types of feedback for planning including binary feedback, preference feedback, action advising, goal advising, and delta action feedback. We also consider inference methods that impact feedback performance, including in-context learning, chain-of-thought, and access to environment dynamics. We find that foundation models can provide diverse high-quality feedback across domains. Moreover, larger and reasoning models consistently provide more accurate feedback, exhibit less bias, and benefit more from enhanced inference methods. Finally, feedback quality degrades for environments with complex dynamics or continuous state spaces and action spaces.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies whether LLMs/VLMs can provide helpful feedback in a range of tasks. More specifically, the paper studies forms of feedback such as binary feedback, preference feedback, action advising, goal advising and delta action feedback. The paper tests this on tasks in Cliff Walking, MiniGrid, HierarchyCraft, ALFWorld, and Robomimic.

### Strengths
1. Feedback conditions. This paper considers a few very interesting methods for asking for feedback from FMs. Firstly, giving access to high-level environment dynamics - it is instructive to see that this, while unhelpful for small FMs, can improve the performance of larger FMs and even more so for largerVLMs.  Similarly, the paper considers in-context learning and thinking guides which are interesting methods for sometimes improving the accuracy of the feedback. 
2. Helpful comparisons and takeaways. There are some helpful takeaways for what helps LMs be better at providing feedback. For example, the comparison of ICL and Thinking Guides between smaller and larger FMs is neat. Similarly, the analysis of various biases like positional bias is quite instructive. The analysis of wrong responses is instructive in helping us understand what exactly may cause issues for FMs when it comes to providing feedback. There are some important model behavior takeaways here - for e.g., R1 preferring to stay away from the cliff in the Cliff Walking domain.

### Weaknesses
1. The method in which you collected the data makes the problem seem too easy unless I am misunderstanding. Let’s consider preference feedback. For each state, you have 3 actions  - $a_{(1)}$ from the expert, $a_{(2)}$ from the uniform (random) policy and $a_{(3)}$ from the mixture (half-expert and half-uniform). Now, if I take a state-action tuple $(s, a_{(i)}, a_{(j)})$ and we ask the FM which action it prefers, a random policy will have a 0.5 probability of being correct. To see this, $P(\text{chosen expert action}) = \frac{1}{3}P(\text{chosen expert action} \mid (a_1, a_2)) + \frac{1}{3}P(\text{chosen expert action} \mid (a_1, a_3)) + \frac{1}{3}P(\text{chosen expert action} \mid (a_2, a_3)) = \frac{1}{2}$ by noticing that if you select the action sampled from the mixture policy, you have still chosen an action from the expert policy with probability 0.5. Note that this issue applies to binary feedback and preference feedback. One naive way to address this could be to make a dataset of more than 3 actions per state and incorporate more actions from the random policy. This way, providing preference feedback by a mere coin toss would have a much lower (than 0.5) probability of being correct. 
2. The choice of domains could cover more difficult and longer horizon tasks. For e.g., Lift is one of the easier tasks in Robomimic compared to others like Square. The need for precision, for example, is significantly greater in the latter. The 0.28 noise tolerance for optimality of actions may not necessarily hold in the more complex environments and, as such, the test of the LM’s feedback capability could be tested more rigorously in these domains. This is an issue because for a large number of states in the continuous action space domain, the number of optimal actions is actually quite high; think about the actions the robot takes to reach the cube - for a pretty large number of time-steps, there are quite a few subtrajectories that are actually equally good. Especially, with the 0.28 noise tolerance, I wonder if this domain is actually testing planning feedback as rigorously as, say, Square. Coupling this with point (1), I am not sure whether we can trust VLMs’, say, preference feedback accuracy of 82% and binary feedback accuracy of 62%. 
3. Similarly, there are longer horizon tasks in BabyAI (like Synthseq or Bosslevel) than the MiniGrid tasks you considered in the paper - these tasks generally require much longer horizon planning and composition of strategies. These also require more visual understanding which would make the LLM vs VLM comparison more rigorous since the agent must plan ahead before committing to a high-level strategy that might send it down a wrong path pretty soon. As such, these would, perhaps, be a better test of reasoning especially when the agent veers far from the optimal states. 
4. There are some surprising results here which the paper does not adequately explain. Consider the action advising metric and consider Table 2. In a relatively very easy environment such as Four Rooms, the feedback accuracy is (0.32, 0.42, 0.61, 0.78) whereas in the more difficult domain ALFWorld, the feedback accuracy is (0.60, 0.83, 0.85, 0.83). Similarly, the paper claims in-context learning benefits larger models but, in table 3, for both Cliff Walking and Door Key, ICL seems to lower the performance of R1(and also often lowers performance of Llama 3.1 70B) across all types of feedback. Perhaps there is something to say here about performance on easier tasks vs harder tasks using ICL? Generally, I think the tables consist of a lot of information and the trends cannot be visualized too clearly and the paper’s analysis in text does not cover all the discrepancies.

### Questions
1. In a similar vein to the point in weaknesses-1, I wonder whether incorporating an action from a uniformly random policy is indeed the correct design choice. In environments with low-dimensional action spaces like CliffWalking and MiniGrid, even a random policy has around 20 to 25% probability of sampling an expert action. 
2. More broadly, how do you ensure that all 3 actions you sampled in your data collection procedure are not optimal? For example, in Robomimic-lift, not all states require high precision and for a large number of states, there actually exists quite a large number of equally optimal actions.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a comprehensive evaluation of how well LLM and VLM can provide right feedback in grounded environments. Key findings reveal that feedback quality improves with larger model size and reasoning capabilities, with reasoning models achieving over 80% accuracy in many domains. The research also examines how inference techniques like in-context learning and chain-of-thought affect performance, and identifies significant challenges in environments with continuous state/action spaces where feedback quality substantially degrades.

### Strengths
- The research question is novel and important. It also provides a reasonable and comprehensive evaluation method.

- Although the conclusions drawn are not surprising, it offers abundant evaluation methods, models, and conclusions.

### Weaknesses
- It does not provide new solutions for obvious defects in a specific LLM or VLM.

- Suggest that the authors improve Figure 1. When reading Figure 1 alone, I cannot understand the method used by the authors and the question they aim to answer; there are even some misunderstandings. I could only understand after reading the main text. For example, enrich the "compare" section with various different methods and different evaluation metrics.

### Questions
- Did the authors discover any counterintuitive conclusions? Because "larger is better," "more knowledge is better," and "low-dimensional discrete action spaces are better than high-dimensional continuous action spaces" are all consensus in the field.

- The paper demostrates that VLMs have failures due to errors in position judgment. Has there been further testing specifically for this aspect?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a systematic exploration of LLM/VLM provided feedback by evaluating reliability of different types of feedback mechanisms such as binary reward, action preference, action suggestion, goal suggestion, and action correction in different domains. The authors present ablations to assess feedback quality under commonly considered prompting patterns of ICL, COT, and additional domain knowledge.

### Strengths
* The paper is well motivated and claims are well supported.   
* The evaluation is thorough and the ablations offer interesting insights into prompting strategies that elicit better feedback.

### Weaknesses
* The evaluation environments are rather toyish and relatively short horizon, but they do offer a reasonable starting point to systematically studying feedback quality.  
* The analysis is done predominately with open weight LLMs, it would be interesting to evaluate how frontier models behind APIs perform (as those serve as a primary source of feedback in existing literature). Conducting this investigation might help offer more insights into prompting strategies that elicit better feedback.

### Questions
* While the metrics primarily report accuracy – I’m curious how reliable the feedback is under stochasticity of LLM sampling? Can the authors provide notions of variance for the evaluations conducted?  
* Recent works in manipulation (e.g. \[1\]) suggest that “visually prompting” of VLMs by suggesting “keypoints” or “colored arrows” provide better feedback on selecting action/goal suggestions. This seems complementary to the observation made in L320-323, I wonder if authors considered this style of prompting when it comes to domains with image inputs.

**References**  
\[1\] Huang, Wenlong, et al. "Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation." *arXiv preprint arXiv:2409.01652* (2024).

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
The paper provides a comprehensive empirical study evaluating how effectively Large Language Models (LLMs) and Vision-Language Models (VLMs) can generate planning feedback in grounded environments. It investigates multiple feedback types including binary feedback, preference, action advising, goal advising, and delta action feedback, across symbolic, textual, and continuous control domains. Models are tested under several inference conditions, including in-context learning (ICL), chain-of-thought reasoning (CoT), and access to environment dynamic.  

Results show that feedback quality improves with model size and reasoning capability, and that VLMs outperform LLMs in visually grounded or continuous domains (e.g., Robomimic). However, feedback performance degrades in environments with complex dynamics or continuous state/action spaces. The work is positioned as the first large-scale, unified analysis of how foundation models provide structured feedback for planning.

### Strengths
- **Comprehensive Evaluation:** Covers a wide range of environments, feedback modalities, and inference settings; far more exhaustive than prior isolated studies.  
- **Systematic Methodology:** Well-defined feedback formulations and evaluation protocols grounded in RL formalism (MDP-based).  
- **Strong Empirical Insights:** Clear patterns linking reasoning capacity, model size, and feedback accuracy; results align with observed trends in reasoning benchmarks.  
- **Relevance:** Addresses a timely research question of using LLMs as feedback providers instead of direct policies, which could reduce dependence on human annotations.  
- **Transparency:** Detailed experimental setup and appendices with environment-specific prompts support reproducibility.

### Weaknesses
- **Limited Novelty:** While the study is broad and thorough, it is primarily empirical and incremental. The paper lacks a clear articulation of how its framework advances beyond those like PlanBench https://arxiv.org/abs/2206.10498.  
- **No Unified Metric or Benchmark:** Results are difficult to compare across domains due to varying dataset sizes and ground-truth computation methods.  
- **Lack of Theoretical Framing:** The paper presents empirical findings without deeper analysis of why certain inference methods or model types work better.  The connection to Interactive Imitation Learning would be useful, and the types of feedback there (e.g. https://arxiv.org/abs/2211.00600). Framing of "action advising" is also not complete, SayCan is placed there, although it uses LLM for high-level task planning and Text2Reward (https://arxiv.org/abs/2309.11489) is also there, but it focuses on generating reward. It is not clear where Reward generation approaches should be placed. Additionally, major approaches like Code-as-Policies (https://arxiv.org/abs/2209.07753) and ExploRLLM (https://arxiv.org/abs/2403.09583) are omitted.
- **Overemphasis on Scale:** Improvements largely correlate with model size, suggesting the method itself may contribute less than the raw capacity of the underlying models.  
- **Limited Practical Implications:** While feedback accuracy is measured, there is no downstream demonstration (e.g., improved policy learning) to confirm that such feedback is actually useful in training.

### Questions
1. How does this study advance beyond prior empirical works such as PlanBench, which also examine LLMS for planning or RL?  
2. Can feedback accuracy translate to improved *policy learning* performance when used for imitation or reinforcement learning?  
3. How consistent are results across random seeds and prompt variations?  
4. Could a unified benchmark or normalized metric enable fairer cross-domain comparison?  
5. Are there observable correlations between reasoning steps (via CoT) and feedback correctness, beyond accuracy scores?

### Soundness
2

### Presentation
2

### Contribution
2

# h1: Bootstrapping Models to Reason over Longer Horizons via Reinforcement Learning

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Large language models excel at short-horizon reasoning tasks, but performance drops as reasoning horizon lengths increase. Existing approaches to combat this rely on inference-time scaffolding or costly step-level supervision, neither of which is scalable. In this work, we introduce a scalable method to bootstrap long-horizon reasoning capabilities using only existing, abundant short-horizon data. Our approach synthetically composes simple problems into complex, multi-step dependency chains of arbitrary length. We then train models on this data using outcome-only rewards under a curriculum that automatically increases in complexity, allowing RL training to be scaled much further without saturating. Empirically, our method generalizes remarkably well: curriculum training on composed 6th-grade math problems (GSM8K) boosts accuracy on unseen, Olympiad-level benchmarks (AIME) by up to 2.65x. Importantly, our long-horizon improvements are significantly higher than baselines even at high pass@k, showing that models can learn entirely new reasoning paths under RL. Theoretically, we show that curriculum-based RL with outcome rewards achieves an exponential improvement in sample complexity over full-horizon training, comparable to the gains from dense supervision, while providing strong training signal without additional annotations. h1 therefore introduces an efficient path towards scaling RL for longer horizons using existing data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new technique for doing synthetic task augmentation and curriculum generation for RL. The proposed augmentation method "h1" simply composes multiple short-horizon problems by using the output of previous problems as variables for future ones. The empirical evaluation is focused on a Qwen2.5 3B model trained on a composable version of GSM8K and evaluated across other math domains.

### Strengths
- Synthetic data augmentation for long-horizon reasoning is an under-explored, but very relevant research direction, as we want to scale RL to increasingly complex domains.
- Overall, the paper is well written and reads smoothly. I found the provided content and method description clear and intuitive.
- I found the cost analysis result in Section 6 insightful, providing interesting results on the tradeoff between cost and performance for curriculum learning, something I have not seen in prior work.

### Weaknesses
My **main concerns** about this work are that I found a large divide between the main claims made, the methodology, and the actual empirical results:
1) The paper makes several resounding claims, e.g., that their method "provides scalable synthetic long-horizon data, with explicit control over the horizon length and complexity without the need for new annotations." (lines 78-80). Yet even in the GSM8K settings, the proposed method requires annotating where variables of future claims can be augmented with previous answers, which is something extraneous to traditional RL datasets. Moreover, it is unclear to me how h1 would be applicable to domains beyond ones with numeric answers (e.g., game playing, coding, or scientific discovery, which the authors themselves mentioned in lines 38-39).
2) The current main results focus on a 3B Qwen 2.5 model on the authors' own synthetic version of GSM8K for multi-hop problems with limited output context. If I am interpreting Table 1 correctly, gains from h1 on this task appear only on the synthetic examples, with performance seemingly lower than regular compute matched non-augmented training L1 on the original set of problems (column 1 of Table 1). While the performance improvements on these longer synthetic problems are interesting, I do not think this result alone provides evidence for generalized "substantial performance gains on multi-step reasoning tasks" (lines 475-476), which I think would require testing the h1 model on other explicitly multi-hop problems across different domains after training.
3) When evaluating on a broader set of tasks in Table 2, the only baseline reported is the author's own RLVR implementation without *compute matching*. I am not sure why the other "compute matched" baselines from Table 1 are not also reported in Table 2.  Also, for the 7b Qwen experiments, I could not find the results on AIME@16. Given the 7B model has been widely used by prior reasoning works this would allow for comparison with a wider range of baselines. I would suggest expanding the range of evaluation tasks to include Minerva Math, OlympiadBench, and AMC as well, matching the evaluation setting of Dr.GRPO, and including this method as an independently trained baseline. Since Dr.GRPO itself is the method used by the authors for optimization, I think adding it as an independent baseline would allow readers to put into better perspective the actual impact of the methodology. At the moment, I do not think that claims like "while improvements obtained from RLVR on standard data is bounded by the base model’s capabilities, our method performs significantly better" (lines 76-78)or "Furthermore, our results show that the model learns genuinely new reasoning capabilities, rather than just refining existing ones." (lines 478-479) are justified, given that all absolute performance score reported with the 3B model for the reasoning tasks (e.g., AIME, MATH) are quite low and well below prior reasoning work.


**Other concerns**:
1) In Algorithm 1, it is described that at each training stage, the model's previous solution $y_h$ is used to construct future prompts, prepended in the dataset, for later curriculum iterations (Algorithm line 8). Thus, it is not clear to me how the "uniform mix" and "only long" baselines' data is constructed. Is this using ground-truth solutions perhaps? I could not find this information in the main text. Additionally, it is not entirely clear to me how compute was "equated" for these baselines. Since the main algorithm was run for 200 iterations at each stage (up to 5 stages), for how many iterations were the other baselines run for?

2) Qwen 2.5 3B instruct has not been post-trained with RL (unlike instruct models from the Qwen 3 family). Models from the Qwen 2.5 family have been shown to improve significantly with random rewards, and have often been tied to data leakage in the base model (e.g., [1]). To this end, I do not think that the sentence "Improving an Instruct model with RL is generally considered more difficult and gains signify performance improvements beyond just instruction tuning, which cannot be directly inferred for improvements on base models" (ln 250 253) is an accurate depiction of current findings.


3) A key component emerging in reasoning models is their ability to self-verify, correct, and re-attempt problems at any point of their reasoning chain [2]. The analysis in Appendix C, initiated in Section 3 seems to ignore this (e.g., the Equation in line 185, assuming intermediate findings are fixed) and makes other assumptions not applicable to the "single-turn" evaluation tasks chosen, that I do not think justify some of the strong definitive statements in the paper (e.g., from the abstract "Theoretically, we show that curriculum-based RL with outcome rewards achieves an exponential improvement in sample complexity over full-horizon training"). I would recommend toning down these claims (e.g., achieves -> "could achieve") and explicitly mentioning that these are made under several simplifying assumptions.

4) Looking at Table 4 in Appendix E, it seems the optimization was run with a quite low learning rate and gradient clipping magnitude. Were these choices due to a hyperparameter sweep and/or some instability observations? I could also not find this information in the paper.

5) In Appendix E, the authors also mention that they evaluated their models every 50 steps for each training horizon and selected the checkpoint with the highest performance on a separate validation set. I found this quite unusual, as reasoning RL methods train on less than a single epoch anyway, meaning validation and training performance should match, and report results with their final checkpoint. I think it would be important to clarify exactly what validation data is being used and whether it is from a different distribution than the training set. Additionally, I think validation curves and the collected performances every 50 steps for both the h1 model and all baselines should be added to the text, as this information is currently missing.

[1] Shao, Rulin, et al. "Spurious rewards: Rethinking training signals in rlvr." arXiv preprint arXiv:2506.10947 (2025).

[2] Guo, Daya, et al. "Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning." arXiv preprint arXiv:2501.12948 (2025).

### Questions
I would appreciate it if the authors could address the areas of improvement and questions raised in the  Weaknesses section of my review, and clarify any potential misconceptions for the discussion phase.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The proposed curriculum-based RL method, trained on simple 6th-grade math problems (GSM8K), dramatically improves performance on challenging, unseen Olympiad-level benchmarks (GSM-Symbolic, MATH-500, AIME), achieving up to 2.65× accuracy gains. These long-horizon reasoning improvements persist and even exceed baselines at high pass@k settings, demonstrating that RL enables models to discover entirely new reasoning pathways.
Theoretical Insight:
The authors prove that curriculum-based reinforcement learning with outcome rewards provides formal guarantees for effective generalization in reasoning tasks.

### Strengths
The method yields up to 2.65× accuracy gains on Olympiad-level benchmarks (GSM-Symbolic, MATH-500, AIME). Long-horizon improvements outperform baselines even at high pass@k, proving that RL enables models to learn entirely new reasoning paths.

### Weaknesses
1.Long-context is common to improve the performance of LLMs, the propsoed noviely is not obvious in the direction of long-context. 
2.The improvement is not obvious for AIME24, AIME25.
3.The experiments on one-single LLM, there is not strong proof if there is not improvements on the other LLMs such as r1-distill, and other reasoning LLMs.

### Questions
1.Given that long-context training is a common technique for improving LLM performance, what specific novel contributions does the proposed method introduce in the long-context direction?
2.Why are the performance improvements not substantial on challenging benchmarks such as AIME24 and AIME25?
3.Why are experiments conducted on only a single LLM, and why is there no evaluation on other reasoning-focused models such as r1-distill to demonstrate generalizability?

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
This paper introduces a novel curriculum-based reinforcement learning framework for improving long-horizon reasoning capabilities in large language models. The key innovation involves synthetically composing short-horizon problems into complex dependency chains of arbitrary length, then training models using outcome-only rewards under a progressive curriculum that automatically increases complexity. The method demonstrates strong empirical results, achieving improvement on challenging benchmarks like AIME while training only on composed GSM8K problems. Theoretical analysis shows curriculum-based RL achieves exponential improvement in sample complexity over full-horizon training.

### Strengths
The method provides a scalable path for improving long-horizon reasoning without requiring new annotations.
Demonstrates significant improvements on challenging benchmarks.

### Weaknesses
The core methodology bears significant resemblance to "Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning" (R³). Both approaches:
Use curriculum learning over reasoning steps
Employ outcome-only rewards
Gradually increase difficulty from simple to complex problems
Focus on improving multi-step reasoning capabilities
While the current work uses problem composition rather than demonstration sliding, the fundamental curriculum RL framework appears quite similar. The paper would benefit from a more explicit comparison and differentiation.

### Questions
Could you elaborate on how your approach fundamentally differs from R³'s reverse curriculum RL? What specific advantages does problem composition offer over demonstration-based curriculum learning?
The method shows strong transfer from GSM8K to math benchmarks, but how would it perform on non-mathematical reasoning tasks (e.g., logical deduction, commonsense reasoning)?

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
3

### Summary
The paper proposes a way to teach language models long-horizon reasoning using chained problems from datasets like GSM8k. The authors synthetically chain simple math problems into multi-step dependency chains, then train an LLM with RLVR under a curriculum, which gradually increase the chain length. This curriculum helps the model learn not just to solve individual steps but to reliably carry intermediate results forward through long chains of thought. The experiments show that model trained this way on composed GSM8k questions can also achieve gains on harder benchmarks like MATH-500 and AIME. The paper also claims that curriculum-based RL with outcome reward can yield exponential sample complexity improvement over full-horizon training.

### Strengths
* The paper's main idea, which systematically compose existing short-horizon problems into explicit long-horizon chains and then doing RL with it seems fresh.They way they formalize long-horizon difficulty via both per-step reliability and "context management" seems nontrivial.
* The experiment shows a set of ablations comparing horizon curricula (Len-2 --> Len-5) against a few baselines like only-l1, only-long, showing that their proposed curriculum based training is effective. They also show results on harder benchmarks like AIME to show the generalization of their proposed method. The authors also provide a sample-complexity analysis of curriculum vs full-horizon RL, which could potentially helps ground the empirical design.
* The paper is generally well written and easy to follow. The method section walks through atomic tasks, adapters and composition with a human-readable example.
* The work addresses a practical bottleneck: there is lots of verifiable short-horizon data but very little high-quality long-horizon data, especially for RL on LLMs. The paper offers a concrete recipe for turning that short-horizon data into a scalable RL curriculum and backs it with evidence that this actually produces meaningful new capabilities

### Weaknesses
* The models and datasets used in the experiment seems toyish. A Qwen2.5-3B model is small compared with other mainstream models. The experiment only includes such a small model, which may weaken the contribution of the proposed method. The dataset's scale is also small. It is unclear if the proposed method can easily scale up to and works on larger datasets (e.g., NuminaMath). 
* The only non-math results are on long-context reading benchmarks (LongBench-v2, Hash-hop), which are used purely for evaluation. That makes it hard to know whether the method really teaches a general long-horizon skill, or just exploits very specific algebraic structure plus better “carry the number” habits.
* The paper’s related work mostly focuses on RLVR, length generalization, and generic curriculum RL, but doesn’t clearly separate what’s really new in this composition + horizon-curriculum recipe vs these.
* Since the paper claims to avoid inference-time search/PRMs and “train the model to internalize long-horizon reasoning structures,” it would help to compare against a strong search-based method in the experiment.
* The paper leans heavily on strong numbers like “2.65x” improvement on AIME 2024 and good gains on MATH-500, GSM-Symbolic, LongBench-v2, and Hash-hop. However, AIME benchmarks are tiny (dozens of problems), so going from 3.96% --> 10.52% may correspond to only a small number of additional questions solved. For long-context tasks, improvements are modest (e.g., 35.3 --> 37.9 on LongBench-v2) and there isn’t much qualitative analysis of what changed. 
* The proposed theory assumes relatively clean factorization of errors and horizon structure that may not hold in messy LLM trajectories. The paper assumes that correctness at each step can be summarized by one Bernoulli variable $q_j$, and long-horizon correctness is exactly the product of those. In real LLM trajectories, there are things like error cancellation, redundant reasoning, or later self-correction. Those cases could break the assumptions made by this paper.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
2

# Cut the Overcredit: Precision First Process Rewards for Reasoning LLMs

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Process reward models (PRMs) supply step level supervision for reasoning LLMs but often \textit{overcredit} incorrect steps, producing high false positives that steer decoding and accumulate across long chains. We show analytically that false positives impose an asymptotic ceiling on Best of N alignment, whereas false negatives mainly slow convergence. To mitigate this, we introduce a label efficient recipe: convert existing step annotations into positive–negative pairs, train with a novel Overcredit Contrastive (OC) loss, and rebalance using lightweight negative augmentation and a simple curriculum. On PRMBench, our approach substantially lowers false positives and improves macro F1 over strong discriminative and generative PRMs. When used for guided beam search and Best of N selection, the resulting PRMs deliver higher downstream accuracy and robustness. Our results indicate that comparison centered training with balanced step data is a practical path to trustworthy process supervision without new human labels.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper discuss the critical issue of overcrediting (high false positive rate) in PRMs for reasoning LLMs. The authors provide both theoretical analysis and empirical evidence showing that false positives impose a ceiling on alignment performance. they propose a label-efficient approach: converting step annotations into positive-negative pairs, training with an Overcredit Contrastive loss, and applying lightweight augmentation with curriculum learning. Experiments demonstrate substantial reductions in false positive rates and improved F1.

### Strengths
1. Clear empirical improvement: The experiment section shows consistent downstream gains in guided search and Best-of-N
2. Good theoretical analysis: The FPR vs FNR asymmetry argument  is elegant and actionable
3. The Label-efficient method achieves substantial FPR reduction and improved macro F1

### Weaknesses
1.  The PRMs are known to be tricky and unstable in many settings, as reported in DeepSeek-R1. I would be more convincing to the community for direct comparison with GRPO (which avoids explicit PRMs). Showing that once some problem of PRMs is solved, it has clear potential to outperform ORM-driven methods. Currently, the paper only mentions some gain over existing SoTA PRMs, which somehow is not what people tend to use for baseline establishment.

2.  Would benefit from comparison with more PRM methods like implicit PRMs.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the "overcredit" issue in PRMs, where incorrect reasoning steps receive high rewards, leading to a high rate of false positives that harms model alignment. The authors introduce a label-efficient solution using a novel Overcredit Contrastive (OC) loss, which trains the model on positive-negative pairs converted from existing annotations.

### Strengths
- This work introduces Over-credit Contrastive loss to address the false positive bias in Process Reward Models.
- This work proposes a label-efficient framework that utilizes existing data and reduces annotation overhead.
- This work successfully improves F1 score and robustness in experiments on PRMBench.

### Weaknesses
Overall, I acknowledge the motivation for this work. I have the following major concerns:
- The ablation study on the OCLoss is missing. It is unclear whether the performance gains stem from the data curation or the reward correction itself.
- The current reward appears to be polarizing. I am concerned this might train the model to be overly decisive (i.e., "black-or-white"), causing it to incorrectly assign extreme reward values to ambiguous or speculative reasoning steps. Therefore, I would like to see case studies that analyze these scenarios.
- The data efficiency appears to be low. Could you provide the learning curves for your proposed training method and compare them against a baseline method?
- Due to the stochastic nature of model sampling, the Best-of-N (BoN) experiments should report the mean and variance across multiple trials.
- Missing Reference:

[1] Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning Large Language Models

[2] From System 1 to System 2: A Survey of Reasoning Large Language Models

### Questions
- Could you provide an ablation study to disentangle the effects of the OCLoss from the data curation process? This would help clarify whether the observed performance gains are attributable to the reward correction mechanism or the data selection method.
- The reward function seems to favor decisive, binary outcomes. Have you analyzed whether this might inadvertently encourage the model to become overly decisive, especially in ambiguous scenarios where nuanced or speculative reasoning is required? A qualitative analysis of such cases would be valuable.
- Regarding training efficiency, could you share the learning curves for your proposed method and compare them against a baseline? This would provide clearer insight into the data efficiency and convergence properties of your approach.
- To account for the stochasticity of the sampling process in BoN experiments, could you report the mean and variance of the results over multiple independent trials?
- Add more references.

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
This paper addresses reward hacking in PRMs for reasoning llm, where PRMs provide step-level supervision but often overcredit incorrect steps, leading to high false positive rates. The authors analytically demonstrate an asymmetry in BoN alignment: FPR imposes a precision ceiling that caps performance asymptotically, while false negative rates merely slow convergence.

### Strengths
1. The formal analysis of FPR vs. FNR asymmetry in BoN provides a rigorous, offering a clear rationale for prioritizing precision and filling a gap in prior PRM literature.
2. By repurposing existing datasets like PRM800K into 220K pairs via augmentation and curriculum, the method avoids costly new annotations, making it architecture-agnostic and easily integrable with discriminative or generative PRMs.
3. Substantial improvements in negative accuracy (e.g., from 30.66% to 50.89% on PRMBench) and downstream robustness (e.g., higher BoN accuracy on OOD policies) demonstrate real-world impact, with visualizations effectively illustrating reduced overcrediting.

### Weaknesses
1. The evaluation focuses primarily on math reasoning (e.g., MATH-500, AIME), with minimal exploration of broader domains like coding, potentially limiting generalizability despite claims of applicability to diverse reasoning tasks.
2. The thresholds in curriculum learning are handcrafted. There lacks ablation study on the sensitivity of such hyperparameters.
3. The compared baseline is merely Qwen-PRM. Other baselines in Table 2 are not compared.
4. The decrease in FPR comes at a cost of increase in FNR, showing that the method introduces a tradeoff.
5. The expansion strategy shows marginal gains, less than 1% improvements on PRMBench.
6. The method's reliance on a predefined difficulty curriculum (based on initial reward differences) may introduce sensitivity to baseline PRM quality; ablation studies on curriculum bins are underreported, risking instability in noisier datasets.
7. While analyzing credit assignment (min/product/sum), the paper assumes uniform step independence, overlooking potential correlations in real chains; this could undervalue compounding effects in long-horizon tasks beyond the tested benchmarks.

### Questions
- Is the proposed overcredit contrastive loss the vanilla Bradly-Terry loss? If so, why give it a new name?
- To mitigate the FPR issue, can we simply tune the threshold of PRM output rewards?
- The equations in Section 2 have wrong markers
- In Figure 3/4 captions, the orange and blue lines should be exchanged
- Why ThinkPRM has different scores in Figure 4 when n=1?

### Soundness
3

### Presentation
2

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
This paper improves process reward models (PRMs) for reasoning-oriented language models, with a focus on mitigating false positives that cap alignment performance during guided decoding and Best-of-N (BoN) response selection. The authors provide an analytical argument explaining how false positives impose a precision ceiling in BoN settings, and propose a approach based on pairwise training using the Bradley–Terry loss. They augment existing step-level annotations with context-matched negative samples and use a curriculum over pair difficulty to further stabilize training. Experimental results on PRMBench and downstream alignment settings show improved precision, reduced false positive rates, and stronger alignment performance, especially for out-of-distribution generators.

### Strengths
- The paper presents a clear and useful analysis showing that false positives impose a hard ceiling on alignment, particularly in Best-of-N setups. The derivation is straightforward and helps justify the focus on reducing false positives.
- The proposed method is simple and effective, simply using Bradley-Terry loss leads to meaningful improvements in precision and reduced false positive rates. 
- The experimental results are strong and consistent, demonstrating both improved PRM performance on PRMBench and better downstream alignment in guided beam search and Best-of-N selection, including for out-of-distribution policies.

### Weaknesses
- The method does not meaningfully address  label efficiency and data imbalance in supervision. The approach still depends on access to both positive and negative annotations at the step level, and similar data balancing could be achieved in pointwise training. The claims of label efficiency and data imbalance mitigation feel overstated.
- The proposed “Overcredit Contrastive Loss” is mathematically identical to the well-known Bradley–Terry (BT) loss, which has been used extensively in outcome reward modeling. Rebranding it without acknowledging the equivalence weakens the paper’s novelty. Bradley-Terry loss is mentioned in appendix but not main body, and I did not find the authors justifying the relation to BT loss.
- The writing could be tightened for conciseness and professionalism. For instance, the section around line 296–300 includes an unnecessary list that takes up space without adding clarity, making the paper feel less compact than it could be.
- While the paper frames reward hacking in the context of reinforcement learning, it does not include any RL experiments. This disconnects the theoretical claims from demonstrated practice and weakens the application to real-world RL training.

### Questions
see above

### Soundness
2

### Presentation
2

### Contribution
2

# Curriculum-Augmented GFlowNets For mRNA Sequence Generation

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
Designing mRNA sequences is a major challenge in developing next-generation therapeutics, since it involves exploring a vast space of possible nucleotide combinations while optimizing sequence properties like stability, translation efficiency, and protein expression. While Generative Flow Networks are promising for this task, their training is hindered by sparse, long-horizon rewards and multi-objective trade-offs. We propose Curriculum‑Augmented GFlowNets (CAGFN), which integrate curriculum learning with multi‑objective GFlowNets to generate de novo mRNA sequences. CAGFN integrates a length‑based curriculum that progressively adapts the maximum sequence length guiding exploration from easier to harder subproblems. 
We also provide a new mRNA design environment for GFlowNets which, given a target protein sequence and a combination of biological objectives, allows for the training of models that generate plausible mRNA candidates. This provides a biologically motivated setting for applying and advancing GFlowNets in therapeutic sequence design. On different mRNA design tasks, CAGFN improves Pareto performance and biological plausibility, while maintaining diversity. Moreover, CAGFN reaches higher-quality solutions faster than a GFlowNet trained with random sequence sampling (no curriculum), and enables generalization to out-of-distribution sequences.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors tackle the problem of mRNA sequence design, where the task is to explore a vast space of nucleotide combinations while optimizing for properties such as stability, translation efficiency and protein expression. They identified GFlowNets as a promising candidate for this task and proposed curriculum-augmented GFlowNets (CAGFN) that integrates a length-based curriculum as well as a new mRNA design environment that trains the model to generate plausible mRNA candidates given a target protein sequence and a combination of biological objectives.

### Strengths
1. This paper gives very detailed and well-versed background on what the authors believe to be good mRNA sequence design: to generate diverse sequences that (1) preserve the desired protein sequence, (2) satisfy biological constraints, and (3) expose explicit trade-offs between competing objectives. 
2. The authors give a fairly reasonable justification for using GFlowNets as a baseline for the development of the proposed method, namely that GFlowNet learn policies that sample complete objects with probability proportional to a user-specified reward, which supports the discovery of diverse solutions and naturally avoids mode collapse.
3. The authors nicely explained the limitations of directly applying GFlowNets on mRNA sequence design and proposed a remedy using curriculum learning.

### Weaknesses
1. The authors introduced three critical criteria for good mRNA sequence design (Strength 1) but the results do not seem to address them point-by-point. If the authors believe they have already done that, I would appreciate an explanation and potentially a revision to make that more obvious.  
2. The results do not seem particularly impressive, with very mild improvements on codon adaptation index, minimum free energy, and GC content compared to a limited set of baselines. More importantly, since one of the main contributions of the paper is justifying an adapted version of GFlowNets for this task, it is worth demonstrating superior performance over other methods, not just variants of GFlowNets.
3. The presentation of the results can be further improved. For example, I would recommend showing the three metrics individually rather than only showing the reward which could be somewhat indirect.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Curriculum-Augmented GFlowNets (CAGFN) that combine curriculum learning with multi-objective GFlowNets for de novo mRNA sequence generation. Specifically, the paper considers the task of generating multi-objective optimized mRNA condon sequences given a target protein sequence. The paper introduces adaptive, length-based curricula that preferentially present tasks (by protein length) where the model is demonstrating the fastest learning progress. 

In empirical experiments, the paper first showed that multi-objective GFN achieves better diversity and rewards than other RL baselines, such as PPO, Reinforce. Then, the paper compared its curriculum strategy against three other baselines, including short-only, long-only, and random order, and showed that CAGFN trains significantly faster while achieving competitive results on Pareto performance and reward metrics.

### Strengths
1. The introduction of mRNA design for a specific target protein and the motivation for leveraging GFN for combinatoric discrete mRNA codon design is well written and clearly explained.

2. The proposed integration of curriculum learning and multi-objective GFN is a practicable and well-motivated approach for addressing the long-horizon, sparse-reward problem inherent in GFlowNet training for long sequences.

3. In the empirical experiments, CAGFN achieves much faster training while maintains competitive performances across different benchmarks.

### Weaknesses
1. The paper's core motivation is directly contradicted by its own experimental results. The paper's primary justification for a curriculum is that training on long sequences is difficult due to "extremely sparse credit assignment". However, the "Long-Only GFN" (LGFN) baseline, which was trained exclusively on these supposedly difficult long sequences, achieves the best or most competitive performance on the 85-120 AA generalization tasks (Table 1). This finding strongly suggests that the sparse reward problem is not as debilitating as claimed and that a curriculum is not necessary to achieve high performance.

2. CAGFN provides no clear performance benefit over non-curriculum baselines. The paper's central claims of "achieving higher rewards, better objective trade-offs, faster convergence, and broader Pareto-front coverage" are not sufficiently supported by the data. In Figure(Table?) 5, the full CAGFN model shows no performance advantage over the standard MOGFN without curriculum. In the main ablation (Table 1 & 2), CAGFN is not better than other baselines in majority cases, especially in 85-122 AA task where LOGFN is the best in 4 out of 5 metrics. In addition, the performance differences between different ablation baselines are quite small in general, especially considering the larger variances of CAGFN. These results indicate that the only demonstrable advantage of CAGFN is training speed over other curriculum baselines, not superior generative quality or generalization.

3. In terms of novelty, I appreciate the authors have acknowledged that the novelty of the paper is combining curriculum learning with GFN on the task of mRNA design. I would highly recommend exploring more meta-learning strategies beyond the simple 3 ablation baselines studied in the paper to explore the potential of curriculum-based methods to genuinely improve GFlowNet performance, rather than just training speed.

### Questions
While the idea is interesting and the training speedup is notable, the core claims are not convincingly supported by the paper's own empirical results (Tables 1, 2, and 5). The strong performance of the LGFN baseline, in particular, undermines the paper's central motivation. The advantages of the proposed method, therefore, appear to be limited to computational efficiency, which is a mismatch with the paper's stronger claims of improved generative performance. 

For improvement, I would highly recommend exploring more meta-learning strategies beyond the simple 3 ablation baselines studied in the paper to explore the potential of curriculum-based methods. 

In addition, although the paper compared with 3 curriculum strategies on 4 different protein metrics, there is no discussion on why a specific curriculum is better or worse on different metrics. A further analysis and interpretation on the results would greatly strengthen the paper by providing insight into how the curriculum is guiding the optimization process, rather than just presenting the final numbers.

### Soundness
1

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
This paper introduces Curriculum-Augmented GFlowNets (CAGFN), a method that integrates curriculum learning with multi-objective conditional GFlowNets to generate diverse and high-rewarded mRNA sequences conditionally on a given set of weights for each objective. The approach tackles long-horizon challenges of this task by using a length-based curriculum that guides the model to learn from an adaptive task distribution. CAGFN is evaluated on mRNA design tasks, a newly introduced environment.

### Strengths
1. Improving GFlowNet training for long-horizon settings is an important research question.
2. A new environment in the science domain is a good contribution for both GFlowNet and the scientific discovery community.

### Weaknesses
1. While the clarity is fine in general, there are some concerns. Most importantly, their usage of "conditional" or "unconditional" generation seems ambiguous. If I understand correctly, the mRNA environment is inherently conditional, conditioned on a target protein $p_{\text{seq}}$ with length $L$. The goal is to train a _single_ GFlowNet policy that generates sequences proportionally to a reward function, for each target, i.e., $P_{F}^{\top}(x;p_{\text{seq}}) \propto R(x;p_{\text{seq}})$ where $P_{F}^{\top}$ is marginal of $P_F$ over $\mathcal{X}$. If we additionally conditioned on the reward weights, the goal will be $P_{F}^{\top}(x; p_{\text{seq}},w) \propto R(x;p_{\text{seq}}, w)$. I was confused about this point when I first read the paper. I believe it would be clearer to explicitly state the target protein condition in section 4.1. Please correct me if I'm wrong. There are also some minor issues (see the "Minor" section below).
2. The method initialises the task distribution (Eq. (3)) as uniform over lengths, and I think this cannot be considered as "curriculum" learning, since it does not encourage progressive learning from easy (short) to harder (longer) tasks. And this contradicts the motivation of this work.
3. Empirical results are weak. 1) Specifically, standard benchmarks for GFlowNets (e.g., GridWorld, bit sequence generation) are missing, while the proposed curriculum learning method can be applied straightforwardly. 2) There's only a marginal performance gain over MOGFN (Figure 5) or ROGFN (Table 1) when considering the reported error range (standard deviation).

(Minor)  
4. Line 66: TB does not reduce the variance compared to FM or DB. It's the opposite.  
5. Line 190: $a$ is already used as an action in section 3.2.  
6. Line 209: "Fig" is missing for the figure 1 reference.
6. Figure 2(a) is hard to digest; I have no idea how I can recognise the Pareto front from it.

### Questions
1. How's the distribution of $L$ if you randomly sample the target protein from the environment?

---

### LLM usage disclosure
I did not use an LLM for my review, but I used grammar check software.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a new GFlowNet algorithm for generating mRNA sequences. Since mRNA sequences are relatively long, the search space is extremely large, making it computationally demanding to train an effective flow function. To address this challenge, the paper introduces a curriculum learning approach that splits the training of the flow function into multiple tasks. The training begins with easier tasks and gradually progresses to more difficult ones. This gradual learning process helps the model develop a better flow function, as it can leverage knowledge gained from simpler tasks to generate more complex parts of the sequence. Experiments on multi-objective mRNA generation is conducted to showcase the effectiveness of the proposed approach.

### Strengths
- Employing GFlowNets for biological sequence generation has shown well-documented performance. However, training a flow function for relatively long sequences remains a significant challenge. This paper proposes a method to address this important issue.
- The use of curriculum learning to achieve this goal is well-motivated.

### Weaknesses
Although the idea of using curriculum learning is well-motivated, I believe the current implementation in the paper may not be as effective as expected and may lead only to marginal improvements. Based on my understanding, the training procedure is largely the same as in other GFlowNet methods, with the main difference being that the order of learning each part of the sequence is determined by inferred difficulty. This modification alone is unlikely to significantly improve the efficiency of GFlowNet training. I believe this is reflected in the experimental results, as the performance gains appear to be quite marginal.

### Questions
If the proposed method indeed splits sequence generation into tasks, with each task corresponding to generating a specific part of the sequence, then since the model generates sequences autoregressively, it would be expected that the most difficult tasks are generating the later parts of the sequence. Is this in agreement with your experimental observations?

### Soundness
3

### Presentation
3

### Contribution
2

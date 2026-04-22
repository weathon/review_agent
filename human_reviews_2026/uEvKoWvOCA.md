# Generative Auto-Bidding in Large-Scale Auctions via Diffusion Completer-Aligner

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 4, 4, 4

## Abstract
Auto-bidding is central to computational advertising, achieving notable commercial success by optimizing advertisers’ bids within economic constraints. Recently, generative models have shown great potential to revolutionize auto-bidding by directly learning a policy from large-scale datasets. Among them, the diffuser is superior in tackling sparse-reward challenges, along with its trajectory stitching and explainability capabilities, making it well-suited for industrial auto-bidding. However, its performance could be limited by generation uncertainty, particularly regarding generations’ dynamic illegitimacy and preference misalignment, which can lead to suboptimal bids and further cause poor performance when competing with other advertisers in highly competitive auctions. To address it, we propose a
Causal auto-Bidding method based on a Diffusion completer-aligner framework, termed CBD. Firstly, we conduct a theoretical analysis and propose a completer to augment the training process with an extra random variable t for enhancing the dynamic legitimacy between adjacent states. Then, we employ a trajectory-level return model as an aligner to refine the generated trajectories in inference, aligning
more closely with advertisers’ objectives. Experiments across diverse settings demonstrate that our approach not only achieves superior performance on large-scale auto-bidding benchmarks, such as a 29.9% improvement of conversion value in the challenging sparse reward setting, but also delivers significant improvements on an online advertising platform, including a 2.0% increase in target cost.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a training procedure to improve diffusion models that participate in auto-bidding systems by introducing completer and aligner components. The diffuser's objective is producing bidding parameters, meaning it does not have to run quickly within milliseconds per ad request, and thus can be a fairly complex model. The additional two components aim to improve the legitimacy of the taken actions and alignment with cumulative reward of the campaign being optimized.

### Strengths
1. Paper clearly explains the motivation behind building a complex and expensive model for an apparently real-time task.
2. The idea of learning to complete appears to be simple and creative. 
3. Performance improvement against baselines is apparent. I do not believe additional datasets should be used, since there aren't any around for simulating actual auctions that I know about (with actual competitors, where the model under test has to participate in auctions, rather than just predict some value).
4. Pretty extensive ablation studies.

### Weaknesses
1. The basics of diffusion-based decision making are very poorly explained, and pretty much assume the reader is familiar. So either you assume the reader is familiar, and just refer to references, or assume the reader is not, and explain clearly what is it. I am not familiar with the idea, and had to go through relevant literature to understand the explanation, and then it was actually redundant when I understood it, and could be shorter. Maybe a simple figure instead of a lot of words and jargon could be better here.
2. Auto-bidding could be many things, such as bid shading, for example. The paper's abstract and title need to be more precise in setting the stage.
3. One thing is not clear - it appears as if the diffusion model is supposed to generate the lambda parameters that are used for bidding, but then in the A/B test section, latency is reported. Shouldn't the latency cost be zero?

### Questions
1. Why is the latency reported, if as far as I understand your proposed model doesn't even compute the auction bids, but just the lambdas that are used for bidding? Please clarify. I assume I missed an important detail.

### Soundness
3

### Presentation
1

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
In this paper, the authors propose a novel methodology called CBD (Causal auto-bidding method based on a diffusion completer-aligner framework) for addressing the auto-bidding problem in the advertising domain. The proposed approach aims to mitigate the uncertainty inherent in generative models and the paper also provides a theoretical analysis supporting this aspect.
In the experimental section, the authors evaluate the performance of the proposed methodology under different budget constraints and against several baselines. The results show that the proposed method consistently outperforms the compared approaches, achieving particularly higher improvements in low-budget scenarios. Moreover, as reported in Section 5, the authors deployed the proposed algorithm on an online advertising platform, where it improved both the Target Cost and CPA Valid Ratio without increasing overall costs—demonstrating the practical applicability of the proposed methodology.

### Strengths
- The paper is complete and well written
- The proposed methodology is well justified well benchmarked against several baselines
- The test conducted on a online advertising platform is a good point, demostrating the practical usability of the proposed methodology

### Weaknesses
- While Table 6 reports the performance of CBD on the D4RL benchmark using multiple seeds and standard errors, Table 1 presents metrics without confidence intervals or measures of variability.
- As acknowledged by the authors, a key limitation of the work is its exclusive focus on the auto-bidding strategy, without addressing other important aspects such as auction mechanism design or cooperation between advertisers.

Overall, while the proposed methodology appears promising, I have some concerns regarding the evaluation process, particularly about the consistency and robustness of the reported results.

### Questions
- Could you clarify why the performance reported in Table 1 is presented as a single value?
- More generally, did you consider multiple random seeds in your experiments? Aside from the D4RL Benchmark, I couldn’t find any other mention of this detail in the paper.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper examines the application of diffusion-based generative models to industrial-scale auto-bidding in online advertising. The authors identify two key failure modes of standard diffusion planners, namely dynamic illegitimacy and preference misalignment; the former is a nuanced and interesting issue that requires careful analysis. To mitigate these failure modes, the paper proposes CBD (Causal Auto-Bidding via Diffusion Completer–Aligner), a two-stage framework consisting of a Completer and an Aligner. The Completer augments training with an additional random variable t to enhance dynamic legitimacy between adjacent states. The Aligner employs a trajectory-level return model to refine generated trajectories during inference, aligning them more closely with advertisers’ objectives. The method is evaluated on AuctionNet and its sparse variant, showing consistent gains over baselines (e.g., +29.9% conversion value on AuctionNet-sparse). An online A/B test reports a +2.0% increase in conversions with no budget increase.

### Strengths
1. The paper articulates a practical issue—dynamic illegitimacy—that requires careful scrutiny and analyzes its causes. It addresses dynamic illegitimacy in diffusion-based auto-bidding via a Completer–Aligner framework. The theoretical analysis of distribution mismatch (Eq. 6) provides insight into diffusion-model failures.
2. The paper presents extensive experiments on public benchmarks (AuctionNet) and in real-world deployment (online A/B tests). The improvement in sparse-reward settings is substantial.

### Weaknesses
1. Overstated novelty: The “Completer” is similar to masked-trajectory modeling (akin to Decision Diffuser with partial observations). While applying it to diffusion models is novel, the core idea resembles concurrent work. The paper does not adequately differentiate itself.
2. The resolution of dynamic illegitimacy is not reported in detail—for example, quantitative metrics before and after mitigation are missing. It is unclear whether the issue is fully resolved.

### Questions
1. What exactly is $y(\tau)$ in Eq. 3 and Eq. 10? Is it the total conversion value ($\sum_t r_t$), or a vector of KPIs (e.g., [conversions, CPA])? How is it computed in sparse-reward settings where most $r_t = 0$?
2. In Algorithm 2, Step 5 uses only the refined next state $ \tilde{s}'_ {t+1} $ (not the full refined suffix $ \tilde{s}'_ {t+1 : T} $) for action generation. Why not use more future states (e.g., $\tilde{s}'_{t+1:t+L}$) to improve action quality?
3. Aligner vs. classifier guidance: Have you compared your gradient-based Aligner with classifier-guided diffusion (using $R_\varphi$ as the classifier)? Classifier guidance operates during denoising and may yield better alignment with similar computational cost.
4. You mention only a 6 ms overhead relative to DT, yet the paper uses 100 diffusion steps, which is theoretically costly. Could you provide details of the online inference setup (e.g., number of diffusion steps, batching, precision) and the corresponding GPU configuration?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a novel framework to improve auto-bidding in large-scale online advertising auctions. It introduces a Completer to reformulate difusser training as a completion process which augments the model with a random variable, and a Aligner to employ a trajectory-level return model to refine the generated trajectories. Comprehensive offline and online A/B test proves the effectiveness of the proposed method.

### Strengths
1. Motivation. The paper identifies generation uncertainty (illegitimacy and misalignment) as a key bottleneck in applying diffusion-based policies to industrial auto-bidding.

2. Evaluation. The authorsn conducted comprehensive offline evaluation and ablation study, as well as online A/B tests.

### Weaknesses
1. Connections with Existing Works. What's the connection and the key difference between the proposed method and existing methods? From my point of view, the proposed method looks like a combination of the decision transformer (since there is an auto-regressive formulation in Eq 10) and the decision difusser. A clearer elaboration on the connections would be great.

2. Trajectory-level return. As stated in the 2nd paragraph in the Introduction, the difussers can "generate a trajectory of states based on accumulative rewards over multiple steps as a condition". If so, why do the authors need to propose a trajectory-level return model? Is it redundant given Diffuer's existing ability to employ accumulative rewards?

3. Presentation. The whole section 3 contains too many technical details but lacks a high-level demonstration of the overall picture of the proposed method, as well as a discussion on the connections and differences with existing works.

### Questions
1. From my understanding, In Eq. 10, using both s_{0:t} and y(\tau) as the condition makes the formulation similar to the auto-regressive style decision transformer. Or, what's the key difference between the formulation in Eq. 10 and the decision transformer?

2. It would be great to have "a high-level demonstration of the overall picture of the proposed method, as well as a discussion on the connections and differences with existing works.", as mentioned in the Weakness.

### Soundness
3

### Presentation
2

### Contribution
3

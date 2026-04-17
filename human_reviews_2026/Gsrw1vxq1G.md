# RoboMD: Uncovering Robot Vulnerabilities through Semantic Potential Fields

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Robot manipulation policies, while central to the promise of physical AI, are highly vulnerable in the presence of external variations in the real world. Diagnosing these vulnerabilities is hindered by two key challenges: (i) the relevant variations to test against are often unknown, and (ii) direct testing in the real world is costly and unsafe. We introduce a framework that tackles both issues by learning a separate deep reinforcement learning (deep RL) policy for vulnerability prediction through virtual runs on a continuous vision-language embedding trained with limited success-failure data. By treating this embedding space, which is rich in semantic and visual variations, as a potential field, the policy learns to move toward vulnerable regions while being repelled from success regions. This vulnerability prediction policy, trained on virtual rollouts, enables scalable and safe vulnerability analysis without expensive physical trials. By querying this policy, our framework builds a probabilistic vulnerability-likelihood map. Experiments across simulation benchmarks and a physical robot arm show that our framework uncovers up to 23\% more unique vulnerabilities than state-of-the-art vision-language baselines, revealing subtle vulnerabilities overlooked by heuristic testing. Additionally, we show that fine-tuning the manipulation policy with the vulnerabilities discovered by our framework improves manipulation performance with much less fine-tuning data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a framework for predicting robot manipulation vulnerabilities by learning a deep RL policy over a vision–language embedding space. It identifies vulnerable regions without costly real-world testing, builds probabilistic vulnerability maps, and uncovers 23% more unique vulnerabilities than baselines, improving downstream robustness.

### Strengths
1. The paper tackles a highly practical and underexplored problem—diagnosing robot manipulation vulnerabilities—using a smart and scalable approach.

2. Reformulating exploration over a learned vision–language embedding is elegant and helps the policy find subtle failure modes efficiently.

3. The experimental evaluation is thorough, spanning both simulation and real-world settings, and clearly shows strong generalization and effectiveness.

### Weaknesses
1. The method doesn’t really explain why vulnerabilities emerge, so it feels more like a tool for finding problems rather than understanding them.

2. The embedding-based approach may not fully capture complex real-world dynamics, which could limit how well it scales to more unpredictable environments.

### Questions
How well would the proposed method handle scenarios where the embedding space fails to capture subtle but critical real-world variations?

Can the framework provide more interpretable insights into why specific vulnerabilities occur, rather than just locating them?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents RoboMD, a novel framework that diagnoses failure modes in robot manipulation policies by training a RL agent to explore a continuous vision–language embedding space. Rather than relying on hand-crafted test conditions or costly physical rollouts, RoboMD learns to search for vulnerabilities virtually by interpreting the embedding space as a semantic potential field that repels from known successes and attracts toward failures.

The core contributions include:

1. A deep RL-based diagnostic agent operating in a semantically structured latent space.

2. A framework that identifies both seen and unseen failure modes, improving upon vision-language models (VLMs) and heuristic baselines.

3. The use of identified vulnerabilities for targeted fine-tuning, leading to substantial improvements in robot manipulation policies.

Experiments are conducted on both simulated environments  and real-world scenarios (UR5e arm), demonstrating improved diagnostic performance and policy robustness.

### Strengths
### Originality
- Reformulation of diagnosis as RL search in a semantically meaningful potential field is elegant and novel.
- Using multimodal embeddings to structure the search space is a strong innovation.
### Quality
- Broad evaluation: 4 manipulation tasks × multiple policy types × real vs. simulated settings.
- Benchmarks vs. GPT-4o, Gemini, Qwen2-VL show strong generalization and robustness.
### Clarity
- Excellent visuals and modular structuring.
- Detailed Appendices with ablation studies and embedding quality analysis.
### Significance
- Addresses core robotic AI challenges: cost and risk of real-world rollouts, poor generalization, and hidden failure modes.
- Offers safe, scalable, and data-efficient diagnosis that’s practical for deployment-stage validation.

### Weaknesses
This work is of great practical significance, but I am allowed to express some concerns.
### Assumption of Embedding Quality:
1. The success of the approach hinges critically on the quality of the semantic embedding space. If poorly structured, RL search may yield meaningless trajectories.
2. Though addressed via BCE + contrastive loss, there is no quantitative measure of embedding smoothness across multiple datasets.
### Limited Real-World Testing:
1. While real-world results (UR5e) are included, most findings are in simulation.
2. Have not seen some of sim2real's settings. The community has always believed that real machine data is the most reliable, although simulation data has the advantage of large data and low cost. This work would have more relevance if real machine data was included in the loop.

### Questions
1. While the authors acknowledge the combinatorial complexity of real-world disturbances, the training and search process still relies on a limited and predefined set of action embeddings (either sampled or pre-collected). This raises a potential contradiction: how well does RoboMD generalize to more open-ended or compositional disturbances beyond the known embedding space? It would strengthen the paper to explicitly discuss the coverage limitations of the action space and possible mechanisms to expand it during deployment.

2. Several recent works [1][2] have begun to explore failure prediction in robot foundation models, especially through uncertainty-aware approaches. However, this paper does not compare against those methods or discuss how RoboMD relates to them. Some of these prior efforts incorporate uncertainty modeling (e.g., epistemic or aleatoric uncertainty) into their optimization process, which allows the model to focus exploration or calibration in regions of high uncertainty. I believe RoboMD would benefit significantly from integrating a similar mechanism. For example, encouraging the RL agent to focus its exploration around uncertain or ambiguous regions in the semantic space. If the authors can demonstrate that such uncertainty-aware exploration improves the agent’s performance or failure detection coverage, it would substantially increase my confidence in the generality and practical robustness of this approach.


3. The framework would benefit from a demonstration of real-world deployment, even at small scale. A promising direction is to implement a sim-to-real flywheel loop, where $\pi_{MD}$  is first trained in latent space using synthetic or simulation-aligned embeddings, then deployed to identify high-risk conditions on a physical robot. The collected failure cases could be added to the embedding space and used to retrain $\pi_{MD}$  in future iterations. This would close the loop and demonstrate practical deployability, which currently remains underexplored.

> [1] SAFE: Multitask Failure Detection for Vision-Language-Action Models Qiao Gu etc. Neurips 2025
> [2] Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure Detection for Imitation Learning Policies Chen Xu etc. CoRL 2024

### Soundness
3

### Presentation
4

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
The paper considers robotic manipulation and preventing mistakes or ‘vulnerabilities’ when using large model vision-language (VLM)  perception, that may include variations in object properties.  Starting with a trained manipulation policy, a new RL process is trained over variations to find such vulnerabilities, called RoboMD. This policy can be queried to predict a failure over a sequence of actions, and given a finite set of failures can put probabilities on these forms. These are then used to fine tune the original manipulation policy.  Experiments are carried out in RoboSuite and show physical plus simulation experiments, and address some key issues.

### Strengths
The idea of creating a failure predictor is a good one, as can be used in many dynamic settings.  This can identify ‘distance’ from key vulnerabilities, and this can include lighting variation and object color and size (cases that are notorious for their difficulty to handle generally). 

The method can be used to isolate and avoid some key failure modes. With sufficient prior examples, generalization over virtual roll outs is carried out. 

The theory shows how a shaped MDP will lead to efficient training. 

While depending on prior knowledge and setting, the method might be applicable for a variety of settings. The method shows some ability to generalize, given a sufficient failure training set. 

The authors provide code and overall the work has good reproducibility.  This work has a good probability of leading to other follow on efforts.

### Weaknesses
The primary weakness is lack of adequate baselines.  There are various approaches for control that provide safety, such as barrier methods. It is not surprising that a conventional RL would not do as well as one supplemented with an additional failure mode predictor. The semantic potential field amounts to a kind of semantic barrier.  Also not clear is what happens to a vanilla RL if it is given some explicit bad examples with negative reward.  

The method keys on the training selection and variations, so this is a critical step that relies on the user, and while generalization is shown in the examples, it isn't clear how this translates to new or more complex environments. 

The term ‘environmental variation’ might be too general, since it appears it is more about sizing objects, color of objects, and lighting variations. 

The number of predetermined versus virtual rollouts is a key question. It was not easy to see how this was precisely addressed in the experiments.

The method can identify some key failure modes, although guarantees are not clear. In the end we are still restricted by the VLM-based approach that doesn’t inherently do well with spatial reasoning. 

The 3 theorems are interesting, showing how the MDP approach can lead to a shaped MDP that leads to efficient exploration and learning convergence.  However, these don’t quite address the key questions of covering the vulnerability space, or semantic related issues in general.

### Questions
How can the predetermined bad cases be incorporated into a vanilla RL approach?  

What is the tradeoff in predetermined examples versus virtual samples?  

What further experiments can be made to compare with other control safety methods?

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
4

### Summary
This paper proposes RoboMD, a framework for diagnosing vulnerabilities in pre-trained robot manipulation policies by recasting failure discovery as a reinforcement-learning (RL) search problem in a continuous vision–language embedding space. Instead of manually probing variations (e.g., lighting, object color, geometry), the authors train a deep RL policy (πᴹᴰ) to navigate a semantic potential field learned from limited success/failure rollouts. This field is shaped through a multimodal embedding (ViT + CLIP + MLP) trained jointly with BCE + contrastive loss. The policy explores this embedding using PPO to identify regions that correspond to likely failures and produces a failure-likelihood map.

### Strengths
1. I really like the idea of reframing failure discovery as an active exploration problem rather than a passive evaluation or stress test. It’s an elegant conceptual step that connects representation learning, uncertainty estimation, and RL in a unified way.
2. Extensive experiments across simulated and physical domains, ablation on embedding design (BCE vs. BCE + contrastive), and real robot validation strengthen the empirical credibility.
3. I appreciate that the paper doesn’t just stay empirical—it also provides theoretical justification via potential-based reward shaping and convergence proofs. Even if some of it is idealized, the effort to link the math with the algorithm is commendable.

### Weaknesses
1. The method heavily relies on the assumption that the learned multimodal embedding forms a smooth manifold where semantic similarity aligns with failure likelihood. While visualizations suggest separability, it’s unclear whether embedding-space distances correspond to physically meaningful variations. Without empirical measures of “semantic continuity,” this assumption could fail for unseen scenarios.

2. The baselines include standard RL and VLM models, but exclude other uncertainty-based failure detection or OOD methods (ensembles, energy-based models, MC-Dropout). These would provide a fairer picture of how much RoboMD really advances the state of the art in model failure analysis.

3. PPO over a 512-D continuous space is expensive and potentially unstable. The paper doesn’t analyze training cost, convergence time, or scaling to multi-object or long-horizon tasks. It’s unclear whether the approach generalizes beyond relatively simple tabletop manipulation.

4. The reported “23 % more vulnerabilities” result isn’t clearly defined. What constitutes a unique failure? A cluster in embedding space, or a distinct environmental configuration? The interpretation of this metric should be clarified.

5. While the paper demonstrates that RoboMD can generalize to unseen environment variations within the same domain, it remains unclear whether the diagnosed vulnerabilities or fine-tuned policies transfer across datasets or task families. For example, if a policy fine-tuned using RoboMD-diagnosed failures on RoboSuite were evaluated on a different benchmark such as LIBERDO or ManiSkill, would it retain improved robustness? Testing such cross-domain transfer would provide stronger evidence that RoboMD captures more fundamental, domain-invariant failure structures rather than merely overfitting to intra-domain variations.

### Questions
1. Table 5 reports that fine-tuning with RoboMD-guided failures outperforms fine-tuning with all failure cases, despite using a much smaller dataset (1.3 GB vs 9 GB as shown in Table 11). This is impressive but counterintuitive. Is the gain purely due to better sample efficiency (targeted failures), or does RoboMD generate more diverse failure cases in embedding space? Some discussion of this discrepancy would clarify the underlying mechanism.

2. Could the authors clarify how an action taken in the embedding space (a ∈ ℝ⁵¹²) translates into a physical environment change? Is there a learned or rule-based decoder that converts latent vectors into concrete variations (e.g., lighting, color, geometry)? If not, how are the “new failure cases” in Table 5 actually generated and validated on the robot?

3. How does RoboMD handle unseen actions or rollouts that lie outside the support of the training distribution?

### Soundness
3

### Presentation
3

### Contribution
3

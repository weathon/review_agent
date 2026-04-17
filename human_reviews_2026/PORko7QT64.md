# Anchored Supervised Fine-Tuning

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Post-training of large language models involves a fundamental trade-off between
supervised fine-tuning (SFT), which efficiently mimics demonstrations but tends
to memorize, and reinforcement learning (RL), which achieves better generaliza-
tion at higher computational cost. Dynamic Fine-Tuning (DFT) recently emerged
as a promising middle ground, reweighting SFT objectives with token probabili-
ties and achieving improvements in certain reasoning domains, though it exhibits
instability in other tasks. We provide a analysis of DFT through the reward-
weighted regression (RWR) framework, revealing that it corresponds to a spe-
cific auxiliary distribution choice that yields provably tighter RL bounds than
standard SFT. However, our analysis also uncovers a critical limitation: this con-
struction lacks distributional anchoring, leading to progressive drift that under-
mines training stability. To address this, we propose Anchored Supervised Fine-
Tuning (ASFT), which augments DFT’s reweighting with lightweight KL regu-
larization to preserve tightness while ensuring stability. Empirically, ASFT con-
sistently outperforms both SFT and DFT across mathematical reasoning, medical
knowledge grounding, and code generation, achieving substantial improvements
with minimal computational overhead. Our RWR framework provides a system-
atic lens for understanding post-training methods and demonstrates that principled
theoretical analysis leads to both stronger guarantees and practical gains.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors indicate that DFT suffers from distributional shift and propose ASFT to prevent it. ASFT is a post-training method which uses KL divergence to anchor it to a reference base model to prevent distributional shift during training.

### Strengths
* Identifies the shortcoming of DFT and suggests a novel way to handle it
* The method is theoretically grounded using a reward-weighted regression framework
* Empirical analysis and ablation studies are thorough and evaluation is done for mathematical reasoning and knowledge intensive tasks
* Clearly stated contributions backed by theory and experimental analysis

### Weaknesses
* Requires the base model as a reference - memory and computational overhead. 
* ASFT-LoRA is proposed to avoid memory overhead, the performance significantly drops when compared to ASFT and is very close to the performance of SFT
* DAPO still performs better than ASFT on an average, more analysis and discussion on why and where RL remains preferable would be nice
* No reporting of standard deviation of the runs

### Questions
* Would anchoring prevent overfitting when we have a small dataset?
* Have the authors considered reporting the mean and standard deviation across several independent runs? This will strengthen the findings further
* Edit suggestions for better readability
1. The second sentence in lines 72 to 75 is a repetition 
2. Font size is very small for Table 1
3. Typo in line 310
4. Font in Figure 3 is illegible 
5.  Consistent color coding of methods across plots will improve readability - it is a little confusing now and readers have to check the legend across plots every time

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
- The paper proposes that SFT maximizes a lower bound on the RL objective
- It introduces auxiliary distribution to control the tightness of the bound, and shows that DFT is a special case which achieves tighter lower bound than SFT
- It shows that DFT suffers from distribution shiift as training proresses
- To mitigate the dirtribution shift, ASFT augments the DFT objective with a KL regularization term between the currect model and reference model (base model)

### Strengths
- ASFT shows better performance than SFT and DFT on math reasoning and medical knowledge
- The paper theoretically derives the importance weighted lower bounds and proves the tightness of the lower bound for DFT

### Weaknesses
- ASFT requires an additional copy of base model to be stored in memory, as well as an additional forward pass through it
- While using LoRA mitigates the memory overhead, it causes a significant performance drop (the performance of ASFT+LoRA is almost equivalent to simple SFT as seen in Table 4)

### Questions
No questions other than those mentioned in the weaknesses.

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
The paper proposes Anchored Supervised Fine-Tuning (ASFT), a variant of SFT and DFT (dynamic fine-tuning) based on a Reward-Weighted Regression (RWR) perspective. It shows that SFT is a lower bound on the RL objective and that DFT provides a tighter bound but suffers from instability. ASFT introduces a reverse-KL anchor to the base model to stabilize DFT while preserving its theoretical benefits. Empirical studies on math, medical QA, and code generation tasks demonstrate consistent improvements over SFT/DFT and comparable or better results than RLHF-style methods (GRPO/DAPO).

### Strengths
+ Using a reverse-KL anchor to control DFT drift is a natural, well-motivated modification that stabilizes training. 

+ The RWR derivation linking SFT, DFT, and ASFT is internally consistent and clearly written. The proofs in the appendix (esp. the tightness lemma and DFT-as-RWR derivation) are correct under the stated sparse-reward and support assumptions.

+ The paper evaluates on multiple domains (math, medical, and code) across LLaMA-2/3 and Qwen models, and includes practical metrics such as GPU memory and wall-time comparisons. The authors acknowledge compute cost, and ASFT-LoRA offers a realistic, resource-efficient version.

+ Ablations studies on reverse vs. forward KL, $\lambda$ sweeps, and model-size scaling are reported and analyzed.

### Weaknesses
- The paper asserts that the KL anchoring “preserves tightness,” but technically ASFT optimizes a regularized surrogate, not the same lower bound. This needs clarification. 

- The theory relies on sparse rewards and full-support overlap; it’s unclear whether results extend to open-ended generative or preference-learning tasks.

- The transition from sequence-level RWR theory to token-level training is described heuristically rather than derived rigorously.

- Results lack multiple random seeds or confidence intervals; improvements may be within run variance for some metrics.

- Full ASFT doubles memory usage (requires both model and reference for KL), and while ASFT-LoRA helps, it trades accuracy for efficiency.

Minor
- Evaluation focuses on reasoning and domain-specific QA; general instruction-following, safety, or long-context benchmarks are missing.

### Questions
Please consider to address questions in the weaknesses.

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
The paper analyzes Supervised Fine-Tuning (SFT) and Dynamic Fine-Tuning (DFT) through the lens of reward-weighted regression (RWR), showing that both can be seen as optimizing lower bounds of an underlying RL objective. Within this framework, DFT corresponds to using an importance-weighted auxiliary distribution that yields a strictly tighter bound than SFT but suffers from instability due to uncontrolled distributional drift. To address this, the authors propose Anchored Supervised Fine-Tuning (ASFT), which augments DFT with a KL regularizer towards a fixed base model, acting as an “anchor” that stabilizes training. Experiments on math, medical QA, and code tasks demonstrate that ASFT outperforms SFT and DFT, and on medical QA it approaches RL methods at lower cost.

### Strengths
The paper makes a clear and conceptually grounded contribution to understanding the link between supervised fine-tuning and reinforcement learning–style objectives. By framing SFT and DFT under a unified reward-weighted regression (RWR) objective, it shows how auxiliary distributions determine the tightness and stability of RL lower bounds. Identifying DFT as a particular importance-weighted auxiliary distribution, and then introducing the “anchored” variant (ASFT) with a KL regularizer towards the base model, is an elegant way to stabilize DFT while preserving its tighter connection to the RL objective.

The theoretical development is careful and transparent: each step from SFT → DFT → ASFT is well-motivated, with assumptions stated and a useful strict-tightness result for DFT. Empirically, the work provides consistent evidence that ASFT improves over SFT and DFT across math reasoning, medical QA, and code generation, and acts as a strong initializer for RL methods such as DAPO in the medical domain. Overall, the paper is clearly written, theoretically sound within its setting, and practically relevant. Its main value lies less in radically new algorithms and more in clarifying and stabilizing the intermediate regime between SFT and RL, offering both conceptual insight and a simple recipe that practitioners can readily adopt.

### Weaknesses
1.Empirical scope w.r.t. the “bridge SFT–RL” claim. The experimental coverage across math reasoning, medical QA, and code generation is solid, but all setups remain in a supervised/offline, correctness-style regime with static datasets. There are no studies on preference-based post-training, open-ended instruction following, or safety/factuality trade-offs, which are central targets of RLHF-style training. Given the paper’s framing as a general method that “bridges SFT and RL”, the current evidence still primarily supports gains on a few curated accuracy benchmarks. Either tempering the scope of the claims or adding at least one preference/RLHF-style experiment would make the story more convincing.

2.Uneven RL comparisons across domains. The paper provides a careful comparison between ASFT and RL-style methods (GRPO/DAPO), and shows that ASFT is competitive and that ASFT+DAPO further improves performance—but this analysis is restricted to medical QA. For math reasoning and code, the baselines are primarily SFT/DFT/iw-SFT, without RL counterparts. As a result, it is difficult to calibrate how close ASFT gets to “RL-level” performance in these domains, or whether its advantages as a cheaper alternative or initializer hold once RL is available. Even a smaller-scale RL comparison on one math or code task would significantly strengthen the empirical case.

### Questions
1.Empirical scope and “bridge SFT–RL” framing. Could you either (i) provide preliminary results or analysis of ASFT under a genuinely RLHF-style setting (e.g., preference-based rewards or open-ended instruction-following), or (ii) clarify and narrow the scope of the “bridging SFT and RL” claim to supervised/offline, correctness-style regimes? Any concrete evidence or discussion in this direction would help readers understand how far your conclusions are meant to generalize.

2.RL comparisons beyond medical QA. For math reasoning and code, do you have any results—perhaps at smaller scale or with fewer training steps—where ASFT is directly compared to GRPO/DAPO or another RL baseline, so that we can calibrate how close ASFT gets to “RL-level” performance beyond medical QA? If such experiments are infeasible, could you provide a more detailed justification or expectation (e.g., based on reward sparsity or domain differences) for whether you believe the medical-domain findings should transfer to these tasks?

### Soundness
3

### Presentation
3

### Contribution
2

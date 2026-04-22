# Semi-Supervised Preference Optimization with Limited Feedback

- Avg Score: 6.00
- Decision: Accept (Oral)
- Scores: 8, 6, 2, 8

## Abstract
The field of preference optimization has made outstanding contributions to the alignment of language models with human preferences. Despite these advancements, recent methods still rely heavily on substantial paired (labeled) feedback data, leading to substantial resource expenditures. To address these challenges, we study the problem of Semi-Supervised Preference Optimization in which the idea is to learn from both a small number of pairwise preference labels and a large pool of unpaired samples simultaneously. Our key theoretical contribution proves the existence of an optimal reward threshold capable of separating winning and losing responses with high probability, which enables a principled pseudo-labeling of unpaired data. By leveraging these pseudo-labels, SSPO effectively distills latent preferences from large-scale unpaired data, thus maintaining human alignment while drastically reducing acquisition costs. Extensive experiments across datasets validate this remarkable data efficiency; for instance, SSPO trained with Mistral-7B-Instruct on just 1% of UltraFeedback consistently surpasses strong baselines trained on 10% of UltraFeedback.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the scenario in which labeled preference data is scarce, which is arguably a realistic setting given that carefully crafting and curating labeled preference data at scale is expensive and challenging. 

Their proposed method is to find an optimal reward threshold that can be used to assign weak-labels to unlabeled data. The optimal reward threshold is carefully derived (section 4), by first defining a Bayes risk value, $R(\delta)$ (Equation 6), which looks at the overlap between (weighted) reward distributions of winning and losing responses. Perhaps at the core of the paper's contributions is a theorem that demonstrates the existence of an optimal reward threshold (assuming reward distributions come from Gaussian distributions). 

The provided theorem depends on knowing groundtruth mean reward values. In Section 4.2 the authors demonstrate how to address this limitation in practice, by estimating the reward densities from the available labeled datasets (Equation 9), which can then be used to solve for the threshold $\delta$ that minimizes the Bayes risk, which then allows one to assign weak labels to the unlabeled dataset.

The authors demonstrate strong results compared to prior alignment algorithms under scenarios in which the amount of labeled dataset is controlled for. 

Overall a good paper that carefully walks through the derivation of their method, followed by a practical implementation of their method which demonstrates positive results for their targeted use cases.

### Strengths
The paper is well motivated to address the challenge in building labeled preference data at scale.
The paper is well organized and written clearly, such that the "build up" to their theorem is well explained step by step. 
The paper demonstrates positive results of their approach working much better than strong baselines in a realistic scenario in which the amount of groundtruth labeled data is controlled for. In particular, I appreciate the thorough sweep of settings for the baselines (Table 8). I think the results are compelling to demonstrate the strength of their suggested approach when labeled data is scarce.

### Weaknesses
One thing I'm curious to see would be the accuracy of the threshold-based reward model in isolation, especially in comparison to 1) an "oracle", and 2) a reward model trained with 10%, 50%, 100% of the available data. A compelling result would be if the pseudo-labels assigned by SSPO, trained on a small set of data, was at par compared to a reward model trained with much more data. 

Also, although SSPO clearly out-performs the alternative strong baselines when data is scarce, just for completeness, I'd be curious to see results for Table 2 (maybe in Appendix) when more data is used (ex: 50%, 90%) just to see when the benefits of the pseudo-labels wear out.

### Questions
One thing I wish to clarify: the reward function $r_\theta(\cdot)$ being used in Equation 9 is being updated as training progresses? and IIUC, this is what motivates the adaptive scheduler, such that the clean labels are being used first? 

Also, are the same weights being used for $r_\theta$ and $\pi_\theta$, or separate weights?

Do you think there's a way to leverage your optimal threshold $\delta$ value in online RL settings, which seem to be the more popular approach nowadays?

Can you add a quick definition for the metrics ("Win Rate", "Length-controlled win rate", MT), and in particular explain why a "length controlled" win rate is needed, for those who haven't been following the alignment literature closely?

Minor typos: line ~376: "whichaims" --> "which aims"

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a semi-supervised approach to preference optimization (SSPO) that uses pseudo-labels on unpaired data to improve alignment without additional human feedback. A reward threshold is estimated via kernel density, and samples above this threshold are treated as preferred. The method aims to make use of large unlabeled datasets to improve data efficiency. Experiments various domains show improvements on benchmarks like AlpacaEval2.0.

### Strengths
1. The paper tackles an important and realistic bottleneck in preference optimization as human-labeled preference pairs are generally scarce and expensive. Hence, the proposed approach of leveraging abundant unpaired data is nice. The motivation of the paper is clearly described. 
2. The proposed pseudo-labeling approach appears to be easily intergratable with existing approaches (especially SimPO) without any notable architectural changes. 
3. The authors conduct extensive experiments across various model types (Phi-2 2.7B, Mistral 7B, Llama3 8B) and benchmarks. Various baselines are also included.

### Weaknesses
For now, my concerns are still in the form of questions that I hope the authors can clarify (see Questions below). To briefly highlight the most important ones here: 
1. The theoretical setup doesn’t seem consistent with the changing reward distributions during training, which makes the “Bayesian” framing and the threshold stability unclear.  
2. I’m worried the pseudo-labeling approach creates a self-reinforcing loop, where initial model biases are amplified over training.  
3. The unpaired data seem very similar to the evaluation data, and combined with longer training runs, it’s hard to tell if the gains come from the method itself or just from having been exposed to similar prompts/styles and compute.

### Questions
1. In Section 4.1 (Eq. 6), you assume two stable conditional reward distributions, $p(r|s=1)$ and $p(r|s=0)$, allowing you to estimate a fixed threshold $\delta^\star$ from $D_L$. However, in your implementation, the reward $r_\theta (x,y)$ is computed by the same policy being trained (SimPO-style), so both $p_\theta (r \mid s)$ and hence $\delta^\star$ are moving during training. 
How do you justify treating $\delta^*$ as stationary? Doesn’t this moving threshold risk making pseudo-labels self-reinforcing? 
2. You train SSPO on models that are already fine-tuned or aligned rather than on base or purely SFT models. Why was this choice made? Does SSPO require a pre-aligned reward structure to be stable? 
3. In Table 2, how do you explain the sometimes decreasing performance with more data for, e.g., SimPO Llama3 across LC, WR, MT? 
4. To me pseudo-labeling can be quite counterintuitive, hence I'd like to clarify the validity and potential confirmation bias loop here. Pseudo-labeling should only work when your unlabeled data is similar enough to your labeled data so that your reward signal is roughly correct, and you are essentially enriching your data. Do you have results on the accuracy of your pseudo labels in a controlled setup (e.g., on held-out human labels)? More broadly, how do you mitigate confirmation bias, i.e., the amplification of existing reward model biases through self-labeling,  given that SSPO uses the same reward for pseudo-labeling and optimization? 
5. Appendix C (Table 8) shows that SSPO is sometimes trained for 2-3 epochs while baselines are trained for 1. Why? Why is this a fair comparison? Could the reported gains be partly due to increased optimization budget rather than algorithmic efficiency?
6. In Section 5.4 you report that unpaired data used by SSPO is semantically similar to AlpacaEval 2.0 prompts, and that performance benefits correlate with this similarity. It makes sense that training on these / pseudo-labeling these will improve performance on the benchmark. However, isn't this essentially contaminating our training data with evaluation data? Even without observing the actual preference from AlpacaEval2.0 the model now sees highly similar prompts and styles during training. How can we rule out that the gains stem from this overlap? 
7. Is your code available somewhere?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes Semi-Supervised Preference Optimization (SSPO), which uses a reward-like score to pseudo-label large unpaired responses via a learned threshold δ (estimated by KDE to minimize estimated Bayes risk), combined with a curriculum scheduler that shifts weight from paired to pseudo-labeled data. Experiments on UltraFeedback and two domains report AlpacaEval2 LC gains over DPO/ORPO/SimPO/KTO/SSRM/SPA under limited paired data.

### Strengths
- Simple, practical recipe: thresholded pseudo-labeling on top of SimPO + a scheduler; easy to implement.
- Broad empirical sweep across backbones (Phi-2, Mistral-7B, Llama-3-8B) and two domains, with some ablations (prior sensitivity, scheduler).
- Shows consistent LC improvements in data-scarce regimes; engineering details (EMA, KDE bandwidth, configs) are documented.

### Weaknesses
- Limited novelty: essentially classic self-training/pseudo-labeling using $ r_\theta $ and $\delta$; close to SSRM/SPA and prior semi-supervised alignment.
- Theory misaligned with practice: Theorem relies on high-probability separation (max loser ≤ $\delta$ ≤ min winner) under sub-Gaussian assumptions, unrealistic with overlapping reward distributions; equality $\delta^*=\mu_l+t_1=\mu_w-t_2$ not generally guaranteed; no analysis of KDE/EMA estimation error or consistency.
- Objective inconsistency: Sec. 3 introduces $f_\theta$ with priors $P(s)$ but implementation reduces to BCE on $\sigma(r-\delta)$; “reward” is the policy’s own likelihood (confirmation bias risk).
- Unfair comparisons: SSPO uses large unpaired sets while baselines do not; missing strong baselines: SFT(unpaired)+DPO/SimPO, “Pseudo-DPO” built from the same $r_\theta,\delta$, confidence-gated SFT. Training budgets appear unaligned.

### Questions
See weaknesses.

I am willing to raise my rating if the authors can address my points above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes Semi-Supervised Preference Optimization (SSPO), which augments preference optimization with abundant unpaired SFT-style data. The method trains a reward-like scoring function from limited labeled pairs, estimates winning/losing reward densities, and selects a Bayes-risk-minimizing threshold via KDE to pseudo-label unpaired responses; an adaptive scheduler shifts weight from labeled to pseudo-labeled data over training. Experiments on UltraFeedback, UltraMedical-Preference, and DSP Business report substantial gains in AlpacaEval2.0 LC and WR using as little as 1 percent labeled data, often surpassing baselines trained with 10 percent.

### Strengths
- Novel framing of preference learning as a Bayes-optimal classification problem that yields a principled reward threshold for pseudo-labeling unpaired data is clear.
- Allowing SFT-style unlabeled data for a preference learning is a promising approach for reducing the annotation cost.
- Strong data-efficiency: with 1 percent labels, SSPO often beats 10 percent baselines.
- Extensive experiemnts, including label-noise testing, ablations, show the robustness of SSPO.

### Weaknesses
- The computational overhead should be addressed in comparison to other baselines.

### Questions
- What is the reason for SSPO performing better than semi-supervised *pairwise* learning algorithms (SSRM, SPA)?

### Soundness
3

### Presentation
4

### Contribution
3

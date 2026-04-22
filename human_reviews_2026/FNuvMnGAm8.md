# Load Balancing Mixture of Experts with Similarity Preserving Routers

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 2, 6, 8

## Abstract
Sparse Mixture of Experts (MoE) models offer a scalable and efficient architecture for training large neural networks by activating only a subset of parameters (“experts”) for each input. A learned router computes a distribution over these experts, and assigns input tokens to a small subset. However, without auxiliary balancing mechanisms, routers often converge to using only a few experts, severely limiting model capacity and degrading performance. Most current load balancing mechanisms encourage a distribution over experts that resembles a roughly uniform distribution of experts per token. During training, this can result in inconsistent routing behavior, resulting in the model spending its' capacity to learn redundant knowledge. We address this by introducing a novel load balancing loss that preserves token-wise relational structure, encouraging consistent expert choices for similar inputs during training. Our experimental results show that applying our loss to the router over a popular load balancing loss results in 35% faster convergence and lower redundancy, while removing balancing hyper-parameters completely.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposed SIMBAL which uses a router orthogonalization loss that is believed to encourage expert specialization. The authors claim significant improvements over the traditional auxiliary load balancing loss, and claim that it removes balancing hyper-parameters completely.

### Strengths
1. The authors claim significant improvements over the traditional auxiliary load balancing loss.
2. The authors claim that it removes balancing hyper-parameters completely.
3. The author provides much analysis.

### Weaknesses
1. The central idea of an orthogonality loss on the router weights is conceptually similar to the "router orthogonalization loss" in the ERNIE 4.5 technical report. It would be beneficial to cite this parallel and frame the work's distinct contributions.
2. The orthogonality of router does not necessarily leads to different expert routing behavior. Suppose R=BQ（Q^TQ = I_E, B\in R^{DxD}）, one can easily apply B^{-1} to input x (e.g. absorb into the former layers' mlp), so that any router's behaviour equals to an orthogonal router on a routed input.
3. I doubt the gain comes from the scale constraint rather than the constraint on direction. It may affect the logits size so as to affect the loss.

### Questions
1. I suggest some experiments may be done to address point3 in weakness.
2. Explanation on point2 in weakness is welcome.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the problem of redundancy and inconsistent routing in MoE models. The authors argue that standard load-balancing losses, which enforce a uniform distribution of tokens to experts, are inefficient. To solve this, they propose SIMBAL, that encourages the router's weight matrix (R) to be orthogonal by penalizing the difference between its Gram matrix (R^T R) and the identity matrix (I_E).

### Strengths
The idea of "similar tokens should be routed similarly" to preserve semantic consistency, is interesting. However, the methodology and computation cost are questionable.

### Weaknesses
Please see the question block.

### Questions
As mentioned in the abstract and throughout the paper: the goal of this model is to preserve the pairwise angles of the inputs (this means if two input $h_1$ and $h_2$ are similar, their routing should also be similar). This is achieved by promoting orthogonality in the router weights, because orthogonal matrices are dot-product ... preserving ....

However, preserving the dot product of the D_M-dimensional inputs require the router R to satisfy R R^T = I_{D_M}.  But, this seems weird for me, as R is a  $D_M \gg E$ matrix, (where the input dimension D_M is much larger than the number of experts E). Also the proposed loss  ||R^T R - I_E||_1, is a standard regularization that enforces orthogonality on the columns of $R$, encouraging diversity among experts.  So, it’s not clear how enforcing this diversity aligns with the goal of preserving input similarity. I think there is some mismatch and needs clarification. 

The paper says the proposed PES requires less computation and is highly scalable. again in the same paragraph states that calculating PES requires inference once with the full model parameter, through all experts (a multiplier of 3.6-4.9x FLOPs in our case). Indeed,  4-5x increase in FLOPs is not cheap or scalable. Running all experts for a single token is computationally expensive.

PES (Eq. 6) measures cos(f_i(x), f_j(x)), the similarity between the outputs of all $N$ experts for the same token $x$. But again, this requires dense computation for all experts for every input. 

Optional suggestion: In table 5, the baselines used for comparison are from 2018-2019. I understand the paper aims to improve upon earlier models, but incorporating recent baselines for the comparison can provide a meaningful evaluation.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes SIMBAL, an auxiliary router loss for Mixture-of-Experts models that preserves token-wise relational structure by softly encouraging orthogonality of the router. The motivation is that angle-preserving routers produce more consistent expert choices for similar inputs, reducing redundancy and speeding training. The authors define a computationally cheap orthogonality loss, propose Pairwise Expert Similarity (PES) as a scalable metric of expert redundancy, and show empirical gains over the standard Load Balancing Loss on two model scales: faster convergence during training, lower PES, and better final perplexity and downstream benchmarks.

### Strengths
The idea is clear and intuitive. A code snippet is also provided in the appendix, showing the simplicity of the implementation. Moreover, the auxiliary loss hyperparameter is not sensitive to tuning, which makes this method easily integratable into existing architectures.

The empirical gains are strong: training convergence is significantly faster with SIMBAL loss, and the final perplexity and benchmark scores are also better for MoE trained with SIMBAL.

New metric for expert redundancy that quantifies the similarity of experts without requiring much computation

Evaluation on a diverse set of pretraining benchmarks, with consistent improvements across all of them.

### Weaknesses
The authors mention that stronger benchmark performance is realized when training on significantly more data than the datasets used in the paper. If possible, it would be good to see some results on how the SIMBAL method performs when training on these larger datasets, compared to traditional MoE.

The paper motivates orthogonality as angle preserving, but a more formal connection between router orthogonality and reduced redundancy / improved specialization (maybe via an analysis of routing variance or router saturation) would improve the justification of this approach.

The authors mention the loss calculation is cheap but it would be more convincing to see the comparison of training throughput between this method and standard MoE (if there is a difference).

### Questions
Can you compare the training throughput of standard MoE and SIMBAL? Is there a noticeable difference with SIMBAL loss or is it negligible? 

How do the improvements depend on number of experts, and number of active experts? Does the performance change if the MoE is more or less sparse?

What happens if you combine SIMBAL and LBL? Can they work together or do they conflict with each other?

PES is computed over 4M sampled tokens, how does the measurement vary over different token samples? Can you report the variance across samples?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper provides a simple strategy to regularize the MoE router to avoid some issues with load-balancing loss (LBL) which is commonly used. The LBL encourages uniform routing to prevent expert collapse and token dropping, but as a result may create redundancy and lack of specialization among the experts. Intuitively, it is desirable that the routing decisions are similar for similar inputs, but this may be suppressed by LBL. To explicitly encourage this, the authors attempt to constraint the router to be orthogonal via a soft-regularization penalty. This penalty is simply the l1 distance of the router gram matrix to the identity. Adding the penalty does not add significant overhead per step and is able to reach a given target loss in fewer steps than LBL.

### Strengths
The authors provide a simple and principled approach to regularizing an MoE router in order to promote expert regularization and mitigate the redundancies in traditional load balancing loss approaches. This appears to substantially speed up MoE training. The empirical evidence is thorough and convincing.

### Weaknesses
Minor: The term load-balancing loss for SIMBAL seems slightly incorrect.

### Questions
The observed balance (high SEU) appears to be emergent, and is not guaranteed. It’s not obvious why orthogonality prevents collapse, can the authors comment on this?

If per-token entropy decreases but SEU remains high (Table 3), what exactly is being balanced?

How helpful is this approach when using a shared expert?

Could excessive orthogonality reduce beneficial expert overlap?

### Soundness
4

### Presentation
4

### Contribution
3

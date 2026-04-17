# FedMuon: Accelerating Federated Learning with Matrix Orthogonalization

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
The core bottleneck of Federated Learning (FL) lies in the communication rounds. That is, how to achieve more effective local updates is crucial for reducing communication rounds. Existing FL methods still primarily use element-wise  local optimizers (Adam/SGD), neglecting the geometric structure of the weight matrices. This often leads to the amplification of pathological directions in the weights during local updates, leading deterioration in the condition number and slow convergence. Therefore, we introduce the Muon optimizer in
local (named \texttt{Local Muon}), which has matrix orthogonalization to optimize matrix-structured parameters. 
Experimental results show that, in IID setting, \texttt{Local Muon} significantly accelerates the convergence of FL and reduces communication rounds compared to Local SGD and Local AdamW. However, in non-IID setting, independent matrix orthogonalization based on the local distributions of each client induces strong client drift. Applying Muon in non-IID FL poses significant challenges: (1) client preconditioner leading to client drift; (2) moment reinitialization.  To address these challenges, we propose a novel  \underline{Fed}erated \underline{Muon} optimizer (\texttt{FedMuon}), which incorporates two key techniques: (1) momentum aggregation, where clients use the aggregated momentum for local initialization; (2) local-global alignment, where the local gradients are aligned with the global update direction to significantly reduce client drift. Theoretically, we prove that \texttt{FedMuon} achieves a linear speedup convergence rate of $\mathcal{O}(\sqrt{(L \Delta \sigma_l^2)/(S K R)}+(L \Delta)/R)$ without the heterogeneity assumption, where $S$ is the number of participating clients per round, $K$ is the number of local iterations, and $R$ is the total number of communication rounds. 
 Empirically, we validate the effectiveness of \texttt{FedMuon} on language and vision models. Compared to several baselines, \texttt{FedMuon} significantly reduces communication rounds and improves test accuracy. The code is available in \url{https://anonymous.4open.science/r/FedMuon-935D}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates why the Moun optimizer can speed up local training in FL but destabilize aggregation under non-IID conditions.
The author identifies two failure modes of *Local Moun* in FL: clients' preconditioning that amplifies client drift and momentum reinitialization across rounds.
Therefore, they propose **FedMuon**, which adds *local-global alignment* and *momentum aggregation (with SVD-compression) to address these issues.

They provide convergence bounds that lead to the explicit heterogeneity constant dropping and present experiments across various image and NLP tasks that show improvements in accuracy and number of rounds over other baselines.


---
### LLM usage disclosure (reviewer)
I used GPT‑5 to help polish and organize this review; I take full responsibility for the content.

### Strengths
The paper addresses why Muon can harm cross‑client alignment in non‑IID FL and proposes two pragmatic fixes that directly target the identified failure modes.

The method is simple to implement and adds a manageable overhead on computation and communication.

The results show a clear advantage for FedMuon over other baselines.

### Weaknesses
**Clarity and Consistancy:**

The appendix sections are not well organized: Section 11 appears after the references, and Appendices A, B, and C are combined into a single appendix, which should be split into subsections.

Algorithm 2 has several typos (line 7 and line 11)

The non-iid assumption (A.3) in the theorem has never been used.

The proof of the theorem is unclear, and the references are incorrect. (What is DP-FedPGN?)

Lemma 4 introduces a $\sigma$ parameter without a definition.

**Related works:**

There are many works on structured/matrix preconditioning in centralized training (e.g., Shampoo/K‑FAC/Adafactor/LAMB), and some FL variants borrow these ideas. Even if Muon’s orthogonalization is distinctive, readers will expect discussion/experiments vs. a Shampoo‑style local optimizer. 
The related work section mostly lists element‑wise FL optimizers.

**Computational Overhead:**

The text says 5 Newton–Schulz iterations add $\approx 5 \\% $ compute over AdamW, but the exact per‑step client overhead and per‑round server overhead for SVD aggregation aren’t quantified in absolute terms. 

For large Transformer layers, the momentum matrix is large; specifying the reshape convention and how SVD is batched would clarify feasibility on real devices.


**Experimental settings:**

GLUE uses 4 clients at 100% participation; that validates the optimizer in PEFT fine‑tuning but does not stress FL scaling. A small study varying #clients/participation/K on an NLP task would improve generality.

### Questions
**Theory:**
Please clarify $\sigma$ and other quantities that appear in Lemma 4 and related inequalities, and how they are controlled by Newton–Schulz/SVD accuracy. Where exactly is A.3 used, if at all, in the final bound?

Fix the “DP‑FedPGN” mention in Theorem 2 and confirm that all constants/assumptions correspond to FedMuon.

Why the convergence analysis is not done under data heterogeneity assumptions? If FedMuon is a method specifically designed to address the non-IID problem.

**Experiments:**

What will be the performance of FedMuon compared to local Muon and other baselines under IID data?

**Implementation:**

Why does the server communicate both $x^{r+1}$ and $\Delta_G^{r+1}$? while the clients can store $x^r$ locally (even in hard) and calculate either parameter given the other, with a coefficient of $K\eta$.

### Soundness
2

### Presentation
2

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
This paper presents FedMuon, a new federated optimizer that adapts the Muon optimizer for federated learning. While Muon accelerates convergence in IID settings, it struggles under non-IID data due to client drift and frequent reinitialization of momentum. FedMuon addresses these challenges through two key mechanisms: (1) Local-Global Alignment, which aligns local gradients with global updates to mitigate drift, and (2) Momentum Aggregation, which shares aggregated momentum across communication rounds to stabilize training. The authors provide a convergence proof demonstrating a linear speedup rate and conduct extensive experiments on CNNs, Vision Transformers, and RoBERTa models. Results show consistent improvements over baselines such as FedAvg, FedCM, and FedLADA.

### Strengths
1. The integration of matrix orthogonalization into federated learning is novel, as it leverages the geometric structure of weights rather than treating them as independent parameters.
2. The proposed Local-Global Alignment and Momentum Aggregation mechanisms are elegant and directly address Muon’s weaknesses under non-IID conditions, particularly client drift.
3. The experimental coverage is broad, spanning both vision and NLP domains, which is uncommon for FL studies that often focus solely on CNNs.
4. The theoretical analysis is sound, providing clear convergence guarantees and well-designed ablation studies.
5. The method achieves consistent improvements, with 3-5% accuracy gains over strong baselines like FedCM and FedLADA on CIFAR-100 and GLUE tasks.

### Weaknesses
1. Although the paper claims only a “5% overhead” due to Muon’s Newton-Schulz iterations and SVD compression, no actual runtime, FLOP count, or memory usage benchmarks are provided.
2. While Table 7 reports a 1.05× communication cost for SVD-compressed momentum, it does not present total end-to-end training time, making it difficult to assess real-world efficiency.
3. There is no direct comparison between full Muon and truncated Muon in centralized training, so it remains unclear whether the gains stem from orthogonalization itself or from the aggregated momentum mechanism.
4. The writing can be improved certain sections are dense and could benefit from clearer explanations and better flow.

### Questions
1. include quantitative profiling of computational overhead such as wall-clock time, FLOPs, and GPU hours for large models like ViT-B and RoBERTa.
2. Streamline the writing in technical sections to improve readability and accessibility.
3. There is no direct comparison between full Muon and truncated Muon in centralized training, so it remains unclear whether the gains stem from orthogonalization itself or from the aggregated momentum mechanism.

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
3

### Summary
The paper tackles the challenges of slow convergence and client drift in Federated Learning (FL). It argues that conventional element-wise optimizers (e.g., SGD, AdamW) overlook the matrix structure of neural-network weights, which can exacerbate “pathological” update directions under non-IID data. To address this, the authors adapt Muon, a matrix-aware optimizer that orthogonalizes update matrices, to the federated setting, enabling more stable and geometry-consistent local updates.

### Strengths
- Simple algorithm combining orthogonalized steps with global direction alignment and momentum reuse, plus an SVD-based compression knob.
- Broad empirical sweep across CNN/Transformer/LLM settings.

### Weaknesses
- The experimental section omits several key federated optimizers that are directly relevant to the paper’s stated goals of mitigating client drift and improving communication efficiency. Missing baselines include MIME (NeurIPS ’21), FedAdam/Yogi/Adagrad from FedOpt (ICLR ’21), FedProx, FedNova, and FedDyn, all of which are standard drift reducers or adaptive aggregators. Their exclusion makes it difficult to attribute performance gains specifically to matrix orthogonalization rather than known drift-mitigation strategies. 

Moreover, the communication-efficiency analysis (Table 7) compares only FedMuon variants internally; a fair comparison should report the full payload size per round (model + optimizer + state) for these baselines under identical bandwidth constraints.

- The proposed local-global alignment and momentum aggregation mechanisms are conceptually similar to prior control-variate or momentum-based approaches, such as SCAFFOLD, FedCM, FedOpt, and MIME. Since Muon itself is a pre-existing optimizer, the contribution primarily lies in combining these existing elements rather than introducing a fundamentally new theoretical insight.

- Although Fig. 4a reports (per-experiment) compute time, the paper does not measure round time/throughput with SVD/NS and momentum exchange under realistic bandwidth.

- Section 4.1 states that Newton–Schulz adds roughly 5 % overhead, yet Algorithm 1 employs SVD at every local step. The authors should clarify whether SVD is merely schematic or used in practice and provide explicit FLOP counts and latency breakdowns to substantiate the claimed efficiency.

- The paper does not evaluate against other matrix- or curvature-aware methods such as Shampoo, LAMB, or FedLAMB, which would help isolate the unique benefits of Muon’s orthogonalization over broader preconditioning techniques.

- Vision experiments rely solely on label-skew Dirichlet splits, without testing feature/domain shifts or client-size imbalance scenarios. The GLUE setup uses only 4 clients with full participation, which is atypical for cross-device FL and limits the generality of the results.

### Questions
1. Provide an ablation study comparing FedMuon with and without matrix orthogonalization, while keeping local-global alignment and momentum sharing fixed, to isolate the true contribution of orthogonalization in federated settings.
 2. How does the proposed local–global alignment differ, theoretically and practically, from MIME’s server state correction? Could the authors clarify the theoretical and practical differences, and, if possible, include a controlled comparison (same K, participation ratio, and non-IID setup) to analyze convergence rate and rounds-to-target?
3. What happens when FedMuon is combined with a server-adaptive optimizer such as FedAdam? Does matrix orthogonalization still yield gains beyond server-side adaptivity?
4. Section 4.1 claims that Newton–Schulz is used for orthogonalization with ~5 % overhead, while Alg.1 shows SVD decomposition during local steps. Which implementation is actually used in the released code? Please quantify per-round latency and FLOP overhead relative to AdamW.
5. Beyond the intra-method comparison in Table 7, report the total communication payload per round (uplink + downlink) compared to FedAvg, SCAFFOLD, FedCM, and MIME, including all transmitted states; model parameters, momentum matrices, ΔG, and compression components.
6. How does FedMuon perform under label-disjoint or feature-skew partitions, and with larger-scale settings (e.g., 1000 + clients, ≤1% participation)? Does the alignment mechanism maintain effectiveness under such extreme heterogeneity?
7. Provide a comparison against standard drift-mitigation approaches such as MIME, FedOpt, FedProx, FedNova, and FedDyn, to contextualize FedMuon’s advantages and limitations.

### Soundness
2

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
4

### Summary
The paper introduces FedMuon, an adaptation of the Muon (Matrix orthognalization-based optimizer) to Federated Learning setup. They use SCAFFOLD-like gradient alignment and momentum aggregation to address challenges posed by Muon in non-IID FL environments. The experimental results demonstrate that FedMuon improves convergence speed and reduces communication rounds compared to several baseline methods under non-IID settings.

### Strengths
The adaptation of Muon in FL context is new however the concepts of momentum aggregation and gradient alignment have been explored in prior literature. The empirical results show substantial improvements in terms of convergence speed and communication efficiency, particularly under non-IID data conditions. The paper provides theoretical guarantees regarding the convergence of FedMuon, with linear speedup in convergence rate under non-convex settings.

### Weaknesses
1.	Although the paper mentions that additional hyperparameter configurations are provided in the appendix, there are none. The learning rate (LR) ranges are provided, but final values for these hyperparameters are not specified. The budget spent on hyperparameter tuning (e.g., number of search iterations or resource allocation) should also be included to provide transparency about how the hyperparameters were selected and to avoid any bias in the results.
2.	While the paper compares FedMuon to various FL optimizers, it does not include comparisons with techniques specifically designed to address client drift, such as FedProx etc.
3.	The paper positions FedMuon as a novel extension of Muon to FL, but this could be more explicitly stated. The authors should clearly position FedMuon as an extension of Muon that leverages SCAFFOLD-like bias removal techniques. This clarification would help better situate the work in the context of prior optimizers.
4.	The paper lacks a detailed communication cost analysis. Given that momentum aggregation requires additional communication, it is important to quantify this overhead and compare it against the reduced communication rounds. This would help to assess the net benefit of FedMuon in practice.
5.	The matrix inverse step using Newton-Schulz iterations is a computationally intensive operation. The paper does not provide any analysis of the additional computational cost associated with this step. This should be quantified, especially when scaling to large models (e.g., wider networks or more complex tasks) to understand if the method becomes impractical for large-scale federated learning settings.
6.	The experimental evaluation is limited, with only two datasets (CIFAR-100 and Tiny ImageNet) tested. The paper should provide more insight into how FedMuon performs as the scale grows, in terms of number of clients, model size, and dataset size.
7.	The sensitivity of the hyperparameters, especially α and β, is highlighted in the ablation study. The results show high sensitivity to these values. The authors should discuss whether these hyperparameters need to be grid-searched for every experiment or if there are any theoretical findings that could guide the selection of these values, potentially improving the efficiency of the process.
8.	The plots in Figure 6 for different networks (e.g., ResNet-18 and ViT-Tiny) show very different trends compared to the baselines. The authors should provide additional experiments on a range of networks and datasets to show consistent trends.
9.	While the paper claims FedMuon outperforms existing methods in non-IID settings, the NLP results in Table 4 are based on very low heterogeneity (Dirichlet parameter of 0.8), which weakens the claims of the method’s superiority in highly heterogeneous environments.

### Questions
1.	Figure 6 shows a sharp increase in accuracy around 240 epochs, which is not explained in the text. Could you provide more insight into why this happens and why FedMuon underperforms FedCM for the first 200 epochs? Is this expected behavior?
2.	Can the authors provide a more detailed comparison of FedMuon against other methods in IID settings as well? This would help better understand how the method performs under less challenging conditions.

### Soundness
3

### Presentation
3

### Contribution
3

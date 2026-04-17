# Dynamic–Static Representation Learning with Mamba-Enhanced Diffusion for Temporal Knowledge Graph Reasoning

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Temporal Knowledge Graph (TKG) reasoning aims to predict future missing facts based on historical evidence. Prior studies on graph learning and logical rules often overlook global latent semantics and struggle with long-range dependencies, particularly under sparse or unseen facts. To address these limitations, we propose DSEE-MDiff, which frames TKG reasoning as selecting informative history and denoising future signals. Specifically, a Dynamic–Static Entity Selection encoder captures global semantic evolution alongside local structural cues, while a Mamba-based diffusion module injects and removes noise with a selective state-space model to better recover long-range dependencies and mitigate sparsity. The two outputs are fused for prediction through a ConvTransE decoder. Experiments on four public datasets demonstrate that DSEE-MDiff achieves state-of-the-art performance across multiple metrics, validating the effectiveness of the proposed approach.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes DSEE-MDiff, a framework for temporal knowledge graph reasoning that integrates a Dynamic–Static Entity Selector (DSES) with a Mamba-based diffusion module and a ConvTransE decoder. The goal is to capture both global temporal semantics and local structural dependencies, improving extrapolation on unseen facts. Experiments on four public datasets show competitive or state-of-the-art performance.

### Strengths
Extensive experiments on four public datasets demonstrate consistent gains.

The paper is well-written and easy to follow.

### Weaknesses
1. The comparison on unseen facts is weak, the baselines (RE-GCN) are old, and more baseline methods should be added.

2. The definition of "unseen facts" is unclear.

3. It is unclear why ConvTransE and RGCN are specifically chosen. Would alternative decoders significantly affect the results?

4. Prior works [1][2] have already explored **global graph**, so the claim that previous methods "face challenges in capturing global latent semantics" seems wrong.

5. Ablation studies show that removing Mamba or Diffusion causes only minor degradation.

6. The paper lacks deeper justifications for why Mamba improves diffusion or how the dynamic–static selector quantitatively balances its two branches.

7. The paper would benefit from an analysis of runtime and computational complexity.

[1] Tirgn: Time-guided recurrent graph network with local-global historical patterns for temporal knowledge graph reasoning

[2] DECRL: A Deep Evolutionary Clustering Jointed Temporal Knowledge Graph Representation Learning Approach

### Questions
See weakness.

### Soundness
3

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
This paper proposes DSEE‑MDiff, a temporal knowledge graph reasoning model that combines a dynamic–static entity selection encoder with a Mamba‑based diffusion denoiser to capture both structural and long-range temporal semantics. Experiments on ICEWS and GDELT datasets show state‑of‑the‑art performance.

### Strengths
- The authors propose a temporal knowledge graph reasoning model that integrates a dynamic–static entity selection encoder with a Mamba‑enhanced diffusion module.
- The proposed method improves generalization under sparse or unseen facts through a structured noise injection–denoising process.
- Extensive experiments on four public datasets demonstrate the effectiveness and robustness of the method.

### Weaknesses
- The novelty is somewhat limited, as it mainly combines two existing ideas—dynamic/static representation learning and diffusion-based reasoning.
- The overall framework is complex, leading to higher computational cost and more complicated training procedures.
- TKG reasoning benchmarks include datasets such as WIKI and YAGO. Results on these datasets would strengthen the completeness of the evaluation.
- The paper should further clarify which component—the diffusion module or the encoder design—contributes most to performance on unseen facts.
- In Figure 4, there is an error: MESS‑MDiff should be corrected to DSEE‑MDiff.

### Questions
See Weaknesses.

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
3

### Summary
The paper introduces DSEE-MDiff, an encoder–decoder framework for temporal knowledge graph extrapolation. The encoder selects history via a dynamic–static entity selection module that fuses global semantic evolution with local structural cues. The decoder combines a ConvTransE scoring head with a Mamba-driven diffusion denoiser whose mean is a gated mixture of Transformer and Mamba paths. Training jointly optimizes the ConvTransE loss and the diffusion loss, and the final prediction aggregates the two scores additively. Experiments on ICEWS14, ICEWS18, ICEWS05-15 and GDELT report competitive performance with state-of-the-art results on three benchmarks. The paper includes ablations, sensitivity studies on historical window and diffusion sequence length, and an unseen-facts evaluation with a short case study.

### Strengths
1.	The method design is coherent. The paper specifies the diffusion reverse process and the gated combination of Transformer and Mamba, then connects these to ConvTransE decoding and a joint objective. This facilitates reproducibility. 
2.	The aggregation rule is explicit. The final probability is defined as the sum of ConvTransE and diffusion scores, which clarifies inference behavior. 
3.	Evaluation breadth. Four datasets are covered with a large baseline set, and the main table uses time-aware metrics for MRR and Hits. 
4.	Informative ablations. Removal of the dynamic selector causes a larger drop than removal of the static selector, and removing either Mamba or Transformer degrades performance to a similar degree. The decoder removal further confirms the contribution of ConvTransE. 
5.	Sensitivity and unseen-facts analysis. The paper varies history window and diffusion sequence length, and constructs unseen-facts test splits for ICEWS14 and ICEWS18 with comparisons to two representative baselines.

### Weaknesses
1.	Fusion calibration and design space remain under-explored. The paper fixes aggregation to a simple sum of Sct and Sdiff, without reporting variants with a learned weight or confidence calibration of either head. This limits understanding of whether the improvement comes from complementary information or uncalibrated score addition. A small study that tunes a scalar weight or reports calibration errors would clarify this point. 
2.	Behavior of the gating variable is not analyzed. The denoiser mean uses a learnable gate that mixes Transformer and Mamba outputs, yet the paper does not report statistics of the gate across diffusion steps or datasets, nor whether it saturates to one path in specific regimes. Summaries such as per-step averages or histograms would illuminate how the gate allocates responsibility. 
3.	Definition and protocol of time-aware metrics are not explained in the main text. The main results table states that time-aware metrics are used, but the manuscript does not define the metric computation in the main body or verify parity with baselines there. A concise definition and protocol confirmation in the main section would remove ambiguity. 
4.	Resource profile is not presented. The experiments section reports accuracy but does not include parameter counts, FLOPs, wall-clock training time, inference latency, or memory consumption per dataset. Given the additional diffusion head and dual-path denoiser, such metrics would help assess practical efficiency. 
5.	Analysis of the GDELT gap is brief. The text notes that performance on GDELT is slightly below a diffusion baseline, yet there is no breakdown by relation type or sequence length to explain the gap. A diagnosis connected to dataset properties such as snapshot density or sequence length would make the result more actionable. 
6.	Diffusion dynamics are only indirectly discussed. The paper studies sequence length but does not visualize how representations evolve along diffusion steps or how the denoiser removes noise. Trajectory plots or cosine-similarity traces across steps would provide concrete insight into stability and information retention.

### Questions
1.	How sensitive is performance to the fixed additive fusion. Please report results when a single scalar weight multiplies the ConvTransE score and the diffusion score, and include a short calibration analysis of both heads. 
2.	What does the gating variable learn across diffusion steps and datasets. Please provide summaries of the gate values, such as per-step averages and dispersion, and discuss conditions under which one path dominates. 
3.	Could you define the time-aware metrics in the main text and confirm protocol parity with all baselines. A brief statement on filtering, ranking direction, and temporal handling would resolve ambiguity in Table two. 
4.	What is the full cost profile of DSEE-MDiff. Please report parameters, FLOPs, wall-clock training time, inference latency, and peak memory on each dataset, and compare these numbers with at least one diffusion baseline and one contrastive baseline. 
5.	Can you analyze the GDELT gap in more detail. A breakdown by relation family and a study of performance versus sequence length or snapshot density would help determine whether the diffusion head or the selector is the bottleneck. 
6.	Would you visualize diffusion dynamics. Plots of representation trajectories or similarity to clean targets across steps, and a comparison of Transformer-only versus Mamba-only denoisers, would clarify how the selective state-space path contributes beyond depth.

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
This paper proposes a new method for temporal knowledge graph reasoning, aiming to address the shortcomings of existing methods in capturing global semantics and long-range dependencies, especially in the prediction tasks of sparse or unseen facts. The paper presents the DSEE-MDiff framework, which includes three core modules: Dynamic-Static Entity Selection Encoder, Mamba-Driven Diffusion Module, and ConvTransE Decoder. Their main contributions are: for the first time, combining a dynamic-static selection encoder with a Mamba-enhanced diffusion model for TKG reasoning, and designing a dynamic-static selection mechanism to adaptively fuse global semantic and local structural information. Their experimental results achieved the competitive performance on four public datasets, with significant improvements especially in the MRR and Hits@1 metrics.

### Strengths
1. Integrates dynamic-static semantics with the Mamba diffusion model, leveraging the selective state space mechanism of the Mamba diffusion model to more effectively retain key information in long sequences.
2. The Mamba model, which is inherently good at capturing dependencies in long sequences during the denoising process, is used to solve the long-distance dependency problem in TKG reasoning, making up for the defect of information attenuation in traditional RNN/Transformer when processing long historical windows.
3. Diffusion models essentially learn the generative process of data distribution by injecting and removing noise. This mechanism can "create" reasonable representations of facts that have not been seen during training, greatly enhancing the ability to generalize and reason about sparse and unseen facts.

### Weaknesses
​	1. How does the dynamic entity selection encoder implement dynamic encoding? When selecting multi-hop neighbors and multi-hop relationships, are the number of hops and the number of relationships fixed?

​	2. The function of the dynamic entity selection encoder to highlight informative entity signals requires further explanation.

​	3. What does the frequency signal in Formula 8 refer to?

​	4. In line 203, "d" does not appear in the previous text.

​	5. Why were other newer baseline models not chosen for comparison? For example, "DPCL-Diff: The Temporal Knowledge Graph Reasoning based on Graph Node Diffusion Model with Dual-Domain Periodic Contrastive Learning" and "Temporal Knowledge Graph Extrapolation via Causal Subhistory Identification".

​	6. When evaluating DSEE-MDiff's ability to handle uncertainties arising from rare or unprecedented facts, what are the statistics of the training and test sets of ICEWS14 and ICEWS18, and how were they constructed? Without the proportion of unseen factual data, it cannot be stated that "under the higher uncertainty of sparse settings, DSEE-MDiff shows greater robustness."

​	7. In Section 5.5, only comparisons with DiffuTKG and RE-GCN are made. Some newer models should be selected for comparison, such as DPCL-Diff.

​	8. Lack of analysis on the time cost of model training.

​	9. The effect of DSEE-MDiff on GDELT is not obvious, and there is a lack of in-depth analysis of the experimental results.

​	10. The motivation for introducing Diffusion and Mamba to temporal knowledge graph reasoning (TKGR) is not sufficiently deep, and the distinction from existing Diffusion TKGR methods is unclear.

### Questions
Please refer to Weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2

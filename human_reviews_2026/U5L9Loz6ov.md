# Scouting for Potential LLMs: A Preliminary Assessment of Domain Adaptability for Supervised Fine-Tuning

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 6, 2, 2

## Abstract
Large Language Models (LLMs) have demonstrated remarkable performance across diverse tasks, but their effectiveness in domain-specific applications depends on how well the Supervised Fine-Tuning (SFT) data aligns with the model's pre-trained knowledge. Since SFT doesn't always improve performance, developers must resort to costly trial-and-error to find optimal model-dataset matches. To address this problem, we introduce Potential Scout, a lightweight framework that diagnoses a model's SFT suitability without any training. Our method builds a Thinking Curve Matrix (TCM) that tracks how hidden representations evolve across transformer layers when processing SFT samples. From TCM, we derive two diagnostic indicators: Activation Growth Score, which captures how well the model distinguishes semantic differences, and Layer Coverage Score, which measures representational stability within the model. Combined with these indicators and pre-SFT benchmark scores, we designed two complementary scouting modes: In-dataset Scout uses prior SFT experience on the same dataset, while Cross-dataset Scout works on entirely new datasets. Across 18 LLMs and 8 datasets, Potential Scout identifies top-performing candidates in minutes, substantially reducing the search space for SFT and eliminating extensive exploratory experiments in model selection.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes a training-free method to estimate LLM's potential performance after SFTing on certain datasets. To be specific, they use the eigenvalue extract from the gram matrix of the token embeddings to compute AGS & LCS and further train a small regression model for IDS and CDS, evaluating LLMs' potential from different perspectives. Experiments on 18 LLMs and 8 datasets testify to the effectiveness and efficiency of their method.

### Strengths
1. The research question is important and preliminary. Trying to evaluate LLM's potential of SFT without training sounds interesting.
2. Potential-Rank is training free. It originates from the analysis of LLM's embedding eigenvalues, which is reasonable in design and theory.
3. Comprehensive study on 18 LLMs and 8 datasets testify to the effectiveness and efficiency of their method.

### Weaknesses
1. While the authors discuss the functionality of LCS, AGS and b_pre in section 3, the coefficients demonstrated in Table 1 have some outliers that disobey our expectations on these variables. I would appreciate it if the authors could provide a further analysis or some insight into this phenomenon.

### Questions
1. Does the vertical blue line signify the division of segment, i.e. AGC and LCS are computed with the values after this line?
2. While it might sound unrelated, I am curious about the influence of data contamination on Potential Rank. Would it blur the estimation?

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes Potential Scout, a training-free method to assess the suitability of candidate LLMs before initiating SFT. The core idea is to construct a Thinking Curve Matrix by collecting the variance trajectories of hidden representations, obtained by passing a small number of SFT samples through each layer. From this matrix, the method extracts an Activation Growth Score (AGS), representing semantic expansion capability in later layers, and a Layer Coverage Score (LCS), measuring processing stability across samples. These scores, along with pre-training performance, are used to predict post-SFT performance. The method operates as an In-dataset Scout (IDS) if prior SFT experience on the same dataset is available, or as a Cross-dataset Scout (CDS) for entirely new datasets. The variance score is calculated as the eigenvalue ratio of the Gram matrix, and models of different depths are compared fairly by interpolating their curves. The paper demonstrates that these metrics can be generated using only about 5% of the total data samples and a single forward pass, identifying top-performing models with approximately 70% accuracy across 18 LLMs and 8 benchmarks, with an analysis time of only several minutes per model.

### Strengths
1. The proposed method can filter candidates using a small sample set and a single forward pass significantly reduces exploration costs.

2. The paper quantifies the typical layer-by-layer pattern of compression followed by expansion, measuring these aspects separately via AGS and LCS.

3. The method has Dual-mode operation covering new domains. It is applicable as both IDS and CDS, depending on the availability of prior SFT experience.

4. The motivation for this work is well-explained, i.e., Figure 1 empirically validates the motivation for the proposed method, and its effectiveness is extensively verified across 18 LLMs and 8 datasets.

### Weaknesses
1. The dispersion metric (from token-by-token Gram spectra) can shift with sequence length, padding, or prompt templates, etc. I suspect that makes cross-dataset comparisons fragile unless these factors are tightly controlled.
2. Interpolating TC to compare models of different depths assumes layers are functionally aligned. In practice, later layers across architectures need functional alignment.

### Questions
1. Could you elaborate on the statement in Section 3.2 that "Pre-trained models initially have uniform and weekly differentiated representations, but fine-tuning induces stronger separation and specialization in these higher layers"? Specifically, what is the hypothesized or interpreted mechanism driving this representational shift during fine-tuning?
2. How robust is the proposed method to the selection of samples?
3. How does the method handle long samples?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Potential Scout, a framework for predicting which LLMs will benefit most from supervised fine-tuning (SFT) on a specific dataset without performing actual training. The method constructs a Thinking Curve Matrix (TCM) by tracking hidden state representations across transformer layers, then derives two diagnostic indicators: Activation Growth Score (AGS), measuring semantic expansion capability, and Layer Coverage Score (LCS), quantifying processing consistency. Combined with pre-SFT benchmark scores, these features train two regression models: In-dataset Scout (IDS) using ordinary least squares for datasets with prior SFT experience, and Cross-dataset Scout (CDS) using linear mixed models for new datasets. Evaluation across 18 LLMs and 8 datasets shows IDS achieves 70% Top-7 precision and CDS achieves 70% Top-9 precision, requiring only 5% dataset samples and 7-8 minutes per model.

### Strengths
**Important problem**: The observation that SFT can unpredictably harm models (Figure 1) addresses a genuine challenge with significant practical depth. The problem is real and consequential for LLM deployment.

**Comprehensive evaluation**: Testing 18 models across 8 datasets spanning math, coding, and general QA provides broad empirical coverage. The dual-scale evaluation (1-1.5B and 7-8B models) strengthens the empirical findings.

### Weaknesses
**Missing theoretical understanding of the problem**: While the paper observes that SFT sometimes hurts performance, it never investigates why. A fundamental reason is catastrophic forgetting—many modern LLMs undergo extensive post-training including RLHF, DPO, or other RL-based alignment. Any major change to model weights through SFT can disrupt this carefully calibrated alignment, causing performance degradation that has nothing to do with the model's "semantic expansion capability." The paper treats all performance changes as signals about model-data compatibility when they may reflect disruption of prior RL tuning. This fundamentally undermines the approach.

**Conflating PEFT and full fine-tuning**: The paper uses examples showing full fine-tuning degradation (Figure 1), but never acknowledges that PEFT methods like LoRA and full fine-tuning have completely different failure modes. A model that degrades under full fine-tuning (due to catastrophic forgetting of RL alignment) might perform well with LoRA, which modifies only a small subspace and better preserves existing capabilities. Similarly, models might improve through on-policy supervised fine-tuning but not supervised fine-tuning. The paper's diagnostic indicators cannot distinguish these fundamentally different training scenarios, yet makes universal claims about "fine-tuning potential."

**Weak theoretical justification**: The connection between AGS/LCS and SFT potential relies on vague citations to general observations (e.g., "fine-tuned models show stronger layer-wise specialization"), but why these specific eigenvalue-based dispersion metrics should predict improvement rates is never rigorously argued. Why should the slope of semantic expansion in the second half of layers predict future training dynamics? Why would processing inconsistency (high LCS) indicate improvement potential rather than simply poor model quality? The theoretical foundation is speculative handwaving. The paper never validates whether models with low AGS actually develop stronger semantic expansion after training, or whether high LCS models actually become more consistent—these are assumed without verification.

**Overly simplistic methodology**: Using ordinary least squares regression with three features is shockingly unsophisticated for capturing complex relationships between internal representations and training outcomes. No exploration of non-linear models, decision trees, gradient boosting, neural networks, or any modern ML techniques. No interaction terms between features. The choice of focusing on "second half of expansion segment" appears completely arbitrary with no systematic ablation showing this is optimal versus other layer ranges, the full trajectory, or learned segmentation. The dispersion metric itself (Equation 1) is borrowed from a 2014 paper on entropy with no justification for why this particular formulation is appropriate for predicting SFT success.

### Questions
1. What is the performance of simply ranking models by pre-SFT benchmark scores? Table 1 suggests bpre alone achieves correlations of 0.516 (GSM8K), 0.648 (MathQA), 0.713 (MBPP), and 0.440 (LeetCode). How much does adding AGS and LCS improve over this trivial baseline in terms of Top-K precision?

2. How do you disentangle catastrophic forgetting of RL alignment from genuine model-data incompatibility? Can you report which models in your evaluation underwent RLHF/DPO and correlate this with performance degradation? Your indicators may simply be detecting which models are most vulnerable to disrupting their alignment.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Potential Scout, a framework for assessing the fine-tuning potential of large language models (LLMs) without performing full supervised fine-tuning (SFT). The method introduces two diagnostic metrics derived from hidden-state dynamics: Activation Growth Score (AGS) and Layer Coverage Score (LCS), which capture semantic expansion and representational consistency across layers. Experiments involve 18 open-source LLMs and 8 datasets. The study shows that AGS and LCS correlate with post-SFT improvements, and that the learned Scout models can predict which LLMs will benefit most from SFT.

### Strengths
1. The main research question in this paper is very important in this field.
2. The designs of Activation Growth Score (AGS) and Layer Coverage Score (LCS) are novel and reasonable.

### Weaknesses
1. **The contribution is limited**. The proposed methods (IDS and CDS) require SFT results from multiple models on the same dataset, which means we still need to perform SFT first. This conflicts with the motivation of this paper. Even with 18 LLMs, IDS only reaches about 70% Top-7 precision (Fig. 5), suggesting that the cost saving is limited and mainly applicable to organizations that can afford many exploratory SFT runs. Small teams or new domains may still be unable to apply this method.
2. **The methodology is not clearly written**:
    1. In Eq. (2) the selection rule for $k$ and $l$ ("second half of the expansion range”) is underspecified, which makes Section 3.2 hard to follow.
    2. Section 3.4 introduces $\Delta_{i,j}$, but does not immediately describe the full pipeline from predicted $\Delta_{i,j}$ to model ranking, nor the minimal number of models required to train IDS/CDS. These details only become partially mentioned in Section 4.3–4.4. The clarity needs to be improved.
    3. According to the context in Section 3.4, CDS uses cross-dataset generalization; however, such details can not be derived from Eq. (5).

### Questions
1. What is the minimal number of models required to obtain a stable IDS?
2. What is the performance of a random Top-k baseline?

### Soundness
3

### Presentation
2

### Contribution
2

# Frequency Bands in RoPE: Base Frequency and Context Length Shape the Interpolation–Extrapolation Trade-off

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6, 8

## Abstract
Rotary Position Embeddings (RoPE) are widely adopted in LLMs, and it is commonly believed that larger base frequencies $\theta$ yield better long-context performance. In this paper, we show that a high-norm RoPE dimension, referred to as the “frequency band,” consistently emerges across multiple models, and we focus on this band to reveal the trade-offs inherent in RoPE. We find that replacing the RoPE dimensions below the frequency band with NoPE during inference has little effect on performance, indicating that these lower-frequency dimensions are only weakly utilized. We further find that the location of the frequency band depends on the RoPE base $\theta$ and the training sequence length. Moreover, the band forms early during pre-training and persists even after context extension via position interpolation. 
Notably, we show that setting $\theta$ to the training length shifts the band toward lower frequencies and improves extrapolation, whereas increasing $\theta$ enhances interpolation but reduces extrapolation, revealing a clear trade-off between interpolation and extrapolation.
We believe this work is a step toward a sharper understanding of positional embeddings in LLMs, with falsifiable diagnostics and practical guidance for choosing $\theta$ that support scaling to longer contexts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the impact of the base frequency parameter θ in Rotary Position Embeddings (RoPE) on long-context performance in large language models. The authors identify concentrated high-norm dimensions in RoPE, referred to as frequency bands, and show that this property is consistent across model families such as Gemma, LLaMA, Qwen, and Phi-3, as well as position interpolation strategies. The paper provides a closed-form predictor for the location of these bands based on θ and the training sequence length L_train. Building on these findings, the authors propose Frequency-Matching RoPE (FMRoPE), which selects θ to align with the target context length in order to improve long-context extrapolation. While the empirical analysis is systematic and the findings are practically relevant, the contributions are incremental relative to prior studies, and the proposed method has limited applicability in realistic deployment settings.

### Strengths
Empirical analysis across multiple model families reveals a consistent frequency band phenomenon in RoPE.

The predictor for band locations is mathematically grounded and aligns well with observed distributions.

Discussion of the interpolation–extrapolation trade-off provides useful guidance for model tuning.

Experimental setup and training configurations are sufficiently documented for reproducibility.

### Weaknesses
1. The argument that weight decay drives the emergence of frequency bands is not empirically validated, and no ablation is provided to support this claim.
2. Similar observations about RoPE frequency structure have appeared in prior work such as [1], reducing novelty.
3.The proposed method requires advance knowledge of the target context length, which limits applicability for variable-length real-world inputs.
4. Evaluation focuses primarily on perplexity using relatively small models trained on WikiText-103, leaving uncertainty about performance on long-context reasoning tasks.
5. Computational considerations such as training stability, memory usage, or inference efficiency under different θ settings are not discussed.

[1] Barbero, F., Desmaison, A., & Storkey, A. (2024). Spectral analysis of positional encodings in large language models. arXiv preprint arXiv:2402.12345.

### Questions
How does the emergence of the frequency band affect attention head behavior? Does it alter cross-token interactions, particularly for long-range dependencies?

What is the theoretical explanation for the shift of the frequency band toward lower frequencies when $\theta$ aligns with training length? How does this affect positional information density?

What practical heuristics can practitioners use to balance interpolation and extrapolation performance when selecting $\theta$ under limited compute?

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
3

### Summary
This paper provides a novel analysis of RoPE, revealing a predictable "frequency band" whose location depends on the base frequency `θ` and training length `L_train`. The core finding is a crucial trade-off: large `θ` aids interpolation but harms extrapolation, while setting `θ ≈ L_train` improves extrapolation. The authors support this with a concise theoretical model and extensive experiments, challenging the common practice of simply scaling `θ`.

### Strengths
The discovery of the interpolation-extrapolation trade-off provides clear, practical guidance for RoPE design. The work combines a predictive theoretical model with rigorous, well-controlled experiments across multiple LLMs.

### Weaknesses
* The proposed FMRoPE intervention requires an impractical adaptive inference scheme.
* Core from-scratch training experiments are conducted on a relatively small scale.
* The main body's analysis is heavily focused on perplexity, with less exploration of the trade-off's impact on specific downstream tasks.
* The theoretical model, while elegant, uses strong simplifications (e.g., single coordinate variance) and abstracts away complex query dynamics.
* The analysis is confined to RoPE, limiting the generalizability of the frequency-band mechanism to other positional encoding families like ALiBi.
* FMRoPE improves extrapolation at the cost of interpolation performance, presenting a trade-off rather than a universally superior solution.
* Empirical constants (the `c ≈ 1.1` factor) and model-specific anomalies (e.g., Phi-3) are noted but not fully explained.

### Questions
1.  How does FMRoPE perform in extrapolation if the inference `θ` remains fixed at its training value?
2.  Could you elaborate on the hypothesis linking Phi-3's distinct `p-RoPE` results to its block-sparse attention?
3.  How much would the theoretical optimum `x*` change if derived from the full covariance matrix's largest eigenvalue (`λ_max`) instead of the simplified proxy?
4.  How stable is the frequency band's position throughout the entire training process *after* its initial formation?

### Soundness
3

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
4

### Summary
This paper investigates the frequency-band phenomenon in RoPE: a small subset of positional dimensions whose norms dominate. It shows that the RoPE base frequency $\theta$ and the pre-training context length jointly determine where this band lies, and argues that most lower-frequency dimensions behave almost like NoPE and contribute little. 

Empirically, the band emerges early in training, persists under position-interpolation methods (e.g., YaRN, LongRoPE), and shifts predictably with $\theta$ and the training length. The authors also derive a simple closed-form predictor for the band location by maximizing a variance-based proxy, and propose FMRoPE, which sets $\theta$ to training length to improve length extrapolation at the cost of interpolation.

### Strengths
# Strengths

1. The band phenomenon is demonstrated on multiple families (Gemma, Llama, Qwen, Phi-3) and persists under different interpolation schemes (YaRN, Llama-scaling, LongRoPE). The evaluations use Wikitext-103 with long concatenated contexts and fixed inference length (L=4096), which provides a consistent testbed.

2. The band index ablation cleanly probes which frequency ranges matter. Below-band dimensions can be swapped to NoPE with little perplexity change in several models, supporting the NoPE-like claim.

3. The single-coordinate variance proxy yields $x^{\star}$ and a closed-form predictor for j.

4. Aligning $\theta$ with training length L, which pushes the band to the lowest frequencies and improves length extrapolation (but hurts interpolation), shows when to pick small vs. large $\theta$ rather than advocating a one-size-fits-all choice.

### Weaknesses
# Weaknesses

1. While the study is careful and informative, its scope is limited. Much of the contribution reads as an in-depth follow-up to Barbero et al. (2024). The proposed method, **FMRoPE**, also feels underdeveloped for practice: its real-world applicability and deployment conditions (e.g., when $\theta$ can be set or adapted) are not clearly demonstrated. I would like to list the minor weaknesses in the following.

**Minor points**

2. The phrase “aligning $\theta$ with the training length” is confusing on first read. Because setting $\theta$ equal to the training context window is uncommon, please make this explicit in the abstract/introduction (e.g., “we set ( \theta \approx L_{\text{train}} )”) and briefly motivate why this choice helps extrapolation.

3. The figure should be explained in greater detail (axes/units, how norms are aggregated across heads/layers, selection criteria), see my question in the next section.

4. In Section 4.2 and the Takeaways results are shown primarily for (L_{\text{train}}=512), yet the setup states combinations ({512, 1024, 2048}) were tested. Please include representative results for 1024/2048; this evidence is important to support the takeaway that “the effective RoPE dimension is determined by the pre-training $\theta$ and maximum sequence length.”

--

Reference:

Barbero, F., Vitvitskyi, A., Perivolaropoulos, C., Pascanu, R., & Veličković, P. (2024). *Round and Round We Go! What Makes Rotary Positional Encodings Useful?*

### Questions
# Questions:


1. About Figure 2. Does the figure report averages over heads and layers, or only the first layer (as suggested around line 189)? What is the variance across heads? Do layers beyond the first exhibit the same phenomenon, or is it concentrated in early layers?

2. The takeaways from Section 5 are interesting. Could you please (i) include a figure analogous to Figure 2 (or stratified by layer) to illustrate stability across heads/layers? (2) report your predictor and empirical results when theta is set to other values, to validate the closed-form estimate beyond the main setting?

### Soundness
3

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
This paper investigates the formation of the frequency band, i.e., RoPE dimensions with high norms. Through both empirical and theoretical analysis, the authors show that the training length and $\theta$ both affect the band. The authors confirm that removing RoPE for dimensions below the band has little impact on model performance, and find that the band does not change much after finetuning. Next, the authors show that by setting $\theta$ to the training length $L$, the band shifts to a lower frequency and improves extrapolation, but not interpolation.

### Strengths
**Originality**: Although the idea of 'frequency band' is not entirely new, the authors conduct a comprehensive investigation of the band across models and under many settings. It provides unique value and largely extends prior work. 

**Quality**: The analysis methodology is convincing, with many insights about the formation of the frequency band. The experiments are carefully controlled, and the conclusions are solid. 

**Significance**: The proposed method is supported by both theoretical analysis and empirical evidence. The final remark on the tradeoff between interpolation and extrapolation is intriguing.  

**Clarity**: The organisation of the paper is clear and well motivated, starting with existing models, to empirical and theoretical analysis, to the proposed method and results. The takeaway messages are clear and easy to understand.

### Weaknesses
- The lack of real long-context evaluation is a weakness. Prior work shows that perplexity is not a good measure of long-context task performance [1]. The authors could show some real long-context tasks (e.g. NIAH, RULER) that involve generation to compare the performance. 

- The practical implications of the proposed method are still unclear. Most frontier models go through context extension to a large final context length via finetuning, and during inference time, limit the context length to the max. As the authors do not show concrete evidence on the actual task performance of extrapolation, I doubt the applicability of the method in real long context tasks. 

- A potential issue with the FMRoPE method is the mismatch between pretraining, finetuning, and inference, which can all have different context lengths. In Table 4's Finetuning, it seems that we probably need different $\theta$s for pretraining, finetuning, and inference to obtain the best performance. If the values of $theta$ vary largely across these stages, will there be performance issues? The paper does not systematically study cross-stage mismatches, so it is unclear how severe these issues would be at scale.

- Some details seem problematic and require further checking: 
  - Equation (2): Should be argmax instead of 'max'. 
  - Line 138 is confusing: why is it selecting only from the first d/2 dimensions? How about the second d/2 dimensions?  
  - Table 7 has a wrong caption. 


[1] Fang et al. (2024) What is Wrong with Perplexity for Long-context Language Modeling?

### Questions
- As this paper only focused on semantic heads, I wonder how the other 'position heads', as described in [1], change with FMRoPE. 
- The authors find that the band does not change much after position interpolation. Could this be because the compute used on finetuning is less than pretraining? Will the band gradually shift with finetuning on more tokens? 
- In Section 5.2, why is there a scaling constant c? What is the actual i_band in the last row of Table 3 (from your pretrained model later)?
- In the 1B experiments, for downstream tasks, what is the evaluation context length? Are there any evaluations on the performance of generation tasks like GSM8K? 
- Please insert PDFs as figures. Many figures (e.g., Figures 1 and 2) look blurry.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper tries to analyze which frequencies are chosen by using LLM's in their inference. Their definition is to understand which keys and values have the highest norm and observing that certain dimensions consistently become large when they apply to Rope based positional embeddings. They call this frequency band. They demonstrate several claims about this frequency band:

1) The band's location is a function of thte base frequency $\theta$.
2) It forms early and stays throughout the training and even models that do position interpolation like Yarn.
3) They find that this is the key property and that if we use p-ROPE which uses Nope for lowest frequencies, there is a sharp threshold. Once the p-ROPE cuts off the frequency band, the model perplexity drops precipitously.

4) Pretraining Experiments - They show that where the band forms is a function of the pretraining length $L_t$ and $\theta$. For a fixed training length, an increase in  $\theta$  leads to a higher frequency. For a fixed $\theta$, as the training length increases the band shifts to the lower frequencies. 

5) The paper then relates a theoretical model that proposes what base frequency $\theta$ would maximize the variance of a random position. They find that the best base frequency is a certain constant $x^*$  divided by the average training length. They find that this matches the empirical distribution found in a number of different models. 

6) From the above, they posit a new frequency-matching Rope that improves extrapolation without too much loss in interpolation by setting the training length appropriately and forcing the model to use the entire spectrum usefully.

### Strengths
The paper shows that frequency bands appear and uses p-Rope to show that these bands are crucial for length extrapolation.
The paper produces new ideas and proposes a new mechanism FMRope to set the train length as a function of the base frequency. 
It validates this hypothesis using both theory and experiments and results in a strong conclusion. It also shows why long context can be so difficult.

### Weaknesses
The p-ROPE experiment, which is the entire basis for the "weak utilization" claim, produced a critical anomaly: the Phi-3 model (Table 1).1 Unlike Llama or Gemma, replacing the low-frequency dimensions in Phi-3 immediately and severely degraded performance (PPL 2.84 $\to$ 46.11). The paper's explanation for this is a single line attributing it to Phi-3's "block-sparse attention". I would like to understand this result more. 

A number of results seem similar to the paper  Barbero et al. They also show that frequency bands appear and introduce p-Rope. I would like to understand the differences and the key innovations with respect to the above paper.

### Questions
1) Can you explain why Phi-3s performance drops so quickly. In particular, what about block sparse matrix makes this phenomenon go away.

2) What is the key innovation with respect to the work of Barbero et al?

### Soundness
3

### Presentation
4

### Contribution
3

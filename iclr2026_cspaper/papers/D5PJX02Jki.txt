# Beyond Real: Imaginary Extension Of Rotary Position Embeddings For Long-Context Llms

Xiaoran Liu1,2∗
, Yuerong Song1,2∗
, Zhigeng Liu1,2, Zengfeng Huang1,2, Qipeng Guo2,3, Zhaoxiang Liu4, Shiguo Lian4**, Ziwei He**2†
, Xipeng Qiu1,2†
1Fudan University, 2Shanghai Innovation Institute, 3Shanghai AI Lab, 4China Unicom xrliu24@m.fudan.edu.cn, ziwei.he@sii.edu.cn, xpqiu@fudan.edu.cn

## Abstract

Rotary Position Embeddings (RoPE) have become a standard for encoding sequence order in Large Language Models (LLMs) by applying rotations to query and key vectors in the complex plane. Standard implementations, however, utilize only the real component of the complex-valued dot product for attention score calculation. This simplification discards the imaginary component, which contains valuable phase information, leading to a potential loss of relational details crucial for modeling long-context dependencies. In this paper, we propose an extension that re-incorporates this discarded imaginary component. Our method leverages the full complex-valued representation to create a dual-component attention score. We theoretically and empirically demonstrate that this approach enhances the modeling of long-context dependencies by preserving more positional information. Furthermore, evaluations on a suite of long-context language modeling benchmarks show that our method consistently improves performance over the standard RoPE, with the benefits becoming more significant as context length increases. The code is available at https://github.com/OpenMOSS/rope_pp.

## 1 Introduction

Large Language Model (LLM) based on attention mechanism (Vaswani et al., 2017) now dominates Natural Language Processing (NLP) (OpenAI, 2023; Sun et al., 2024; OpenAI, 2024; Yang et al., 2025a), particularly in the long-context arena (Hassabis & Kavukcuoglu, 2024; Young et al., 2024; ?), where attention overcomes the long-dependency bottlenecks of earlier architectures (LeCun et al.,
1995; Schmidhuber et al., 1997). Recent work extends their context length to the million-token scale (Liu et al., 2024b; InternLM, 2025), and the key driver is position-embedding design (Su et al.,
2024; Press et al., 2022; Peng et al., 2024). Among current LLMs, Rotary Position Embedding
(RoPE) (Su et al., 2024) has become the canonical choice (Dubey et al., 2024; Meta, 2024a;b). It encodes the absolute position of every query and key vector qt, ks, namely token indices *s, t* with a rotary matrix or complex multiplication, and when the two vectors make a dot product, it injects their relative position t − s, namely the relative distance, into the attention scores, thus combining the merits of traditional absolute and relative position embeddings (Vaswani et al., 2017; Dai et al., 2019; Yan et al., 2019) and securing widespread adoption.

Nevertheless, RoPE also has notable shortcomings, including poor length extrapolation (Press et al., 2022; Chen et al., 2023; bloc97, 2023), lack of data-sensitivity (Golovneva et al., 2024; Yang et al., 2025b), and no design for heterogeneous multi-modal input (Su, 2024a), prompting extensive research into its improvement. Most efforts concentrate on refining RoPE through interpolation designs (Peng et al., 2024; Liu et al., 2024d; Su, 2023), data-awareness (Zheng et al., 2024a;b), and feature-dimension partitioning (Wang et al., 2024; Wei et al., 2025). However, few work revisits the intrinsic computation of RoPE or analyze its inherent limitations (Hua et al., 2024; Dai et al., 2025). Re-examining RoPE in its complex-multiplication form reveals that the standard implementation keeps only the real part of the resulting complex attention score and discards the imaginary part outright (Su et al., 2024). Although taking the real part preserves the direct equivalence between complex multiplication and vector rotation, it incurs an irreversible information loss.

∗ Equal contribution. † Corresponding Author.

1

![1_image_0.png](1_image_0.png)

A closer look at the imaginary attention, strictly, the negative imaginary part of attention, shows that, compared with the real attention exhibiting stronger semantic locality, the imaginary heads attend more to long-context information as shown in Figure 1, promising gains on long-context tasks.

Moreover, adding imaginary attention also exposes qt, ks to a wider positional information range, implicitly improving length extrapolation. Therefore, we propose **RoPE++**, as illustrated in Figure 1, which re-injects the discarded imaginary component as a new group of attention heads computed in parallel with the real attentions. Particularly, we introduce **RoPE++**EH that keeps equal attention head number while halving QKV parameters as well as KV cache, and **RoPE++**EC that keeps equal cache size and doubles the number of attention heads. Theoretical analysis and pre-training experiments validate the above advantages. Both RoPE++EH and RoPE++EC outperform vanilla RoPE and other position embeddings on general tasks. On long-context benchmarks, RoPE++EH achieves comparable results with vanilla RoPE with half the cache, whereas RoPE++EC outperforms significantly at the same cache cost. Our contributions can be summarized as follows:
- We first identify the loss of imaginary information in standard RoPE and find it advantageous for capturing long-context dependencies by analyzing the properties of imaginary attention.

- Building on this, we propose RoPE++, which reintroduces the imaginary computation into attention in two configurations, RoPE++EH with equal head number and halved KV cache, and RoPE++EC with equal cache size and doubled attention heads. Both preserve the unified absolute–relative position-embedding format.

- Pre-training and evaluation at 376M and 776M sizes show that RoPE++EH and RoPE++EC
outperform vanilla RoPE and other position embeddings on average across short- and longcontext benchmarks. Further analysis reveals that the imaginary attentions play a dominant role in modeling long-context dependencies, confirming the effectiveness of introducing imaginary attention for improved long-context capability.

## 2 Related Work

Rotary Position Embedding (RoPE) is the dominant position embedding in current LLMs (Dubey et al., 2024; Meta, 2024a;b; Yang et al., 2025a). We analyze its good properties in Appendix B, including unifying relative and absolute information via rotation matrices and complex multiplication, and semantic aggregation as well as long-context decay. Yet it still faces many other challenges, attracting a great deal of effort to its improvement as mentioned above. A large body of work targets length extrapolation, scaling the rotary base (bloc97, 2023; Liu et al., 2024d; Xiong et al., 2024), interpolating or compressing index ranges (Press et al., 2022; Peng et al., 2024; Jin et al., 2024), or coupling RoPE with sparse attention (Lu et al., 2024; Xiao et al., 2024a; Liu et al., 2024c) to let models process contexts far longer than the training window. Other efforts extend RoPE to heterogeneous, cross-modal inputs (Su, 2024a), especially text–video sequences (Wang et al., 2024; Wei et al., 2025). Parallel lines design parametric schemes that encode contextual cues (Golovneva et al., 2024; Zheng et al., 2024a; Lin et al., 2025), refining or replacing RoPE to yield data-dependency. However, few works revisit RoPE's intrinsic computation or analyze its inherent limitations (Hua et al., 2024; Yang et al., 2025b; Dai et al., 2025). Particularly, the imaginary information loss of RoPE in rotation format compared with the complex multiplication format remains overlooked. Although prior work has tried to incorporate the full complex computation into the self-attention mechanism or neural networks (Wang et al., 2025; Lee et al., 2022), the characteristics and functionality of the imaginary component in position embedding remain unexplored. Therefore, we propose RoPE++ and close this gap through a deep analysis of the mathematical properties of imaginary attention and extensive validation on both short- and long-context downstream tasks.

## 3 Methodology

We begin our method by revisiting the complex form of RoPE. Only the real part of the complex product is retained, and the imaginary part is discarded, as shown in Equation 1. Although current LLMs perform well with this real-only attention, omitting the imaginary component may remove physical information. LLM no longer sees the full magnitude and phase of the complex attention result. This raises the question: can the imaginary part be re-incorporated into the attention computation?

$$\mathbf{A}_{t,s}=\text{Re}\left[\sum_{n=0}^{d/2-1}\tilde{q}_{t}^{(n)}\tilde{k}_{s}^{(n)*}e^{-i\theta_{n}(t-s)}\right]=\text{Re}\left[\sum_{n=0}^{d/2-1}\left(\tilde{q}_{t}^{(n)}e^{-i\theta_{n}t}\right)\left(\tilde{k}_{s}^{(n)}e^{-i\theta_{n}s}\right)^{*}\right]\tag{1}$$ $$=\sum_{n=0}^{d/2-1}\left(q_{t}^{(2n)}k_{s}^{(2n)}+q_{t}^{(2n+1)}k_{s}^{(2n+1)}\right)\cos\theta_{n}(t-s)+$$ $$\left(q_{t}^{(2n)}k_{s}^{(2n+1)}-q_{t}^{(2n+1)}k_{s}^{(2n)}\right)\sin\theta_{n}(t-s)$$

In this section, we will first propose our RoPE++ by re-introducing the imaginary information, in Section 3.1, as a new group of attention heads, namely imaginary attentions, compared with original real attentions. We then analyze the strengths from three aspects, the imaginary heads' stronger capture of long-context dependencies in Section 3.2, the cache and parameter reduction by combining imaginary and real heads in Section 3.3, and the impact on length extrapolation in Section 3.4.

## 3.1 Imaginary Extension Of Rope

We first recover the imaginary part that is discarded in Equation 1. The resulting expression is given in Equation 2. Strictly speaking, it is the negative imaginary part, and the reason will be detailed in Section 3.2. Similar to the real part, the imaginary part carries relative position information between qt, ks, so the formula can be rearranged into a vector form as shown in Equation 2.

AIm t,s = −Im  n=0 q˜ (n) t˜k (n)∗ se −iθn(t−s)   = −Im  n=0 q˜ (n) te −iθnt ˜k (n) se −iθns∗   d/ X 2−1  d/ X 2−1  (2) q (2n) t k (2n) s + q (2n+1) t k (2n+1) ssin θn(t − s)− = d/ X 2−1 q (2n) t k (2n+1) s − q (2n+1) t k (2n) scos θn(t − s) n=0
We observe that the imaginary attention still follows a rotation form and can be decomposed into absolute position embeddings on qt, ks, as shown in Equation 3. Specifically, the embedding applied to ks is identical to that used in the real attention in Equation 6 in Appendix B. For qt, the embedding is equivalent to rotating the vector by −π/2 before applying the same embedding in the real case.

AIm t,s = d/ X 2−1 #⊤cos θn(t − s) sin θn(t − s) − sin θn(t − s) cos θn(t − s) "k (2n) s k (2n+1) s n=0  "q (2n+1) t −q (2n) t # | {z } Relative PE #!⊤ cos θns − sin θns sin θns cos θns "k (2n) s k (2n+1) s = d/ X 2−1 n=0  cos θnt − sin θnt sin θnt cos θnt "q (2n+1) t −q (2n) t #! | {z } Absolute PE
$${\mathrm{(3)}}$$
We thus obtain an expression for the imaginary attention, strictly speaking, the negative imaginary attention. If we denote the rotation matrix as R· and RΘ,·. The latter is parameterized with θ0, · · · , θd/2−1. The computation of real and imaginary attention can be summarized in Equation 4.

ARe t,s = Re  n=0 q˜ (n) t˜k (n)∗ se iθn(s−t)   d/ X 2−1  = (RΘ,tqt) ⊤ RΘ,sks = q ⊤ t RΘ,s−tks AIm t,s = −Im   d/ X 2−1 n=0 q˜ (n) t˜k (n)∗ se iθn(s−t)   =RΘ,tR− π2 qt⊤ RΘ,sks = (R− π2 qt) ⊤RΘ,s−tks
(4)
Notably, the newly introduced imaginary component retains the key property of the original RoPE, that it can still be formulated either as a relative position or as an absolute position embedding. The only required adjustment is to rotate qt by −π/2 and then apply the standard position embedding to obtain the imaginary term. We refer to RoPE augmented with this imaginary extension as **RoPE++**. This augmentation raises further questions: what semantics does the imaginary attention convey, does it introduce additional overhead, and can it enhance model performance?

## 3.2 Capture Longer Dependency

As stated in Preliminary in Appendix B, the original RoPE-based attention or real attention exhibits semantic aggregation and *long-context decay*, both governed by its characteristic curve, as shown in Equation 7 and Figure 1. Similarly, we can derive the characteristic curve for the imaginary attention in RoPE++. It is the average of sin(θ∆t) over the same frequency distribution, approximating a sine integral function as shown in Equation 5 and Figure 1.

$$c_{\rm Im}(\Delta t)=\frac{2}{d}\sum_{n=0}^{d/2-1}\sin\left(10^{-\frac{\Delta n}{2}}\Delta t\right),\quad\bar{c}_{\rm Im}=\int\limits_{10-4}^{1}\frac{\sin\theta t}{\theta\ln10^{4}}{\rm d}\theta={\rm Si}(\Delta t)-{\rm Si}\left(\frac{\Delta t}{10^{4}}\right)\tag{5}$$

Although modeling distance with sin(θ∆t) is counter-intuitive, since sin(θ∆t) is zero at zero relative distance, rises, then falls, unlike cos(θ∆t)'s monotonic drop in the first half-period, the characteristic curve of the imaginary attention still shares the semantic-aggregation property of the real part. For
∆t > 0, when qt, ks are similar, their attention is on average larger regardless of relative distance, which is the reason why we take the negative imaginary part as imaginary attention. Moreover, on average, this component attends more to distant positions. As shown in Figure 1, its characteristic curve declines very slowly beyond a certain distance. Consequently, the imaginary part assigns more weight to the long-context region than the real part, helping LLM retrieve long-context information.

## 3.3 Cache And Parametric Efficiency

As described earlier, computing the imaginary attention requires only rotating the qt by −π/2, while every other operation is identical to the original RoPE. Because the positional embedding of ks is unchanged, we can interleave the −π/2-rotated qt with the original qt and perform the real and imaginary attention in a single pass in FlashAttention (Dao, 2024). Consequently, no extra KV

![4_image_0.png](4_image_0.png)

![4_image_1.png](4_image_1.png)

Figure 3: Comparison of trained position embedding interval between RoPE and RoPE++. The area within the dashed line represents trained relative position, and that beyond is in length extrapolation, with learned position embedding values colored in yellow and the opposite in gray. cache is introduced, and the method plugs directly into MHA or GQA (Ainslie et al., 2023), merely doubling the attention head group size, as shown in Figure 2b. We refer to this configuration as RoPE++EC, namely RoPE++ with equal cache size. The only cost of RoPE++EC is an additional imaginary attention computed alongside the real one under the fixed QKV parameter budget. Conversely, if the total head number is kept fixed, both QKV parameters and KV cache sizes are halved. We refer to this configuration as **RoPE++**EH, namely RoPE++ with equal attention head number, as shown in Figure 2c. In long-context scenarios, RoPE++EH halves the cache and raises throughput. Because the imaginary attention doubles the number of output heads, Wo must be twice as large as Wq. Therefore, Wo in RoPE++EH equals the original RoPE size, whereas Wo in RoPE++EC is double-sized. Experiments in Section 4 show that RoPE++EC outperforms the original RoPE, especially on long-context tasks, and RoPE++EH delivers comparable or even superior results.

Importantly, the imaginary and real attention, though computed independently and treated as separate heads, must share the same parameter. Both RoPE++EH and RoPE++EC share Wq between the real and imaginary attention. Allocating distinct subsets of heads to imaginary and real attention would effectively collapse back to standard RoPE, since rotating qt in imaginary attention by π/2 yields real attention, with no architecture modification. In other words, imaginary attention is defined relative to real attention and cannot exist independently. Therefore, configurations such as 75% imaginary vs. 25% real or 100% imaginary (applying only the imaginary part) are impossible under RoPE++.

## 3.4 Impact On Length Extrapolation

A closer inspection of the real and imaginary attention computations reveals an interesting discovery. In vanilla RoPE-based attention, or real attention, as shown in Equation 6, even-index query dimensions q
(2n)and odd-index key dimensions are multiplied only by cos θn(t − s) and sin θn(t − s)
whose values are always non-negative when θn is small. Once the input length exceeds the pretraining context length, these dimensions encounter out-of-distribution (OOD) negative embeddings as shown in Figure 5f and thus extrapolate poorly (Liu et al., 2024d; Peng et al., 2024). In RoPE++
as shown in Equation 3, these dimensions are multiplied by - cos θn(t − s) and sin θn(t − s) in the imaginary attention, so during pre-training, they have already observed both negative and positive position embedding as well as their maximum and minimum value ±1. Consequently, these dimensions no longer suffer from the length extrapolation problem in longer contexts (Liu et al., 2025b). Likewise, odd-index query dimensions q
(2n+1) and even-index key dimensions k
(2n)encounter only cos θn(t − s) and - sin θn(t − s) in the real attention, and the imaginary attention further exposes them to cos θn(t − s) and sin θn(t − s). Yet this alone does not expand the position embedding range trained in pre-training, as shown in Figure 5h and Figure 5j. However, when real and imaginary attention are combined, qt, ks in RoPE++ attains the full cos and sin value range, once the training length exceeds half the sinusoidal period, whereas the vanilla RoPE requires a full period. Consequently, more dimensions in RoPE++ observe complete positional information. Therefore, perplexity grows more slowly beyond the maximum supported context length (Liu et al., 2024d; Men et al., 2024).

## 4 Experiment 4.1 Setup

We validate RoPE++ at both 776M and 376M model sizes, with architectural details in Appendix C. Both models are pre-trained on DCLM-Baseline-1.0 corpus (Li et al., 2024) by HuggingFace Transformers (Wolf et al., 2020) on 8 NVIDIA H200 160 GB GPUs. For each size, we use a batch size of 0.5M tokens and pre-train for 50B tokens. We use AdamW (Loshchilov et al., 2017) optimizer with weight decay 0.1, a maximum learning rate of 5e-4, and a warmup-stable-decay scheduler. We use the first 0.5B tokens for warmup, and the final 5B tokens for decay, and the learning rate ends at 0. We compare our RoPE++ with standard RoPE (Su et al., 2024) and other well-known position embedding designs, including FoPE (Hua et al., 2024), Pythia (namely, partial RoPE with only last 1/4 dimensions being rotated) (Biderman et al., 2023), as well as ALiBi (Press et al., 2022). We pre-train all methods on 4k context length with an initial rotary base of 10000. For RoPE and RoPE++, we conduct continuous long-context pre-training. Following Xiong et al. (2024); Lv et al. (2024), we scale the rotary base from 10000 to 500000 and train for 10B tokens from DCLM on 32k context length, using a cosine-annealing learning rate scheduler and keeping all other settings.

## 4.2 Short-Context Evaluation

We evaluate both short-context and long-context tasks based on OpenCompass (Contributors, 2023).

For short-context evaluation, we measure perplexity on WikiText (Merity et al., 2017) and LAM- BADA(Paperno et al., 2016) and assess downstream tasks mainly in Open LLM Leaderboard (HuggingFace, 2023), including TruthfulQA (Lin et al., 2022), PIQA (Bisk et al., 2020), HellaSwag (Zellers et al., 2019), WinoGrande (Sakaguchi et al., 2020), ARC-e (Clark et al., 2018), GPQA (Rein et al., 2023), SocialIQA (Sap et al., 2019), OpenBookQA (Mihaylov et al., 2018), and SuperGLUE (Wang et al., 2019). All models are tested within a 4k context length.

The results are shown in Table 1. Our RoPE++EC and RoPE++EH achieve the best average scores on short-context tasks compared with RoPE and every other position embedding design. Notably, RoPE++EH surpasses standard RoPE with only half the KV-cache and QKV parameters. After further long-context pre-training, RoPE++ still retains this edge over RoPE on short-text benchmarks.

## 4.3 Long-Context Evaluation

For long-context evaluation, we evaluate downstream performance at varying lengths with the classical synthetic benchmarks, RULER (Hsieh et al., 2024) and BABILong (Kuratov et al., 2024). The results are shown in Table 2 and Figure 6. We highlight the comparison with RoPE in long-context training because RoPE is the position embedding currently most widely used by long-context LLMs. On RULER and BABILong up to 64k context, our RoPE++ again acquires the highest scores.

Particularly, RoPE++EH achieves comparable performance with vanilla RoPE using half the KV-

Wiki LMB TQA PIQA Hella Wino ARC-e GPQA SIQA OBQA SG Avg.

ppl ↓ ppl ↓ acc ↑ acc ↑ acc ↑ acc ↑ acc ↑ acc ↑ acc ↑ acc ↑ acc ↑

376M Short RoPE 19.9 32.7 35.5 66.3 34.8 50.9 39.3 24.8 38.6 **27.4** 43.7 40.1 FoPE 19.3 33.0 33.8 65.9 34.5 **53.0** 37.0 **28.8** 39.5 24.2 43.6 40.0 Pythia **19.2** 32.9 34.7 65.8 34.9 51.5 41.3 21.2 39.7 25.6 42.5 39.7 ALiBi 21.2 34.6 33.8 66.1 34.2 51.1 **44.4** 24.8 38.7 27.4 43.9 40.5 RoPE++EH 20.8 33.6 36.3 66.4 34.5 52.5 40.9 23.7 **40.5** 24.8 43.2 40.3 RoPE++EC 19.4 **32.6 37.3 68.0 35.6 53.0** 41.3 25.8 40.3 23.2 **44.8 41.0** 376M Long RoPE 20.4 **33.8** 35.4 64.9 34.1 50.6 40.4 21.2 **39.4** 27.4 43.5 39.6 RoPE++EH 21.7 34.8 35.2 64.5 **34.3** 49.9 **41.5 22.7** 40.0 27.0 43.1 39.8 RoPE++EC **20.0** 33.9 **37.1 66.1** 34.1 **53.4** 38.1 21.2 39.2 **28.4 43.7 40.1** 776M Short RoPE 14.8 27.3 35.5 **70.1 43.7** 52.3 43.4 25.8 41.3 21.8 43.6 42.0 FoPE **14.7** 27.1 33.6 68.7 43.4 52.9 **45.0** 24.8 39.7 24.8 45.4 42.0 Pythia 14.8 **26.9** 35.8 68.8 42.9 52.1 39.5 22.2 42.0 21.2 43.6 40.9 ALiBi 15.2 28.3 35.2 70.2 **43.7 53.6** 43.2 23.7 40.6 **27.6 45.9** 42.6

RoPE++EH 15.6 28.1 35.4 69.6 42.7 53.5 **45.0** 15.8 **41.6** 26.8 42.4 42.5 RoPE++EC 14.8 27.3 **36.1** 69.3 43.6 52.3 43.7 **28.3** 40.1 **27.6** 44.4 **42.8**

776M Long RoPE 14.6 27.3 35.1 68.9 43.1 51.5 **47.6** 21.7 40.7 20.2 42.6 41.3 RoPE++EH 15.3 28.1 **35.4** 69.9 41.9 **52.6** 43.2 28.3 **41.0** 22.2 43.4 42.0 RoPE++EC **14.4 27.1** 35.2 **70.4 43.7 52.6** 44.8 **31.8** 40.8 **27.6 44.3 43.5**

Table 1: Results on short-context tasks for 776M and 376M models pre-trained in 4k context length and further trained on 32k. Best results are highlighted in bold, with the second best underlined for broader comparison. Our RoPE++ achieves the best average performance on different model sizes.

RULER BABILong

4k 8k 16k 32k 64k Avg. 2k 4k 8k 16k 32k 64k Avg.

376M Long RoPE 31.6 25.6 22.0 9.5 5.5 18.8 17.7 16.1 9.1 9.4 5.9 7.8 11.0 RoPE++EH 29.9 28.4 17.6 9.4 5.9 18.2 14.1 15.6 12.2 9.9 8.3 9.7 11.6 RoPE++EC **36.1 33.0 29.1 17.7 9.0 25.0 19.8 19.8 16.1 15.8 12.3 12.8 16.1** 776M Long RoPE 37.4 35.1 33.0 21.2 10.4 27.4 **33.5 30.7** 23.6 22.0 15.1 12.1 22.8

RoPE++EH 38.7 35.4 **33.8 24.6** 10.7 28.6 31.9 26.5 18.6 16.2 11.0 12.2 19.4 RoPE++EC **42.7 38.6** 33.4 21.7 **10.9 29.4** 32.4 29.9 **24.4 24.5 18.6 14.8 24.1**

Table 2: Results on long-context tasks, including RULER and BABILong for 776M and 376M models further trained with 5B tokens in 32k context length. Best results are highlighted in bold. Our RoPE++ achieves the best performance on average, especially in long-context scenarios.

cache and QKV parameters, while RoPE++EC delivers significant gains at the same cache size.

Although RoPE occasionally edges ahead at a few shorter context lengths, RoPE++, including both RoPE++EC and RoPE++EH, maintains more stable performance as context length grows and achieves best performance in 64k context length extrapolation consistently.

## 5 Discussion 5.1 Rope++ As Cache Optimization

As mentioned in Section 3.3, RoPE++EH halves KV cache and QKV parameters while keeping the attention head number equal, yielding evident efficiency gains. We validate this efficiency strength by assessing the memory cost as well as Time-Per-Output-Token (TPOT) of 376M and 776M models,

![7_image_0.png](7_image_0.png)

![7_image_1.png](7_image_1.png)

![7_image_2.png](7_image_2.png) 

![7_image_3.png](7_image_3.png) 
from 2k to 32k context length. We conduct the efficiency evaluation on a single NVIDIA H200 160BG GPU, with a batch size of 8 samples. The results are shown in Figure 4. At both 376M
and 776M, RoPE++EH consistently reduces memory cost and speeds up decoding, with the margin widening as context length increases.

## 5.2 Attention Pattern Of Rope++

To verify how imaginary attention captures long-context dependencies and to contrast it with real attention in RoPE++, we inspect the attention patterns of short-context-trained RoPE++EC at 376M and 776M as shown in Figure 5. Odd-index imaginary attention highlights the initial positions more strongly than even-index real heads, indicating a stronger global focus. Since prior work (Liu et al., 2025a; Wei et al., 2025) shows that dimensions attending globally are more critical for long-context semantics, imaginary attention may play the dominant role in long-context tasks.

For further verification, we design the following validation experiment. We add Gaussian noise with equal standard deviation to the imaginary and real attention components separately, and monitor the change in RoPE++ performance on long-context tasks, such as the average score of RULER-4k. Curves for RULER-4k versus standard deviation are plotted for both real and imaginary attention. When the standard deviation σ is small (σ < 0.2), scores with corrupted real or imaginary attentions stay close to the baseline; when it is large enough (σ = 1.5), both drop sharply. Importantly, in the

Short RULER BABILong

ppl score 4k 8k 16k 32k Avg 2k 4k 8k 16k 32k Avg

376M Long PI RoPE **33.4** 42.0 36.5 **33.6** 19.7 **10.6** 25.1 19.3 12.3 10.2 10.9 10.9 12.7 RoPE++EH 34.7 41.7 28.0 27.6 15.8 6.9 19.6 13.3 12.4 12.8 8.9 10.4 11.6 RoPE++EC 33.7 **42.8 37.0** 32.4 **28.3 10.6 27.1 24.0 20.7 15.9 14.3 12.3 17.4** 376M Long YaRN RoPE **32.8** 42.2 **36.4** 32.9 28.4 15.0 28.2 22.4 16.4 11.4 10.7 11.1 14.4 RoPE++EH 33.9 42.2 32.7 30.2 24.9 10.7 24.7 8.7 9.3 12.1 11.3 10.9 10.5 RoPE++EC 32.9 **43.4** 36.0 **33.9 31.7 17.8 29.8 27.4 23.6 18.0 16.9 12.3 19.6** 776M Long PI RoPE **27.8** 40.4 37.8 34.4 **30.5** 13.4 29.0 15.3 16.9 12.7 11.8 9.3 13.2

RoPE++EH 28.8 40.4 37.9 35.0 27.5 **14.6** 28.8 21.0 22.4 **17.1 13.7 11.1 17.1** RoPE++EC **27.8 40.5 43.0 38.7** 28.8 13.6 **31.0 25.7 23.4** 16.4 9.4 8.0 16.6

776M Long YaRN RoPE **27.3** 40.9 37.6 35.0 33.9 **27.5** 33.5 26.9 **25.6** 19.5 16.4 12.2 20.1 RoPE++EH 28.3 40.6 37.9 34.9 32.2 26.1 32.8 **28.0** 23.9 18.6 17.8 11.7 20.0 RoPE++EC **27.3 41.5 42.9 36.5 36.3** 22.2 **34.4** 26.3 24.1 **21.1 19.8 16.9 21.6**

Table 3: Results of 776M and 376M models further trained with 5B tokens in 32k context length with YaRN and Linear PI. Our RoPE++ still achieves the best performance on average. intermediate range, adding noise to the imaginary attention always performs worse than corrupting the real part. When σ = 1.0, for example, the real-noised RoPE++ outperforms the imaginary-noised one by 5 points at 376M and 8 points at 776M, which demonstrates a significant gap. Thus, impairing the imaginary heads degrades long-context performance more, confirming that imaginary attention plays a more dominant role in long context modeling.

## 5.3 Combination With Other Long-Context Techniques

RoPE++ can not only be combined with NTK for context extension during long-context training, but can also be combined with other long-context techniques such as Linear PI (Chen et al., 2023) and YaRN (Peng et al., 2024). Across 376M and 776M model sizes, we conduct extensive experiments of long-context further pre-training in 32k context length, with the interpolation coefficient s = 8 for Linear PI and s = 32 for YaRN, the default values in the original paper. The results are shown in Table 3. We report the perplexity on WikiText and the average score of tasks we have presented in Table 1 as the summary of short-context performance, with the full results in Table 10. Results show that RoPE++ consistently achieves the highest scores on RULER, BABILong, and short-context average score, confirming its advantage and generalization. More analysis on larger model scale and training convergence is detailed in Appendix C. More discussion on the extrapolation performance and limitation of RoPE++ can be found in Appendix D.

## 6 Conclusion

We introduce RoPE++, which employs both real and imaginary attentions. Mathematical analysis first reveals the imaginary attention's potential for modeling long-context dependencies. Building upon this, we re-incorporate the originally discarded imaginary attention as a new group of heads while preserving the unified absolute–relative position embedding format. Particularly, we introduce RoPE++EH, with equal head as well as halved cache, and RoPE++EC with equal cache and doubled heads. Pre-training and evaluation at 376M and 776M model sizes show that both RoPE++EH and RoPE++EC outperform vanilla RoPE and other position embeddings on average across shortcontext tasks and acquire even larger gains in long-context scenarios. Further analysis confirms that imaginary attentions are more dominant in long-context modeling compared with original real attention, validating their effectiveness in enhancing long-context LLMs.

## Acknowledgement

This work was supported by the National Natural Science Foundation of China (No. U24B20181) and Shanghai Pilot Program for Basic Research - Fudan University 21TQ1400100 (22TQ018). We greatly appreciate all reviewers for their constructive reviews, and thanks to Jiasheng Ye for the discussion on scaling verification of model architecture.

## Ethical Statement

This research follows established ethical standards and practice principles. To our knowledge, our study processes no sensitive personal data, involves no human subjects, and targets no ethically risky applications. All experiments and analyses comply with recognized guidelines, ensuring integrity, transparency, and reliability.

## Reproducibility Statement

To ensure the reproducibility of and to support the open-source community, we have publicly released RoPE++, its trained checkpoints, and the complete training and evaluation code. We expect these as a reference for future work on long-context LLMs, facilitating progress in this field.

## References

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit ´
Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023.

Federico Barbero, Alex Vitvitskyi, Christos Perivolaropoulos, Razvan Pascanu, and Petar Velickovi ˇ c.´
Round and round we go! what makes rotary positional encodings useful? arXiv preprint arXiv:2410.06205, 2024.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pp. 2397–2430. PMLR, 2023.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: reasoning about physical commonsense in natural language. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pp. 7432–7439. AAAI Press, 2020. doi:
10.1609/AAAI.V34I05.6239. URL https://doi.org/10.1609/aaai.v34i05.6239.

bloc97. Dynamically scaled rope further increases performance of long context llama with zero fine-tuning, July 2023. URL https://www.reddit.com/r/LocalLLaMA/comments/ 14mrgpr/dynamically_scaled_rope_further_increases/.

Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation. *arXiv preprint arXiv:2306.15595*, 2023.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the AI2 reasoning challenge.

CoRR, abs/1803.05457, 2018. URL http://arxiv.org/abs/1803.05457.

OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models.

https://github.com/open-compass/opencompass, 2023.

Chang Dai, Hongyu Shan, Mingyang Song, and Di Liang. Hope: Hyperbolic rotary positional encoding for stable long-range dependency modeling in large language models. *arXiv preprint* arXiv:2509.05218, 2025.

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860, 2019.

Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. In The Twelfth International Conference on Learning Representations, 2024.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Olga Golovneva, Tianlu Wang, Jason Weston, and Sainbayar Sukhbaatar. Contextual position encoding: Learning to count what's important. *arXiv preprint arXiv:2405.18719*, 2024.

Demis Hassabis and Koray Kavukcuoglu. Introducing gemini 2.0: our new ai model for the agentic era, 2024. URL https://blog.google/technology/google-deepmind/
google-gemini-ai-update-december-2024/.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. Ruler: What's the real context size of your long-context language models? *arXiv preprint arXiv:2404.06654*, 2024.

Ermo Hua, Che Jiang, Xingtai Lv, Kaiyan Zhang, Youbang Sun, Yuchen Fan, Xuekai Zhu, Biqing Qi, Ning Ding, and Bowen Zhou. Fourier position embedding: Enhancing attention's periodic extension for length generalization. *arXiv preprint arXiv:2412.17739*, 2024.

HuggingFace. Open llm leaderboard. 2023. URL https://huggingface.co/spaces/
HuggingFaceH4/open_llm_leaderboard.

InternLM. Internlm3-8b, January 2025. URL https://huggingface.co/internlm/
internlm3-8b-instruct.

Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H Abdi, Dongsheng Li, Chin-Yew Lin, et al. Minference 1.0: Accelerating pre-filling for long-context llms via dynamic sparse attention. *arXiv preprint arXiv:2407.02490*, 2024.

Hongye Jin, Xiaotian Han, Jingfeng Yang, Zhimeng Jiang, Zirui Liu, Chia-Yuan Chang, Huiyuan Chen, and Xia Hu. Llm maybe longlm: Self-extend llm context window without tuning. *arXiv* preprint arXiv:2401.01325, 2024.

Yuri Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rodkin, Dmitry Sorokin, Artyom Sorokin, and Mikhail Burtsev. Babilong: Testing the limits of llms with long context reasoning-in-a-haystack. arXiv preprint arXiv:2406.10149, 2024.

Yann LeCun, Yoshua Bengio, et al. Convolutional networks for images, speech, and time series. The handbook of brain theory and neural networks, 3361(10):1995, 1995.

ChiYan Lee, Hideyuki Hasegawa, and Shangce Gao. Complex-valued neural networks: A comprehensive survey. *IEEE/CAA Journal of Automatica Sinica*, 9(8):1406–1426, 2022.

Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Yitzhak Gadre, Hritik Bansal, Etash Guha, Sedrick Scott Keh, Kushal Arora, et al. Datacomp-lm: In search of the next generation of training sets for language models. *Advances in Neural Information Processing* Systems, 37:14200–14282, 2024.

Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
ACL 2022, Dublin, Ireland, May 22-27, 2022, pp. 3214–3252. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.ACL-LONG.229. URL https://doi.org/10. 18653/v1/2022.acl-long.229.

Zhixuan Lin, Evgenii Nikishin, Xu Owen He, and Aaron Courville. Forgetting transformer: Softmax attention with a forget gate. *arXiv preprint arXiv:2503.02130*, 2025.

Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixture-ofexperts language model. *arXiv preprint arXiv:2405.04434*, 2024a.

Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with ringattention. *arXiv e-prints*, pp. arXiv–2402, 2024b.

Xiaoran Liu, Ruixiao Li, Qipeng Guo, Zhigeng Liu, Yuerong Song, Kai Lv, Hang Yan, Linlin Li, Qun Liu, and Xipeng Qiu. Reattention: Training-free infinite context with finite attention scope. arXiv preprint arXiv:2407.15176, 2024c.

Xiaoran Liu, Hang Yan, Chenxin An, Xipeng Qiu, and Dahua Lin. Scaling laws of rope-based extrapolation. In *The Twelfth International Conference on Learning Representations*, 2024d.

Xiaoran Liu, Siyang He, Qiqi Wang, Ruixiao Li, Yuerong Song, Zhigeng Liu, Linlin Li, Qun Liu, Zengfeng Huang, Qipeng Guo, et al. Beyond homogeneous attention: Memory-efficient llms via fourier-approximated kv cache. *arXiv preprint arXiv:2506.11886*, 2025a.

Xiaoran Liu, Zhigeng Liu, Zengfeng Huang, Qipeng Guo, Ziwei He, and Xipeng Qiu. Longllada:
Unlocking long context capabilities in diffusion llms. *arXiv preprint arXiv:2506.14429*, 2025b.

Ilya Loshchilov, Frank Hutter, et al. Fixing weight decay regularization in adam. *arXiv preprint* arXiv:1711.05101, 5(5):5, 2017.

Yi Lu, Xin Zhou, Wei He, Jun Zhao, Tao Ji, Tao Gui, Qi Zhang, and Xuanjing Huang. Longheads:
Multi-head attention is secretly a long context processor. *arXiv preprint arXiv:2402.10685*, 2024.

Kai Lv, Xiaoran Liu, Qipeng Guo, Hang Yan, Conghui He, Xipeng Qiu, and Dahua Lin. Longwanjuan:
Towards systematic measurement for long text quality. *arXiv preprint arXiv:2402.13583*, 2024.

Xin Men, Mingyu Xu, Bingning Wang, Qingyu Zhang, Hongyu Lin, Xianpei Han, and Weipeng Chen. Base of rope bounds context length. *arXiv preprint arXiv:2405.14591*, 2024.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. URL https:
//openreview.net/forum?id=Byj72udxe.

AI Meta. Introducing meta llama 3: The most capable openly available llm to date. *Meta AI.*, 2024a. AI Meta. Llama 3.2: Revolutionizing edge ai and vision with open, customizable models. *Meta AI.*,
2024b.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? A new dataset for open book question answering. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun'ichi Tsujii (eds.), Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, pp. 2381–2391. Association for Computational Linguistics, 2018. doi: 10.18653/V1/D18-1260. URL
https://doi.org/10.18653/v1/d18-1260.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-
Rong Wen, and Chongxuan Li. Large language diffusion models. *arXiv preprint arXiv:2502.09992*, 2025.

OpenAI. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023. OpenAI. O1: Openai's first model, 2024. URL https://openai.com/o1/. Accessed: 202412-25.

Denis Paperno, German Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, ´
Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernandez. The LAMBADA dataset: ´
Word prediction requiring a broad discourse context. In *Proceedings of the 54th Annual Meeting* of the Association for Computational Linguistics, ACL 2016, August 7-12, 2016, Berlin, Germany, Volume 1: Long Papers. The Association for Computer Linguistics, 2016. doi: 10.18653/V1/
P16-1144. URL https://doi.org/10.18653/v1/p16-1144.
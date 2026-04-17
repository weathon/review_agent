# Holistic protean block for long-range DNA sequence modeling

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Modeling DNA sequences, which are defined by a complex interplay of local motifs, long-range dependencies, and periodic patterns, is a fundamental challenge in computational biology. Existing foundation models based on CNNs, Transformers, and SSMs are constrained by static or time-domain-only signal processing operations, which limit the architectural flexibility and multi-domain perspective needed to fully capture the diverse features of DNA. Here, we introduce the **Holistic Protean Block (HPB)**, a novel scalable architecture that achieves multi-level plasticity through three synergistic layers. Its Locus Plasticity Layer (LPL) provides token-level plasticity by employing token-specific convolution operations, allowing it to precisely model fine-grained, local patterns. Its Domain Plasticity Layer (DPL) establishes perspective-level plasticity by concurrently modeling both sequential (time) and spectral (frequency) features, enabling it to form multi-domain, global representations. Its Saliency Plasticity Layer (SPL) realizes information flow plasticity by learning saliency scores along dual axes, thereby permitting it to focus information flow on the most critical features. These layers work in tandem, extracting a holistic representation of diverse genomic patterns by adaptively reshaping their computational strategy. The DNA model constructed with HPB (**HPB-DNA**) not only achieves state-of-the-art performance on various genomic benchmarks with a quasi-linear complexity, but is also validated by in-depth model analyses, which collectively establish the HPB as a more powerful and principled paradigm for DNA sequence modeling. Code will be available upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors introduced a Holistic Protean block (HPB) as a new building block for long context gLMs. HPB combines three modules including the Locus Plasticity Layer that learns token-specific convolution filters, the Domain Plasticity Layer (DPL) that fuses local and long convolutions, and the saliency plasticity layer (SPL) that performs dual-axis feature gating mechanism. By stacking these three types of blocks, HPB aims to capture both local and long range interactions of long genomics sequences while having quasi-linear time complexity. The proposed model is tested on Genomics Benchmarks and NT benchmark. Author also show improvements on chromatin profile prediction and a long-context VEP task.

### Strengths
1. HPB's design unifies two convolution block variants and a gated mechanism into a single operator block. This enables the model to capture both short and long-range interaction of the long sequence. Benchmarking on both short and long range tasks show improvements comparing to existing DNA LMs.

2. The model demonstrate also improvement on a VEP prediction task, surpassing both existing DNA lms and a supervised expert model of Enformer.

### Weaknesses
1. Despite the holistic framing, each HPB component echoes a previously known design. The LPL is essentially a Conditional Conv layer, and DPL is pretty similar to Hyena's long-convolution operator. The motivation for combining these components in this way is questionable.

2. The improvements on the benchmarks seems minimal and doesn't show significance in statistics. Maybe harder benchmarks like DNALongBench/BEND can be considered to better show benchmark result.

3. Though the paper states about quasi-linear computation time, it doesn't show an efficiency analysis section comparing to other architectures.

### Questions
1. Can the author evaluates their model on some newer and harder benchmarks like BEND or DNALongBench?

2. Can the author add an analysis section about training/inference throughput and memory consumption to show the performance potential of this HPB architecture comparing to the other existing ones like Mamba, Hyena or Transformers with flash attention?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes the Holistic Protean Block (HPB) for DNA sequence modeling, consisting of three components: (1) Locus Plasticity Layer (LPL) using dynamic multi-scale convolutions for local features, (2) Domain Plasticity Layer (DPL) combining global convolutions and wavelet transforms for time-frequency modeling, and (3) Saliency Plasticity Layer (SPL) for channel and position feature weighting. The authors claim HPB achieves "multi-level plasticity" to capture diverse DNA patterns including local motifs, long-range dependencies, and periodic signals. Evaluated on genomic benchmarks, the HPB-based model (HPB-DNA) reports state-of-the-art performance with quasi-linear complexity O(L log L).

### Strengths
1. Modeling DNA from a frequency perspective is novel and interesting, as DNA sequences indeed exhibit periodic patterns that current methods overlook. However, this novel point is obscured by excessive packaging of incremental architectural designs, diminishing the paper's clarity and impact.

2. The experimental evaluation is comprehensive, covering diverse benchmarks including regulatory annotation, histone markers, and variant effect prediction.

3. The ablation studies systematically validate the contribution of each component.

### Weaknesses
1. Most architectural components are incremental improvements: multi-scale convolutions, gated convolutions, and dual-axis feature gating are well-established techniques. While incremental engineering is acceptable, the authors unnecessarily complicate these straightforward designs with unclear terminology, making the paper difficult to read and understand.

2. The paper emphasizes "long-range DNA sequence modeling" yet pre-trains on only 1K bp sequences (Appendix A.2), far shorter than Enformer (196K bp). This length cannot capture enhancer-promoter interactions mentioned in the motivation nor demonstrate the claimed dynamic modeling capabilities. Moreover, when both pre-training and evaluation use short sequences (<= 1K bp), the O(L log L) complexity advantage over linear methods becomes negligible. This undermines the "long-range" claim and makes comparisons with methods pre-trained on longer sequences potentially unfair.

3. The abstract and introduction lack clarity from both biological and technical perspectives, obscuring rather than clarifying the contributions. I strongly recommend the authors use simpler, community-standard terminology rather than fancy-sounding phrases that obscure meaning. If the authors truly want to use these terms, they should provide clear explanations in context; otherwise, it will be very confusing for readers.  For examples: 
   - Lines 14-15: "single/multi-domain" operations are undefined, what is "domain" here?
   - Line 17: "plasticity" is non-standard in both AI and genomics literature
   - Line 18: "Locus Plasticity Layer" uses "locus" (a genomics term for gene position) when "position" suffices
   - Line 85: "construction coefficients" is simply gating/weighting as in MoE but needlessly renamed
   - Line 86: "meta features" lacks clear definition
   - "Holistic Protean Block" uses uncommon terminology to convey only a general description of the model design  
 
  4. The related work omits discussion of wavelet transforms and frequency-domain methods, despite frequency modeling being the paper's main novelty. Since most genomics researchers are unfamiliar with signal processing techniques, comprehensive background is essential.

5. Line 82 claims the HPB is a "scalable building block," but the largest model tested is under 2M parameters. Given the architectural complexity (three parallel modules with wavelets, FFT, and dynamic convolutions), it is unclear whether this design can scale up like foundation models in NLP or other genomics models.

6. It appears that different pre-trained models were used for different downstream tasks, which seems to contradict the concept of a foundation model. Additionally, the hyperparameters across downstream tasks vary drastically, raising concerns that the downstream performance may be attributed to hyperparameter tuning rather than fundamental improvements.

### Questions
1. Line 51 claims Transformers "lack strong inductive biases for local patterns" citing Enformer, but this requires stronger evidence or clarification of what Enformer actually states.

2. Have the authors considered applying this architecture to sequence-to-function modeling tasks like Enformer (predicting gene expression from long genomic sequences)? The motivation emphasizes long-range dependencies and frequency-sensitive patterns, which are most relevant in such tasks requiring long-context predictions. The current evaluation on short-sequence benchmarks cannot adequately demonstrate whether the proposed complexity genuinely improves biological modeling or merely provides marginal gains on oversimplified tasks.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new architecture that integrates three different functional layers (LPL, DPL, SPL) that work together to learn a holistic representation of DNA sequences. The HPB-DNA model demonstrates strong empirical performance, setting new STOA results on genomic benchmarks with efficient, quasi-linear scaling.

### Strengths
- Formulating DNA analysis in terms of multi-level plasticity (locus, domain, saliency) is novel and seems like a good way to conceptualize genomic data.

- The authors evaluate across multiple genomic datasets and show consistent improvements under comparable model sizes.

- The quasi-linear complexity and small parameter count appear to be a promising way to address genomics tasks.

### Weaknesses
- The paper introduces several novel architectural operations, but there is not enough work done in the current ablation study isolating their individual contributions. It would strengthen the work to include experiments showing how the model performs when each new component is removed or replaced. 
- The paper does not clearly articulate the theoretical or biological motivation behind combining a global convolution with a wavelet transform. The rationale for the wavelet path, in particular, should be clarified - what specific objective does it serve? Providing stronger domain-grounded justification would make the design choice more compelling.
- The paper asserts that HPB is scalable to foundation-model scale, but empirical evidence is limited to relatively small (< 5M parameter) models. The authors should demonstrate that HPB is actually scalable in a way that improves model performance. 
- Table 4 presents performance on a limited subset of datasets, which may give the impression of selective reporting. Showing results across all genomic benchmarks would provide a more comprehensive and fair evaluation of model performance. Similarly, the same issue is present in Table 5. 
- Given the model’s efficiency claims, the paper should include a quantitative comparison of computational cost across methods, covering memory usage, wall clock training and inference time, and FLOPs to provide a comprehensive assessment of efficiency.

### Questions
- How sensitive is the model to changes in the number of wavelet decomposition levels and to kernel-base selection in the LPL? In general, further reporting on how hyperparameters were derived would be helpful. 

- Can the authors quantify how much each plasticity layer contributes to the overall improvement?

- Are there domain-related failure modes where multi-domain modeling introduces noise or instability (such as with sequences with no clear periodicity)?

- Could the authors provide additional analyses on whether the learned features are biologically interpretable?

### Soundness
3

### Presentation
2

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
The manuscript presents the Holistic Protean Block, a DNA language model that is specifically designed to add the capability of identifying structural periodic signals to existing models that are based on CNNs, transformers and state spaces. This capability is implemented in a newly constructed Domain Plasticity Layer that runs parallel to a Saliency Plasticity Layer, and downstream of a Locus Plasticity Layer that tokenizes the input sequence. Results are encouraging, but for some benchmarks inconclusive

### Strengths
1. Identificaiton of a systemic weakness in existing models in handling approximate periodicity, and tackling it using the DPL
2. Diverse benchmarking
3. Fig 5a is very convincing in the identification of  agenuine effect

### Weaknesses
1. Inconsistency of results
2. Failure to take advantage of existing models that already do much of the required work
3. Lack of clarity regarding parameters 
4. Lack of comparison to local models

### Questions
1. The authors should separate out the contribution of the DPL (their innovatiive part) from other changes to standard models.(that muddy the water, whether or not they provide some benefit). Table 5 seems to address this, but only on 3 prediction tasks raising the concern of cherrypicking. 

2. There are good models for local motifs (the authors cite great models from a decade ago, but progress continued). It is unclear why the authors choose ot have this component redone, rather than stand on the shoulders of giants. and use an existing model. Given that they have not they should compare to such models, esp. if they wish to claim advantage (4 stars vs. 3 in Fig 1). Table 4 seems to try to address this, but only on 3 prediction tasks raising the concern of cherrypicking. Also it compares to strawmen rather than strong local models. Worse, Table 2 shows ConvNova to beat the proposed model by a significant margin on several tasks, whereas when the proposed model beats ConvNova, it is almost always with overlap ping confidence interval and always close to that. 

3. I am confused by the changing numbers of parameters of models between tables. If this is due to different lenghts of the sequences, then some sense of consistency should be provided by a unifying formula for this number

4. Fig 5b is unconvincing withiut a comparison to other methods

5. Fig 4 appears after Fig 5.

### Soundness
2

### Presentation
2

### Contribution
3

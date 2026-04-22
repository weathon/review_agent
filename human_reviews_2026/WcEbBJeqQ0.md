# TempoPFN: Synthetic Pre-training of Linear RNNs for Zero-shot Time Series Forecasting

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 8, 2

## Abstract
Foundation models for zero-shot time series forecasting face challenges in efficient long-horizon prediction and reproducibility, with existing synthetic-only approaches underperforming on challenging benchmarks. This paper presents TempoPFN, a univariate time series foundation model based on linear Recurrent Neural Networks (RNNs) pre-trained exclusively on synthetic data. The model uses a GatedDeltaProduct architecture with state-weaving for fully parallelizable training across sequence lengths, eliminating the need for windowing or summarization techniques while maintaining robust temporal state-tracking. Our comprehensive synthetic data pipeline unifies diverse generators including stochastic differential equations, Gaussian processes, and audio synthesis with novel augmentations such as time-varying TSMixup, differentiation, and integration. In zero-shot evaluations on the Gift-Eval benchmark, TempoPFN achieves state-of-the-art performance, matching models trained on real-world data while being significantly more efficient than existing baselines. We open-source our complete data generation pipeline and training code.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces TempoPFN, a foundation model for univariate time series forecasting that makes two key contributions: it is trained entirely on synthetic data, including waveforms, Gaussian Processes (GPs), and Stochastic Differential Equations (SDEs), and it employs a lightweight linear gated recurrent neural network architecture with approximately 35 million parameters and a simple yet effective mechanism. Despite relying solely on synthetic data and a relatively modest model design, TempoPFN achieves competitive zero-shot performance, in comparison to state-of-the-art foundation models trained on real-world datasets.

### Strengths
* This work enhances the notion of real-data free training, suggesting that real data can be spanned by a combination of processes. 
* A rather simple model, with solid performance. Additionally, this work emphasizes the growing potential in RNN (or SSM) models in a Transformer dominated environment together with works such as Tirex and FlowState.
* Solid results on the GIFT-Eval benchmark, which is well accepted in the TSFM community.
* Interesting combinations of common and custom waveforms, GP, and SDE's.
* Up to date comparisons.

### Weaknesses
* The rationale for combining the proposed synthetic components remains insufficiently justified. Given the diversity of these components and their varying parameter ranges, there is a concern that the authors may have curated them specifically to perform well on the chosen benchmark. Although some motivation is provided in Appendix Section A, this explanation lacks depth. To address this concern, the authors are encouraged to: (1) offer a stronger theoretical foundation or principled motivation for how these components interact and contribute to model robustness; (2) include additional evaluations on benchmarks such as LTSF [1,2] or Chronos/FEV [3,4] to demonstrate generalizability; or (3) provide other forms of analysis to convincing why this setup is essential.

It also appears from this work that the inclusion of any synthetic generator may positively contribute to the model’s success. If this is not the case, the authors should clarify why certain generators were chosen over others and demonstrate that the selected components are not arbitrarily beneficial but rather collectively essential and complementary. Without such justification, it remains unclear whether the performance gains stem from the specific design of the synthetic pipeline or simply from the presence of diverse synthetic signals.


* Generation overhead: The authors mention generating 10 million synthetic series but provide no analysis of generation time or comparison to existing methods like KernelSynth [3] or waveform-based approaches, leaving the efficiency of generation unclear.


* This is not the first work to challenge TSFM with real-data free training [5,6]. This work should also discuss the differences to Freq-Synth [6] or other methods in [7].

* This work isn't particularly better for long term forecasting as shown in Tab. 3.

### Questions
1) **Major**: Why are the results for TempoPFN in  Fig 4.  (MASE=0.797) different from the Base model in Tab.3 (MASE=0.842)?

2) In light of the first concern, why is sawtooth and sine waveforms used and not square/triangle waveform?

3) **Minor**: Have the authors tried comparing the model's performance trained with real data? for example training from LOTSA or datasets from the Monash repo? It may offer an interesting ablation study, measuring the model's performance alone


[1] Sundial: A Family of Highly Capable Time Series Foundation Models

[2[ Timer: Generative Pre-trained Transformers Are Large Time Series Models

[3] Chronos: Learning the Language of Time Series

[4] TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning

[5] From Tables to Time: How TabPFN-v2 Outperforms Specialized Time Series Forecasting Models

[6] Beyond Data Scarcity: A Frequency-Driven Framework for Zero-Shot Forecasting

[7] Empowering Time Series Analysis with Synthetic Data: A Survey and Outlook in the Era of Foundation Models

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce a time series foundation model (TSFM), called TempoPFN, based on linear recurrent neural networks that is pretrained solely on synthetic data.

### Strengths
The model of the authors is trained exclusively on synthetic data which is advantageous for zero-shot generalization on various benchmark tasks.
Also, the authors open-source their data processing and generation pipeline, which the authors can really appreciate, as it allows others to quickly build on top of this work and test out new research ideas.

### Weaknesses
Unclear description of the core architecture of the paper:\
Unfortunately, the authors describe their architecture in not much detail on roughly half a page (including a Figure). Instead, the author refer the reader to the paper of the DeltaProduct, which forms the core of this architecture. Since the authors present this as a novel model architecture, it would be good to at least have a better description of this critical building block. There appears to be more text on page 3 introducing and describing the linear RNNs than there is text describing the DeltaProduct. This makes the paper awkward to read, because if the reader is not familiar with the DeltaProduct, he is basically forced to look a the other paper to understand it.

Overly focused on TiRex:\
It feels like the work of the authors is overly focused on the comparison point with TiRex. Alone the term "TiRex" is mentioned over 80 times in the paper! and in a lot of places the reference feels awkward and artificial. Especially given that there are other SSM, or linear RNN-based architectures in the literature, some of which the authors put in their results tables, but don't even mention once in the text.

Extensive pretraining:\
The authors mention that "all ablation experiments were conducted for a limited budget of 500,000 iterations. It is crucial to note that this is only a tenth of the full training schedule for our main model". However, their apparent kryptonite enemy TiRex only pretrains for 500,000 in the first place, so in total ten times less than TempPFN, however, the authors remain silent about this point.

### Questions
Why the extreme fixation on TiRex? I find the paper per se nice, but this artificial comparison makes the paper read very awkward an distracts from the merits of the authors' own work.\
Why is so much pretraining needed? Is this the tradeoff one has to take when only dealing with synthetic data in the first place?\
Why was the DeltaProduct architecture chosen as the basis? It remains a bit unclear until the end why the authors didn't opt for any other architecture, such an an SSM-based architecture?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces TempoPFN, a univariate TSFM based on linear RNNs. It adopts a GatedDeltaProduct architecture that enables fully parallelizable training across sequence lengths. TempoPFN is entirely pre-trained on synthetic data. The data synthesis pipeline incorporates diverse generators, including SDEs, Gaussian processes, and audio synthesis, alongside novel data augmentation methods such as time-varying TSMixup, differentiation, and integration. In zero-shot evaluations on the Gift-Eval benchmark, TempoPFN achieves performance comparable to models trained on real-world data, while demonstrating significantly higher computational efficiency than existing baseline methods.

### Strengths
- Unlike the currently dominant Transformer architecture, this paper employs a linear RNN as its framework, achieving performance comparable to models with hundreds of millions of parameters while maintaining under 35M parameters itself.
- The paper introduces a comprehensive approach to data synthesis and augmentation, enabling the complete elimination of real-world data for model training. Combined with the compact model architecture, this significantly lowers the barrier for training and deploying TSFMs.

### Weaknesses
- The paper employs GatedDeltaProduct, which is a relatively novel network architecture. However, the paper lacks a clear explanation of this architecture and justification for its selection.
- The authors mention the concepts of "state tracking" and "state weaving".  However, the authors neither define these terms nor clarify their relevance to time series forecasting tasks. This reflects a broader trend in the field, where methodologies from LLM and computer vision are frequently adopted without a formal discussion of their applicability to time series data. In such cases, sequences are often treated as textual or visual analogs without rigorous justification. 
- There are some writing flaws: an incomplete sentence at line 194, duplicated references at lines 540 and 544.

### Questions
- In Figure 2, the input of TempoPFN includes both historical and future time indices, specifically $t_0, t_1, \cdots, t_5$ in the Figure. Do these time indices contain granular information such as year, month, and day as mentioned in line 176? Given that all data used in this study are synthetically generated, how are these temporal attributes actually assigned?
- At line 202, could the authors elaborate on how TempoPFN achieves probabilistic forecasting for arbitrary future horizons?
- TempoPFN is a foundation model trained entirely on synthetic data, which means the diversity of synthetic data is critically important. Do the authors have a methodology to evaluate how well their proposed data synthesis approach can cover real-world data distributions? Additionally, what guiding principles should inform the construction of such data synthesis methods?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper TempoPFN addresses the task of zero-shot time series forecasting, which is an importance task in the time series domain, and recently getting popularity. The paper has 2 main contributions. It trains linear RNNs, recently proposed in literature, only on synthetic data and evaluate on a exhaustive benchmark GIFT-Eval. The paper proposes a pipeline to generate a diverse set of synthetic data.

### Strengths
1. Presentation of the paper is good.
2. The synthetic data generation pipeline is pretty exhaustive, and covers many types of synthetic data.
3. The author(s) promise to open source the codes and pipelines.

### Weaknesses
However, the paper has a few weaknesses.
1. The novelty of the paper is limited. First, linear RNNs are not new, they are adopted from the literature. Training models purely on synthetic data is not new; the paper just creates a more diverse set of synthetic data. Given the idea of TabPFN or ForecastPFN, the proposed work can be tried quite trivially, without much complications.
2. The performance of the model is not up to the mark and marginally better than TabPFN-TS in CRPS and (from 0.544 to 0.536), but worse in MASE (0.771 to 0.797) after including significantly more and diverse set of synthetic data. This questions the efficacy and strength of the proposed work. The method is significantly worse than other methods that use real data alongwith synthetic data. 
3. "We developed several new generators to fill gaps in existing approaches and capture specific temporal behaviors." -- There are infinite ways to generate time series, why only these types? Is there any real motivation behind the time series generation?
4. Comment: The paper title is TEMPOPFN where PFN stands for Prior Data Fitted Networks. The paper should discuss PFNs (at least briefly) for the readers who are unaware about it. I did not find any mention of PFNs in the methodology.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1

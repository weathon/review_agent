# From Internal Representations to Text Quality: A Geometric Approach to LLM Evaluation

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
This paper bridges internal and external analysis approaches to large language models (LLMs) by demonstrating that geometric properties of internal model representations serve as reliable proxies for evaluating generated text quality. We validate a set of metrics—including Maximum Explainable Variance, Effective Rank, Intrinsic Dimensionality, MAUVE score, and Schatten Norms measured across different layers of LLMs, demonstrating that Intrinsic Dimensionality and Effective Rank can serve as universal assessments of text naturalness and quality. Our key finding reveals that different models consistently rank text from various sources in the same order based on these geometric properties, indicating that these metrics reflect inherent text characteristics rather than model-specific artifacts. This allows a reference-free text quality evaluation that does not require human-annotated datasets, offering practical advantages for automated evaluation pipelines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the relationship between the geometric properties of internal representations within Large Language Models (LLMs) and the perceived quality (specifically, "naturalness") of the text they generate. The authors propose using a set of "tester" LLMs to analyze text produced by different "generator" LLMs. By extracting hidden states from the tester models processing the generated text, they calculate several geometric metrics: Schatten Norms, Maximum Explainable Variance (MEV), Effective Rank (ERank), Resultant Length, MAUVE (applied to internal representations), and various Intrinsic Dimensionality (ID) estimators (MLE, MOM, MADA, CorrInt).

### Strengths
The paper effectively connects the often separate fields of mechanistic interpretability (analyzing internal representations) and black-box evaluation (assessing output text quality). Showing that internal geometric properties reliably predict external quality is a significant conceptual contribution. The demonstration that diverse tester models (different sizes, architectures like autoregressive vs. diffusion) yield highly consistent rankings (Spearman > 0.947) for generator models based on internal metrics is a powerful result. This strongly supports the claim that these metrics capture inherent text properties.

### Weaknesses
The correlation analysis between the internal geometric scores and external text quality metrics (Figure 4) relies on scores aggregated across tester models, with the number of data points being the number of generator models (plus human text). As acknowledged by the authors, this limits the statistical power and prevents robust p-value calculation. Correlations based on individual text samples might provide stronger evidence.

### Questions
Regarding the correlation analysis in Figure 4: Would it be feasible to compute correlations at a finer granularity (e.g., per text sample, averaging metrics across layers but not across generators) to obtain more data points and increase statistical confidence in the relationship between internal geometry and external quality metrics?

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
5

### Summary
The paper evaluates many metrics, including Schatten norms, Maximum Explainable Variance (MEV), Resultant Length, Effective Rank (ERank), Intrinsic Dimensionality (ID) variants, and MAUVE, based on text embedding from LLMs averaged across layers. The paper argues that ERank and ID are the most reliable “universal” indicators. Across six tester models (0.5B–8B), the authors report consistent rankings of generators’ text, suggesting these metrics capture properties of the text rather than idiosyncrasies of a particular tester. They also report correlations between internal geometry and external text metrics (e.g., BLEURT, GPT-PPL), and explore English, German, Russian movie-review paraphrases as a testbed.

### Strengths
1. Internal geometry of a small tester model can diagnose properties of text produced by other models. This enables reference-free evaluation, which is very important to supplement classic reference-based scores like ROUGE.
2. Rankings of generators are broadly consistent across heterogeneous tester models. This is actually very interesting observation I didn't expect. 
3. The paper formalizes and compares many geometry metrics (MEV, Resultant Length, ERank, ID variants, Schatten norms) with precise definitions and layer-wise/averaged reporting. 
4. The paper also presents comprehensive results of Spearman correlations between geometry and BLEURT/GPT-PPL/MAUVE. 
5. It also extends to non-English text (German, Russian).

### Weaknesses
1. The evaluation is built around paraphrasing movie reviews to “synthetic” text with preserved meaning/length. This seems very narrow. Can you apply on real-world conversation prompt dialogue, such as UltraChat [1]? My concern is that the results in the paper are mainly due to dataset characteristics rather than text quality in general.
2. The results could cofound with each LLM text preprocess. In fact, if you generate responses with batch inference (like using vLLM), the padding could affect the resulting embedding. The individual embedding for a specific text would be different from that if using native inference one by one. How do you handle the artifacts in tokenization and padding for each LLM.
3. LLM-as-a-judge becomes a popular approach to judge generation quality. It is also found that it very correlates with human judgement [2]. How does any of the metrics evaluated in this paper correlate with LLM-as-a-judge. It would also investigate how the judge model would prefer its own generation, or the embedding is more model-agnostic metric?

References:
1. Ding et al. Enhancing Chat Language Models by Scaling High-quality Instructional Conversations
2. Chiang et al. Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference. ICML 24'

### Questions
1. How do the metrics studied in this paper correlate with one another? I think this investigation brings more value compared with PPL or AVG Len.
2. Also the different models yield identical rankings of texts generated by various generator. This seem contradictory with results self preference bias [1] using an LLM-as-a-judge. Is it because the embedding more model-agnostic to the input? Can you provide some explanations to reconcile the gap.
3. Recent studies [2, 3] find that generated text from different LLMs based on the same prompt are idiosyncratically different in word choices, semantics, and style. How do you see your findings on diversity metrics can be used as a tool to quantify the degree of difference in these cases. This can also be extended to detecting AI vs machine-generated text, or which LLM generates a specific text. 

Reference:
1. Wataoka et al. Self-Preference Bias in LLM-as-a-Judge.
2. Sun et al. Idiosyncrasies in Large Language Models. ICML 25
3. Dunlap et al. Vibecheck: Discover and quantify qualitative differences in large language models. ICLR 25

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes to evaluate the quality of text generated by large language models (LLMs) through geometric analysis of internal model representations.Specifically, the authors compute several geometric metrics—Maximum Explainable Variance (MEV), Effective Rank (ERank), Intrinsic Dimension (ID), Schatten Norm, and MAUVE—from the hidden states of multiple “tester” LLMs while processing text generated by eight “generator” LLMs. The main observation is that different tester models (ranging from 0.5B to 8B parameters) produce nearly identical rankings of generator quality based on these geometric metrics. The authors further report that some of these metrics (notably ERank and ID) correlate with conventional text quality measures such as BLEURT and GPT Perplexity, claiming that these geometric features can serve as reference-free proxies for text naturalness.

### Strengths
1.The goal of establishing reference-free automatic text evaluation is important and timely, given the rapid scaling of LLMs and the impracticality of human annotation.
2.The idea of connecting internal geometry of representations with external text quality is conceptually interesting
3. The study consistently compares multiple generators and testers across languages, and provides explicit mathematical definitions for all geometric metrics.

### Weaknesses
1. The work primarily aggregates existing geometric and evaluation metrics (ERank, ID, MAUVE, MEV) and computes correlations.It reads more like an experimental survey report than a scientific contribution.The paper lacks a coherent theoretical foundation or an explanatory mechanism linking geometry to text quality.
2. The key claim—that different tester models produce similar rankings of generator quality—is unsurprising and of modest insight.Such ranking consistency could trivially arise from shared lexical or fluency statistics rather than any intrinsic geometric property.
3. The study does not verify whether the metrics can capture fine-grained differences in text quality, thus the proposed framework lacks predictive utility. And no variance estimation, confidence intervals, or sensitivity analysis are reported.
4. The Introduction section is logically confused, mixing philosophical remarks (“beauty vs. utility”) with technical motivations. It fails to establish a clear problem statement, research question, or hypothesis.
5. The authors claim that Intrinsic Dimension and Effective Rank are “universal” quality measures, yet this is based on small-scale correlations from limited data.Non-English experiments even contradict their claim (smaller human–synthetic gap), but this inconsistency is ignored rather than analyzed.

### Questions
1.	Why were relatively small open-source models (e.g., Starling, Phi, Mistral) selected instead of stronger or more diverse systems like GPT-4, Claude, or Gemini?
2.	How do you ensure that the observed ranking consistency across tester models is not driven by superficial lexical or stylistic cues (e.g., token frequency, sentence length) rather than intrinsic geometric structure?
3.	Did you analyze which layers are most correlated with text quality instead of simply averaging across all layers? Could layer-specific behavior be more informative?
4.	Have you tested whether the geometric metrics are stable across random seeds, prompts, or domains?

### Soundness
2

### Presentation
2

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
Previous approaches using LLMs (referred to as test model $T$ in this paper) to evaluate the quality of text generated by other LLMs (referred to as generation model $G$ in this paper) typically require task-specific and scenario-specific customized annotation guidelines and metrics. Some methods also necessitate annotating reference datasets for training the evaluation model $T$ or providing reference responses.

This paper argues that such tedious customization for each individual task is suboptimal. Therefore, the authors propose leveraging the internal states of model $T$, using geometric characteristics of these internal representations as the basis for ranking the quality of external text generated by $G$. The paper primarily employs metrics including Maximum Explainable Variance, Effective Rank, Intrinsic Dimensionality, MAUVE score, and Schatten Norms.

Experimental results demonstrate that ranking text generated by $G$ based on the internal states of $T$ produces consistent text quality orderings across different $T$ models, indicating the reliability of the proposed internal state metrics for measuring external text quality.

### Strengths
The proposed approach of utilizing the internal states of model $T$ to evaluate the quality of text generated by $G$ is well-motivated and promising. It can effectively alleviate the burden of task-specific customization. However, the paper lacks discussion on a comprehensive range of text quality dimensions, which limits its broader applicability.

### Weaknesses
1. Insufficient definition of text quality: The paper lacks a rigorous and formal definition of text quality. Relying solely on "naturalness" as the text quality indicator is inadequate. Given that the authors criticize task-specific evaluation criteria as suboptimal, they should provide a method that is universally applicable across different tasks with effective generalization capabilities. However, the paper fails to clearly demonstrate through experiments which specific aspects of text quality each metric captures.

2. Limited task scope: Line 161 mentions using model $G$ for text paraphrasing. This raises the question of whether the proposed method is merely an evaluation metric for paraphrase tasks and cannot generalize to other text generation tasks.

3. Lack of baseline comparisons: The paper does not compare the proposed metrics with existing evaluation methods under the same experimental settings as shown in Figures 1 and 2.

4. Missing critical details:
    1. What decoding parameters (e.g., temperature, top-p, top-k) are used when model $G$ performs paraphrasing?
    2. What is the distribution of text lengths in the evaluation? Is the method effective for longer texts? This is particularly important given that current LLMs routinely generate thousands or even tens of thousands of tokens.

### Questions
1. The paper employs multiple metrics for evaluating internal representations. Could the authors clarify: (a) What are the motivations and theoretical foundations for selecting these specific metrics? (b) What distinct aspects of text quality does each metric capture? (c) Are these metrics collectively comprehensive in covering different dimensions of text quality, or are there potential gaps?

### Soundness
2

### Presentation
2

### Contribution
2

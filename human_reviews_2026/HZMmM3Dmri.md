# Neuron-Level Analysis of Cultural Understanding in Large Language Models

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
As large language models (LLMs) are increasingly deployed worldwide, ensuring their fair and comprehensive cultural understanding is important.
However, LLMs exhibit cultural bias and limited awareness of underrepresented cultures, while the mechanisms underlying their cultural understanding remain underexplored.
To fill this gap, we conduct a neuron-level analysis to identify neurons that drive cultural behavior, introducing a gradient-based scoring method with additional filtering for precise refinement.
We identify culture-general neurons contributing to cultural understanding regardless of cultures, and culture-specific neurons tied to an individual culture.
Culture-general and culture-specific neurons account for less than 1% of all neurons and are concentrated in shallow to middle MLP layers.
We validate their role by showing that suppressing them substantially degrades performance on cultural benchmarks (by up to 30%), while performance on general natural language understanding (NLU) benchmarks remains largely unaffected.
Moreover, we show that culture-specific neurons support knowledge of not only the target culture, but also related cultures.
Finally, we demonstrate that training on NLU benchmarks can diminish models' cultural understanding when we update modules containing many culture-general neurons.
These findings provide insights into the internal mechanisms of LLMs and offer practical guidance for model training and engineering.
Our code is available at https://github.com/ynklab/CULNIG

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces CULNIG, a gradient-based pipeline for identifying neurons that support cultural understanding in large language models. By leveraging a BLEnD control dataset and a CountryRC filter, CULNIG isolates “culture-general” and “culture-specific” neurons while excluding those merely responsive to country names or task formats. The authors find that less than 1% of neurons, mainly in shallow-to-mid MLP layers, are responsible for most culture-related behavior, and masking them causes a 30% performance drop on cultural benchmarks with minimal NLU impact. They further show that culture-specific neurons affect historically or linguistically related cultures and that fine-tuning culture-rich modules can erode cultural ability, highlighting implications for reliable model adaptation.

### Strengths
(1) This paper employs dual control datasets (BLEnDctrl, CRC) to eliminate spurious lexical or task-format activations.

(2) This paper provides clear empirical evidence that <1% of neurons drive most cultural behavior. And reveals layer-level localization of cultural knowledge in shallow–mid MLPs.

(3) Demonstrates cross-model and cross-benchmark robustness across multiple open LLMs.

(4) Code and data with a detailed README file are shared.

### Weaknesses
(1) The paper’s definition of “cultural understanding” is narrow and tied to country-labeled QA tasks, which may not capture deeper cultural reasoning or values.

(2) This paper conflates cultural knowledge with cultural alignment (norms, values, empathy), yet reports all under a single “culture-general” construct.

(3) The paper does not clearly articulate why gradient-based neuron analysis is superior to attention- or concept-level analysis for cultural representation.

(4) A key limitation is that the application study only shows a negative case—fine-tuning culture-rich modules harms cultural competence—without demonstrating how the identified neurons can be leveraged to enhance or transfer cultural understanding.

### Questions
(1) Can the identified culture-related neurons be leveraged to improve or transfer cultural understanding, rather than only showing degradation when masked or fine-tuned?

(2) How sensitive are the results to the chosen thresholds (percentiles, z-scores, filtering ratios)? 

(3) How do you distinguish between neurons representing knowledge (facts about culture) versus values or norms? Could a separate analysis clarify this conceptual distinction?

(4) Prompt robustness: You mention using multiple prompt formats—what is the variance in neuron identification and benchmark scores across these prompts?

(5) For culture-specific neurons, do related cultures (e.g., neighboring or linguistically similar) consistently show correlated effects? Could you quantify this relationship?

### Soundness
3

### Presentation
3

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
This paper investigates the internal mechanisms underlying cultural understanding in large language models (LLMs). While previous studies have documented cultural bias or disparities in model outputs, little is known about how LLMs internally represent and process cultural information. To address this, the authors propose CULNIG (Cultural Neuron Identification Pipeline with Gradient-based Scoring)— a neuron-level interpretability framework that uses gradient attribution combined with control datasets to identify neurons responsible for cultural reasoning.
CULNIG distinguishes between culture-general neurons and culture-specific neurons. Through extensive experiments on six open-source LLMs, the authors show that these neurons constitute less than 1% of all neurons, are concentrated in shallow-to-middle MLP layers, and that masking them leads to substantial degradation (up to 30%) on cultural benchmarks such as BLEnD, CulturalBench, NormAd, and WorldValuesBench—while general NLU benchmarks remain largely unaffected.
The paper also presents an engineering case study showing that fine-tuning modules rich in culture-general neurons causes a loss of cultural understanding, suggesting a practical pathway for parameter-efficient and culturally robust fine-tuning.

### Strengths
The study tackles an important and underexplored question: how LLMs internally encode culture. Given the growing concerns around fairness and cross-cultural bias in LLMs, this direction is highly relevant to the ICLR community.

CULNIG combines gradient-based attribution with two control datasets to precisely isolate neurons genuinely related to cultural inference. This design elegantly avoids the pitfalls of activation-based methods, which often conflate linguistic or topical cues with cultural representations.

The evaluation spans multiple models, benchmarks, and modalities. The consistency of results across architectures strongly supports the validity of the findings. The experiments are well-documented and reproducibility is addressed.

The finding that culture-related neurons cluster in mid-level MLP layers aligns with prior “knowledge neuron” research, suggesting that cultural reasoning may rely on factual recall mechanisms rather than contextual inference. The observed cross-cultural interference (e.g., Mexico–Spain) provides additional interpretive depth.

### Weaknesses
CULNIG relies primarily on BLEnD, which emphasizes everyday cultural knowledge (e.g., food, work, sports). Although the authors claim generalization to higher-level cultural attributes such as values or etiquette (NormAd, WVB), Section 4.4 admits that the pattern is “less clear” for NormAd. This suggests that the current pipeline is more effective at identifying *cultural knowledge neurons* than *value-oriented or normative neurons*, and that the captured concept of “culture” may be biased toward knowledge representation rather than deeper socio-cognitive dimensions.

Key thresholds in CULNIG are selected heuristically based on inflection points in Figure 2. The main conclusion—that fewer than 1% of neurons govern cultural understanding—could depend on these particular values. A sensitivity analysis (e.g., varying thresholds from 0.5% to 2%) would strengthen claims about sparsity and robustness.

The reported layer distribution (shallow-to-middle) contradicts CAPE, which located culture neurons mainly in upper layers. Although the authors hypothesize methodological causes (gradient attribution vs. activation probability, accuracy vs. perplexity metrics), the inconsistency remains unclarified, leaving open a significant question for the field.

The paper defines these neurons functionally—as those beneficial across cultural tasks—but does not explore *what they actually compute*. Do they encode an abstract notion of “cultural context,” or simply a meta-cognitive control signal for out-of-context reasoning? A deeper probe into their semantic or computational roles would improve interpretability.

### Questions
1. CULNIG mainly relies on BLEnD${\mathrm{neur}}$ (a knowledge-oriented dataset) for neuron identification. As observed in Section 4.4, culture-specific neurons show weaker patterns on NormAd (etiquette/value tasks). Does this suggest that “cultural knowledge” and “cultural values” are encoded by different neurons or mechanisms? If CULNIG were re-run using NormAd or WVB as $D{\mathrm{neur}}$, would you expect to discover a distinct set of neurons?
2. The main claims—such as $ <1% $ sparsity and the clear separation between cultural and NLU functionality—depend on the thresholds $ t_{\text{MLP}} = 1% $ and $ t_{\text{attn}} = 0.2% $ selected in Section 4.2. How stable are these findings under threshold variation? For example, if $ t_{\text{MLP}} $ increases to $ 2% $ or $ 3% $, would NLU tasks (e.g., QNLI) start to degrade? Such analysis would clarify the boundary between cultural and general-purpose neurons.
3. These neurons generalize across tasks and attributes, but beyond their functional importance, do you have any hypotheses about what they compute? Do they encode high-level “cultural semantics” or act as meta-reasoning units that signal when external knowledge is needed? Clarifying this would help bridge the gap between functional discovery and mechanistic interpretation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper uses a gradient-based attribution metric to identify neurons relevant for models’ performance on global/cultural understanding benchmarks. The authors are able to find both cultural-general neurons and cultural-specific ones. They also show that modular fine-tuning of LMs informed by locations of culture-relevant neurons can avoid diminishing LMs’ cultural understanding.

### Strengths
Overall, this paper is well-written and organized in a logical manner. 

During several points in the paper, I saw that there was some care involved in the authors’ experimental decisions. For example, to show that their findings around culture-related neurons generalize, the authors split their multiple choice question dataset into train-like and test sets based on topical categories (e.g. if a neuron pertains to culture, its presence in food questions should generalize to its presence in holiday questions). As additional examples of care, the authors also experiment with multiple prompt variations and also account for cases where neurons may pertain to surface level features rather than culture, e.g. country names. 

I appreciate that the authors dedicate a section of their paper to applications of their findings. Often, a pitfall of mechanistic interpretability research is that though its findings may satisfy one’s curiosities around the inner workings of language models, the practical implications of such findings are not always demonstrated.

### Weaknesses
This paper focuses on identifying culture neurons, but the paper doesn’t really give us much insight that justifies this conceptual focus. The paper reuses existing methods from Yang et al. 2024 to identify neurons, and its application of its findings (module section for fine-tuning) also doesn’t seem to need to be culture-specific. What makes identifying and working with culture neurons unique from identifying other types of neurons? It’s good that it seems like you managed to find them, but what does your work offer us that identifying any other kind of topical neuron doesn’t offer? 

The authors’ last sentence of their conclusion also hints at some of my qualms: “While our findings do not directly improve the cultural understanding of LLMs, they provide a foundation for future studies to do so.” More evidence of this foundation and its implications specifically for culture would make this paper stronger.

### Questions
I wasn’t quite sure why there was so much space dedicated to a discussion of how transformer neurons work in 3.1?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper investigates how cultural understanding is represented at the neuron level in LLMs. It introduces CULNIG (CULture Neuron Identification with Gradient-based Scoring), a pipeline for identifying neurons that contribute to cultural understanding. Using gradient-based attribution and multiple control datasets, the authors isolate Culture-general neurons and Culture-specific neurons, and show that masking them reduces performance on cultural benchmarks but not on general NLU benchmarks, masking culture-specific neurons affect both the target and related cultures on cultural benchmarks, and fine-tuning modules containing many culture-general neurons on general NLU tasks reduces cultural understanding.

### Strengths
1. First systematic neuron-level investigation of cultural mechanisms in LLMs, extending prior interpretability work (bias, language, factual neurons)
2. CULNIG uses a rigorous multi-step filtering pipeline (BLEnD neur – ctrl – CRC control) to remove spurious correlations
3. Generalization of result on six modern open source models and across benchmarks
4. Practical insight demonstrating the effect of fine-tuning on cultural competence depending on target modules

### Weaknesses
1. Ambiguous scope of "cultural understanding": all benchmarks have different coverage of cultures and definition of cultural values/knowledge, the authors should clarify on this work's scope of cultures and definitions of cultural understanding.
2. Limited interpretability: the paper identifies neuron clusters but does not probe what these neurons encode semantically (e.g., cultural values vs. lexical associations).
3. Fine-tuning only tests QNLI/MRPC, so it's unclear how results scale to other downstream tasks or general instruction tuning.
4. Country-token filtering (CRC) may not fully remove lexical signals and cultural context could still leak through token patterns, so the authors should have a discussion/ablation/evaluation on this.

### Questions
1. If masking culture-general neurons doesn’t affect NLU, does that imply they’re unused for NLU or because of neuron and layer redundancy?
2. Definition of “cultural understanding” needs to be clearer. The paper mixes factual, normative, and value-based culture; boundaries between these types are not operationally clean.
3. Prompting procedure – The role of ChatGPT-generated prompts for CRC/BLEnD not clearly justified (possible data leakage or stylistic bias).
4. Authors claim generalization “to multilingual settings,” but examples and quantitative evidence are minimal.
5. Concentration in shallow/mid layers aligns with knowledge neurons, but it’s not clear whether this overlaps with existing factual memory neurons.

### Soundness
3

### Presentation
3

### Contribution
3

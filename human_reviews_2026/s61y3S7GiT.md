# Did You Faithfully Say What You Thought? Bridging the Gap Between LLMs Neural Activity and Self-Explanations

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Large Language Models (LLMs) can generate plausible free text self-explanations to justify their answers. However, these natural language explanations may not accurately reflect the model's actual reasoning process, indicating a lack of faithfulness. Existing faithfulness evaluation methods rely primarily on behavioral tests or computational block analysis without examining the semantic content of internal neural representations. This paper proposes NeuroFaith, a flexible framework that measures the faithfulness of LLM free text self-explanation by identifying key concepts within explanations and mechanistically testing whether these concepts actually influence the model's predictions. We show the versatility of NeuroFaith across 2-hop reasoning and classification tasks. Additionally, a linear faithfulness probe based on NeuroFaith is developed to detect unfaithful self-explanations from representation space and improve faithfulness through steering. NeuroFaith provides a principled approach to evaluating and enhancing the faithfulness of LLM free text self-explanations, addressing critical needs for trustworthy AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces NeuroFaith, a way to measure faithfulness of LLM self-explanations (in natural language). NeuroFaith identifies concepts mentioned in a model’s explanation and tests whether these concepts actually influence the model’s predictions through mechanistic analysis of internal activations. The framework is instantiated for both 2-hop reasoning and classification tasks, using techniques like Patchscopes to quantify alignment between explanations and internal reasoning. Experiments on Gemma-2 models show mixed results, but generally that larger or more accurate models tend to produce more faithful explanations. Finally, the authors use NeuroFaith to label examples for high and low faithfulness, compute steering vectors from that to increase LLM faithfulness.

### Strengths
* The paper is well-motivated, clearly identifying the limitations of current faithfulness evaluation methods and convincingly taking related work's argument that assessing faithfulness requires examining the model’s internal neural representations, not just behavioral agreement. This makes NeuroFaith is conceptually elegant, combining interpretability and faithfulness measurement in a unified, mechanistic approach that grounds explanations in internal model activity.
* The experiments cover multiple model scales (Gemma-2 2B–27B), which provides evidence into how faithfulness increases with scale.
* The faithfulness steering sections go beyond evaluation to show practical ways to improve explanation faithfulness at inference time.

### Weaknesses
* **Lack of metric validation:** The paper does not empirically validate how well NeuroFaith correlates with existing faithfulness measures or human judgments of explanation quality. It would strengthen the claim of “measuring faithfulness” to compare against recent dedicated ways to even automatically test how much we can trust faithfulness metrics: https://arxiv.org/abs/2502.18848
* **Not comparing against other faithfulness metrics:** The evaluation relies primarily on custom experimental setups rather than established benchmarks (e.g., Parcalabescu & Frank, 2024 cited by this paper or https://arxiv.org/abs/2502.18848), making it difficult to assess how NeuroFaith performs relative to prior metrics. This continues the trend of isolated metrics without direct comparison, which limits the interpretability of reported results.
* **Weak model type coverage:** The study is restricted to the Gemma-2 model family (2B–27B), leaving open how general the findings are to other architectures or training paradigms (e.g., LLaMA, Mistral, Vicuna, Qwen3, Deepseek, etc).
* **Unstudied effect of the interpretability method**: The framework’s results likely depend on the chosen interpretability method, but the paper does not systematically analyze this dependency or the potential unfaithfulness of these underlying tools.
* **Unquantified effect of llm as a judge:** The reliance on LLM-as-a-Judge for concept extraction could introduce biases that propagate into the faithfulness score, yet the paper only briefly acknowledges this limitation without quantifying its impact.

### Questions
Do you believe that activation steering alone is sufficient for improving faithfulness, or would it be worthwhile to explore reinforcement learning or fine-tuning using NeuroFaith as a reward signal?


Suggestion: In the abstract you use passive voice in "Additionally, a linear faithfulness probe based on NeuroFaith is developed to detect unfaithful self-explanations from representation space and improve faithfulness through steering." It makes it unclear whether you did this, or someone else or whether this is left to future work. Passive voice is bad writing style because it hides the actant.

### Soundness
2

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
4

### Summary
A framework is proposed where self-generated natural language explanations of models are evaluated on their faithfulness through interpretability techniques. Specifically it is proposed to extract from each explanation a set of concepts that it evokes; then, one of a variety of interpretability techniques can be used to see if those concepts are present in the latent states of the model. The degree to which they are present is used to estimate faithfulness.

A number of experiments are performed to show how this framework can be instantiated. These experiments generally show that: (1) self-explanations are more faithful when they come from larger models, (2) explanations of accurate predictions (as opposed to inaccurate ones) are more faithful, and (3) faithfulness depends on task.

This measure of faithfulness is further used to look for a direction in the models' latent spaces which correlates with it. Such directions are found, and depending on the task, they predict faithfulness with F1 scores roughly between 60 and 75. 
Finally, the directions are used to steer the model towards producing more faithful explanations.

### Strengths
The use of interpretability techniques to evaluate the faithfulness of (self-)explanations makes a lot of sense, and the paper makes sensible choices in pursuing this.

The presentation is generally good, clear language, diagrams, tables, figures, and sensible sectioning.

High information density, multiple experiments using different interpretability techniques to showcase the versatility of the framework.

### Weaknesses
Lines 457-459 state that 'Direct faithfulness amplification consistently outperforms hallucination and deceptivenes inhibition', but from Table 3 that looks like the wrong take-away: Hallucination inhibition performs very similarly to faithfulness amplification, especially for the smaller models. Looking at the cosine similarities in the appendix, we see something similar: for smaller models the faithfulness direction is not negatively correlated to the same extent. That also seems worth mentioning in the paragraph of lines 409-431.

Potentially some information missing (see question 2).

### Questions
1. Does the statistical significance calculation for Figure 3 take into account the fact that you are selecting (taking max) over many different layers, thereby increasing the chances of getting lucky?

2. Is the steering done in the layer where the faithfulness direction was most predictive of faithfulness? or some other selection of layers? What about tokens, is the steering vector added to all tokens? What about the hallucination and deceptiveness baselines?

3. What data is used for calculating the concept-activation vectors described in lines 314-323? Generally speaking, do we always have enough of this data for each concept that might be detected? Also, a fixed set of concepts $\mathcal{C}$ is assumed, where do these come from?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies faithfulness: do the natural language explanations (NLEs) models produce for their own decisions actually reflect their true reasoning process?

The paper uses methods based on internal model representations to determine whether certain concepts are represented (via probing) and/or important (via representation engineering). In particular, the paper runs two sets of experiments:

2-hop reasoning: the paper evaluates three Gemma models on a 2-hop reasoning task. It assesses whether models contain latent representations of these bridge entities while solving the questions, and also whether self-NLEs contain these bridge entities.

Classification: the paper evaluates three Gemma models on a classification task, and has them produce self-NLEs. It extracts relevant concepts from these NLEs using Qwen3-32B. It then determines the causal influence of the concepts by erasing their vectors during forward propagation, to determine whether the concept was actually necessary to complete the task.

Finally, the paper tests interventions for improving model faithfulness: steering models toward faithfulness vectors determined by the paper’s own faithfulness evaluations, as well as vectors for hallucination and deceptiveness from prior work.

### Strengths
1) Using mechanistic methods to determine faithfulness is a promising approach: it has the potential to give a better view into the model’s actual decisionmaking process, so that we can check whether explanations are accurately representing this process.

2) The paper combines a number of interesting techniques from the mechanistic interpretability literature in novel and creative ways.

3) The steering results are interesting: it would be very cool if there was a linear vector which could generally increase model faithfulness! Evaluating the faithfulness vector on the same tasks used to determine its direction risks Goodharting, but it’s a cool result that steering away from hallucination and deceptiveness vectors from prior work also improves performance on the paper’s own faithfulness metrics.

### Weaknesses
1) I often found it hard to figure out exactly what the paper’s experiments are doing. For example, Section 3 is highly abstract, and I came away not being sure what had actually been described, other than “extract concepts and measure their presence and/or importance mechanistically”. I think the formality and notation in this section is obfuscating more than it’s clarifying.

This also makes section 4 and 5 hard to follow, since they need to spend time explaining how their components map to the notation described in section 3, instead of just explaining how the specific instantiation works. The paper might be clearer by presenting the two methods more independently, rather than attempting to define the NeuroFaith framework to encapsulate both of them.

2) The assumption behind the 2-hop experiments is that “For f to correctly answer, it must internally compute the bridge object during the first hop before executing the second hop.” This isn’t always true: LLMs may develop shortcuts, e.g. by encountering head and answer entities in the same sequence, or guessing the answer with frequency-based priors. See Yang et al. 2025, e.g. Table 2 showing that some datasets can have very high shortcut prevalence (https://arxiv.org/abs/2411.16679).

But if a model ends up with the correct answer via a shortcut, rather than via the bridge entity, self-NLE “faithfulness” isn’t really a faithfulness metric, since the true faithful explanation of the model’s internal process would be the shortcut.

Indeed, Table 1 from the authors’ paper shows that for Gemma models, when the model’s prediction is accurate, “Latent Hop 1 Correctness” is between 48%-65%. IIUC, this means that in 35-52% of examples where the model ends up with the correct answer, the mechanistic interpretability method is unable to identify coordinates in the circuit such that the natural language description of those coordinates include the concept of interest. This seems consistent with frequent shortcut usage. (Though it’s also consistent with a failure of the latent extraction method to correctly identify feature usage.)

If you want to rely on models actually depending on bridge entities, you should evaluate using a dataset which explicitly filters out shortcuts, e.g. SOCRATES (https://github.com/google-deepmind/latent-multi-hop-reasoning). This would also give more signal on whether the latent extraction method is functioning correctly.

3) The 2-hop results use probing to detect concept presence. But probing alone doesn’t show causation: a bridge entity might be present in the residual stream, but ignored by later layers in favor of a shortcut. If the goal is to determine the model’s actual process, why use probing rather than concept importance?

4) The paper’s experiments on classification rely on Qwen3-32B to extract concepts from the self-NLE, but does it actually perform this job reliably?

More generally, the paper’s methods make use of a number of components, but it’s unclear to what extent we can trust each component. If the paper’s methods give surprising results, can we trust this to be a genuine finding? Or could it just be a failure of one of the individual components, e.g. the LLM doing concept extraction? How could we tell?

5) The paper’s tables and figures are mostly lacking confidence intervals or statistical tests (other than Figure 3), so it’s unclear which group differences are significant and which may be noise.

### Questions
1) “We leverage this information to define circuit Γ and generate the natural language description  ̃hℓk of hidden state hℓk” - how are you generating this natural language description? I’ve reread paragraph “Probing-based Concept Interpretation” but I’m having difficulty figuring out which specific method you’re using.

2) The classification experiments extract concepts from the self-NLE via Qwen3, but how were these mapped to internal representations? If you’re using a concept mapping from existing work, can you explain the form of that mapping, and how it was generated?

More generally, there are a lot of components being used from prior work; I think it would help readability to give more description of these methods.

3) The paper provides examples in appendices G and H; I think it would substantially clarify the presentation to include at least one example for each methodology in the main paper.

4) I’m struggling to understand the “Importance-based Concept Interpretation” paragraph: how are you actually arriving at the score determining whether a given example is faithful or not?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes NeuroFaith for evaluating the faithfulness of self-NLEs from large language models. The authors demonstrate this framework's flexibiity on both 2-hop reasoning and classification tasks. They then successfully manipulate using activation steering to improve explanation faithfulness at inference time, offering a path toward more transparent models.

### Strengths
1.The authors propose a reasonable faithfulness definition, checking if the model's explanation aligns with its actual computation and successfully improve the model's faithfulness via activation steering. 

2.The framework is successfully applied to both 2 hop reasoning tasks and classification tasks.

### Weaknesses
As mentioned in conclusion, the faithfulness score depends on what the auxiliary model considers a concept, which could introduce biases or blind spots. The authors also concede that circuit selection 'requires domain expertise or expensive circuit discovery', which undermines the framework's scalability and ease of application to new tasks

### Questions
1.Did the authors measure the corresponding change in task accuracy for the steered examples? Does this intervention also happen to correct the final prediction, or does it merely make the explanation for the wrong answer more faithful?

2.How sensitive are your final faithfulness scores to the choice of the auxiliary LLM used for concept extraction? If you were to use a different judge, how much would the F(x, e) scores and the resulting steering vector change?

### Soundness
2

### Presentation
2

### Contribution
2

# Do Vision-Language Models Reason Like Humans? Exploring the Functional Roles of Attention Heads

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Despite excelling on multimodal benchmarks, vision–language models (VLMs) largely remain a black box. In this paper, we propose a novel interpretability framework to systematically analyze the internal mechanisms of VLMs, focusing on the functional roles of attention heads in multimodal reasoning. To this end, we introduce CogVision, a dataset that decomposes complex multimodal questions into step-by-step subquestions designed to simulate human reasoning through a chain-of-thought paradigm, with each subquestion associated with specific receptive or cognitive functions such as high-level visual reception and inference. Using a probing-based methodology, we identify attention heads that specialize in these functions and characterize them as functional heads. Our analysis across diverse VLM families reveals that these functional heads are universally sparse, vary in number and distribution across functions, and mediate interactions and hierarchical organization. Furthermore, intervention experiments demonstrate their critical role in multimodal reasoning: removing functional heads leads to performance degradation, while emphasizing them enhances accuracy. These findings provide new insights into the cognitive organization of VLMs and suggest promising directions for designing models with more human-aligned perceptual and reasoning abilities.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper investigates the role of attention heads for multimodal reasoning. Using a new dataset with multiple levels of questions, they try to map attention heads to human cognitive functions.

### Strengths
**Originality** The underlying question is interesting.

**Quality** The data set seems well motivated and the presentation is clear for the most part.

**Clarity** The goal of the paper is clear. 

**Significance** The question of whether specific attention heads for subprocesses of multimodal reasoning exist in VLMs has not been explored as far as I know.

### Weaknesses
While the overall investigation seems well motivated, I think some of the take-aways are overstated. The writing is a bit sloppy at parts and should be improved. Some results are not explained very well. I appreciate taking inspiration from human cognitive processing, and I do think that there is more to be understood in regard to how well LLM and human cognitive processing relate to each other, but this paper does not present reliable insights in its current state. To summarize, I like the general idea but feel the paper is not in a state to be accepted.

### Questions
**Main questions:**
- While I think taking inspiration from human cognition is always nice, I do not think the abilities tested in the data subsets map as neatly onto brain areas as they are presented here. In the example in Figure 1 for example, the counting of particles in the respective solutions is surely something that other parts of the brain, such as the parietal lobe and the frontal lobe are involved with also. I see this whole approach as more of a guided questioning, where questions build on their predecessor. 
- Line 153 for the data filtering and continued in the Appendix you write "A subquestion is deemed invalid if at least two annotators mark it as false. If over 40% of the subquestions in a QA pair are invalid under this rule, the whole QA pair is removed." What happens to subquestions that are deemed invalid? If two raters deem a single subquestion invalid, does it remain in the QA set anyways or is it removed while the other subquestions remain?
- Line 175 "To support coherent multi-step reasoning, we include preceding subquestions and their answers as contextual input" does this mean that even if a model does not correctly answer previous subquestions, it is queried on later subquestions given the correct answer as contextual input? I feel like this is kind of counterintuitive given your motivation of sequential processing. In this case, the model does not really perform sequential processing in a way that would be comparable to humans. Instead, it is given a very informative context with parts of the solution.
- Line 248 "These results demonstrate that VLMs rely on highly specialized, localized components for distinct cognitive abilities." I'm not sure this is a fair assessment, looking at Figure 2, there seems to be quite a lot of overlap in head activations between different abilities. In general, I think it'd be nice to compute a correlation between the activations compared to "accuracies over the eight functions". Also for this claim on the next line "Moreover, this sparse functional organization is consistent across architectures and scales: heatmaps for five additional model", I would want to see a correlation or consistency metric of sorts, rather than visual comparisons between the heatmaps. I think you go into this somewhat in line 369 with "8% of cognitive heads across eight functions participate in more than one function." but I don't know what exactly this means. I'd want you to clarify what participating in more than one function means here. 
- For Figure 3 what is the difference between LLM and ACC lines? In line 307 it says "To quantify the impact, we employ both an LLM-based judge (Qwen3-30B LLM (Yang et al., 2025)) and an integrated accuracy metric". I understand the accuracy metric but what does the LLM based judge mean? Is the output of the masked model judged by this other LLM and then the fraction of answers judged as correct is shown here? I feel like this could use more detailed explanation. I'm similarly confused about Table 2, the numbers next to the arrows seem to indicate that performance before was at 100 and turning off specific heads reduced performance by this amount. Does this mean that for all analyses, you only analysed questions that the respective model got right before? 
- Line 308, again explaining FIgure 3 "An output is considered unaffected if its BLEU score (Papineni et al., 2002) exceeds 0.8, or if either the ROUGE score (Chin-Yew, 2004) or the semantic similarity score surpasses 0.6." Which score then is reported in Figure 3? The y index only reads "score".
- Just so I understand correctly, in the masking analysis in section 4.2, you mask out the top % of heads that have the highest accuracies for a given subtask, not over all tasks, right? Am I right in thinking that Figure 4 shows the performance differences in relation to the unmasked model performance (basically, how much worse do the model outputs become for all of the different functions if you ablate the highest % for one specific function)?
- Line 367 "The neural system is inherently complex, with individual neurons often participating in multiple functions (Mante et al., 2013). We observe a similar phenomenon in VLMs" to me this feels like you are interpreting all results to mean the data fit humans. In line 248 you write "These results demonstrate that VLMs rely on highly specialized, localized components for distinct cognitive abilities" don't these two assessments clash? 


**Minor comments**:
- Line 37 beginning should probably read "for a human" or "for humans"?
- LIne 51 you write "attention heads, an important component in VLMs". I think, since this paper is geared towards a machine learning crowd as well as the neuroscience crowd, it would be good to give some more explanation on what attention heads actually are.
- Line 53 "a dataset that bridges" bridges is missing the s, also Line 198 "triplets" are missing the s
- Line 106 you write "To systematically capture the cognitive processes involved in complex reasoning tasks, we consider eight functions related to complex multimodal reasoning, inspired by established frameworks in cognitive science (Anderson, 2014; Diamond, 2013)." Maybe you can give some more detail on how exactly the frameworks map to the eight functions you have chosen? This sentence reads a bit like these functions are generally taken to be fundamental in these cognitive frameworks, but as far as I remember ACT-R for example does not focus specifically on these.
- Line 200 "the attention-head output values that lead to true answers recognized as positive class while lead to other functions as negative class." is hard to understand and not grammatically correct
- Line 213 "(InternVL3-8B and InternVL3-8B)" should read 2B and 8B
- Line 284 "Intervention results (%) of cognitive heads" percent of correct answers?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors built a dataset and a probing+intervention framework to show that specific attention heads in VLMs act like “functional heads” (e.g., math, visual reception, decision), are sparse and organized, and causally matter for multimodal reasoning. The authors probe VLM attention heads by extracting per-head activations from token-selected answer traces and training simple probes to score each head’s association with a function. The authors then run interventions: (i) negative—masking/attenuating selected heads—and (ii) positive—nudging heads along learned “functional directions.” Experiments span three VLM families. The work suggests VLMs contain function-specialized, causally relevant mechanisms that can be studied and steered, opening doors to interpretability and targeted test-time control.

### Strengths
1) Head-level masking and directional “positive interventions” move beyond correlational probing and show functional necessity/sufficiency signals.
2) Results replicated on InternVL, Qwen-VL, and Gemma at multiple sizes which mean claims aren’t model-specific. 
3) Shows where (layers) and on what (image vs text tokens) different functions concentrate, plus cross-modal bridge heads.
4) Interventions on function-specific heads affect OK-VQA/Clevr-Math, suggesting external validity beyond the in-domain dataset.

One thing I love the most is its robust use of control experiments to demonstrate that the identified cognitive heads are function-specific and not random artifacts. The authors repeatedly compare the impact of masking their identified functional heads versus masking an equal number of randomly selected heads. In Figure 3, for instance, they show that masking up to 10% of random heads in Qwen2.5-VL-3B has negligible effect on performance, whereas masking the same fraction of cognitive heads leads to a steep drop in accuracy—underscoring their causal importance. Table 1 reinforces this across six models and eight functions, where performance on the corresponding cognitive skill sometimes drops to near-zero after masking, while random masking leads to minor degradation. They further strengthen this claim via cross-function controls in Figure 4, where masking heads from unrelated functions causes far less harm than masking those directly associated with the function under evaluation. They need to do statistical tests like human psychology control experiments however.

### Weaknesses
1. The paper does not appear to report standard statistical significance testing (e.g., t-tests, ANOVA, or confidence intervals) for its main results. The paper relies primarily on descriptive accuracy comparisons—such as consistent drops/improvements in Figure 3, Table 1, and Table 4—rather than formal statistical analysis. It supports claims through visual inspection (e.g., attention heatmaps in Figures 2 and 5–9) and contrasts between cognitive vs. random head masking.
2. The authors claim to investigate "cognitive" side of MLLM but have insufficient engagement of cognitive vision in MLLM literature such as,

Schulze Buschoff, L. M., Akata, E., Bethge, M., & Schulz, E. (2025). Visual cognition in multimodal large language models. Nature Machine Intelligence, 7(1), 96-106

Li, Y., Gao, Q., Zhao, T., Wang, B., Sun, H., Lyu, H., ... & Deng, H. Core Knowledge Deficits in Multi-Modal Language Models. In Forty-second International Conference on Machine Learning.

minor 
1. Uniform scaling (ε) can cause off-target side effects; no comparison to alternative causal methods (path patching, causal tracing, activation patching).
2. Linear probes on selected tokens may conflate feature presence with linear separability; top-k token selection itself is LLM-driven.
3. Focus is only on attention heads; MLPs, attention patterns (Q/K), and routing components aren’t analyzed; limited stats on variance/seed stability.

### Questions
1) How consistent are subquestion function labels across annotators, and how often did GPT-o3 disagree?
2) How sensitive are head importance maps to (a) top-k token choice, (b) prompt format/order, and (c) random seeds across probes/interventions?
3) Do you think analogous “functional directions” exist in MLP blocks or cross-attention vs self-attention splits, and does combining head+MLP interventions yield larger, more stable gains?

### Soundness
3

### Presentation
2

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
This paper aims to explore the internal working mechanisms of Vision-Language Models (VLMs), specifically analyzing whether their Attention Heads play specific roles in multimodal reasoning that are analogous to human cognitive functions.  To achieve this goal, the authors make two main contributions: Construction of the CogVision Dataset: a new interpretability dataset. Proposal of an Analytical Framework: The authors use a probing-based method to identify 'functional heads'—attention heads that are highly correlated with specific cognitive functions.

### Strengths
1. The CogVision provides a structured method for analyzing multi-step, multimodal reasoning processes, which surpasses many standard VQA benchmarks. Mapping reasoning steps to specific cognitive functions provides a tool for fine-grained model analysis.
2. The study spans three different VLM families and multiple model scales, making the conclusions about the "universality" and "intrinsic" nature of functional heads more convincing.

### Weaknesses
1. **Overextension of the Core Claim ("Reasoning Like Humans")**: The paper's findings merely demonstrate functional specialization within the VLM, which is not equivalent to reasoning in a human-like manner. Any complex system designed to solve multifaceted tasks is likely to evolve modules for handling specific sub-tasks (like visual processing or mathematical computation). The title poses the question, "Do VLMs reason like humans?" but the research findings (the discovery of sparse functional heads) are far from sufficient to answer this. What the authors have found is "functional specialization," not a "human-like reasoning process."

2. **Flaws in the CogVision Dataset:** Data Source is "LLM Cognition," Not "Human Cognition": The dataset's construction relies heavily on GPT-4.1 to generate sub-questions and chains of thought. This means the "cognitive steps" being analyzed are, in fact, the thought processes of another large language model, not those of humans.

3. **Methodological Ambiguities:** The 8 "cognitive functions" defined in the paper are extremely broad (e.g., "Inference" or "Decision-Making"). Can one attention head truly be responsible for all "Inference"? This classification is simplistic, and the probing task may only be capturing superficial statistical correlations related to these coarse labels. 

 In Section 3.1, the authors use another LLM (Qwen3-30B) to "select the top-k most important tokens" for extracting head activations. This selection itself is a black-box process, introducing an unnecessary and unanalyzed confounding variable into the experiment. Why is this the best method? How sensitive are the results to the choice of 'k' or the specific LLM used? This makes the source of the probe's input features questionable.

### Questions
See above

### Soundness
2

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
3

### Summary
The paper asks whether current VLMs exhibit human-like, functionally organized reasoning and answers this by building an interpretability pipeline centered on attention heads. It introduces CogVision, a multimodal QA dataset that decomposes each question into CoT-style subquestions, each labeled with one of eight perceptual/cognitive functions (e.g. low-/high-level visual reception, language knowledge recall, math reasoning, decision making), creating supervision aligned with a cognitive hierarchy. Using a probing-based method on several VLM families (Intern, Qwen, Gemma), the authors identify “functional heads” whose activations strongly predict these functions and show these heads are sparse, universal across architectures, and layer-structured. Causal interventions—masking vs. amplifying specific heads—demonstrate that these heads are not just correlated but necessary for the corresponding multimodal reasoning steps, and manipulating them transfers to downstream VQA-style tasks. Overall, the work argues that current VLMs contain an emergent, human-analogous hierarchy of attention heads, offering a handle for designing more interpretable and human-aligned models.

### Strengths
1. The paper performs a series of well-controlled manipulations (like versus random), augmented (layer-level) probes, and tests on modality sanity to validate the role of attention heads.
2. Introduces CogVision, a multimodal QA dataset that decomposes each question into chain-of-thought (CoT) subquestions richly human-labeled with eight perceptual and cognitive functions—ranging from low-level and high-level visual reception to inference, math reasoning, and decision-making
3. clear organization, formation, and visualizations

### Weaknesses
1. Because each “cognitive function” is inferred from GPT-4.1 decompositions of existing benchmarks, the resulting subQAF groups may differ systematically in surface form—such as question phrasing, token length, or modality density—rather than in underlying cognitive process. More analysis of in-group question diversity or ulternation of piepline can clarify if it's confounded by dataset-level artifacts
2. Section 4.3 briefly notes that (for example) 18 % of heads participate in multiple functions and that early-stage functions support later reasoning, yet these statements remain vague and builds on the assumptions that heads can be first identified independently instead of potential activation or not as a result of an interplay of multiple cognitive abilities as the authors also acknowledge the inherent complexity and interdependency.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

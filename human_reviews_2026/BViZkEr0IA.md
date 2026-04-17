# The Mechanistic Emergence of Symbol Grounding in Language Models

- Decision: Reject
- Scores: 2, 6, 6, 4, 6

## Abstract
Symbol grounding (Harnad, 1990) describes how symbols such as words acquire their meanings by connecting to real-world sensorimotor experiences.
Recent work has shown preliminary evidence that grounding may emerge in (vision-)language models trained at scale without using explicit grounding objectives. 
Yet, the specific loci of this emergence and the mechanisms that drive it remain largely unexplored. 
To address this problem, we introduce a controlled evaluation framework that systematically traces how symbol grounding arises within the internal computations through mechanistic and causal analysis. 
Our findings show that grounding concentrates in middle-layer computations and is implemented through the aggregate mechanism, where attention heads aggregate the environmental ground to support the prediction of linguistic forms. 
This phenomenon replicates in multimodal dialogue and across architectures (Transformers and state-space models), but not in unidirectional LSTMs. 
Our results provide behavioral and mechanistic evidence that symbol grounding can emerge in language models, with practical implications for predicting and potentially controlling the reliability of generation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper designs tasks where certain context token (environmental token, can be vision token) can be good signal for predicting another token (linguistics token). When trained on these tasks, models can find their connection and make use of it. The authors put this process in the context of symbol grounding in VLMs (how linguistic tokens is grounded to vision patterns). They aim to understand grounding in VLM by studying these simplified and controlled settings. Authors also show clear evidence supporting that such connection differs from normal co-occurrence. They also analyze and show there is a gather and aggregate mechanism in the model that achieve this.

### Strengths
- The study uses R-squared statistic to show that the grounding can not be explained by simple co-occurrence.
- The study includes 3 different tasks involving symbol grounding and various model architectures (transformers, SSMs, LSTMs of different layers)

### Weaknesses
- The dataset is small, containing only 100 nouns, and templates for constructing text containing \<LAN\> are simple (often without much extra text as noise). This makes the grounding signal potentially much stronger than the case in real world pretraining. Such difference might affect the phenomenon shown in the paper, and can be the key to understand grounding (i.e. how strong does it need to be to make grounding happen)

- The paper uses too much space to show such grounding can emerge in the proposed tasks. But content for investigating how it happens thoughout training (I mean more detailed studies besides just information gain throughout training) and what is the mechanism is limited, while the authors say that the goal is to understand the emergence of grounding.

- The mechanistic interpretation is limited and evidence is vague. The results show some important layers and heads. But the usage of Saliency flow is only an estimation of coarse-grained importance. There’s also categorization of heads according their attention patterns. But they only show higher-than average causal effect. We do see some patterns in Table 2, which shows heads that tend to attend \<ENV\> is more important than average attention head. But this information is limited, it does not mean they or the hypothesized mechanism fully explains the grounding (in that case other heads should have almost zero effects).

### Questions
- The paper argues that the main mechanism is gather-and-aggregate, so both the gather heads and aggregate heads should play important roles. But why in table 2, intervening on gather heads does not affect the surprisal? Because they are needed by aggregate heads, zeroing out them should also destroy the circuit.

- The naive co-occurrence cannot explain information gain. Do you think it might be the case that the model just figure out when to use co-occurrence statistics? or which tokens can be predicted from occurrence of some other tokens.  So it just uses it for some tokens, like nouns.

### Soundness
2

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
4

### Summary
This paper claims to causally and mechanistically trace the emergence of symbol grounding in language models. The authors design a two-token system, where each lexical item appears as a linguistic token (\<LAN\>) and an “environmental” token (\<ENV\>), then measure reductions in surprisal when the linked token is present. They further identify “aggregate heads” responsible for propagating information from \<ENV\> to \<LAN\> tokens.

### Strengths
Well-structured evaluations and mechanistic interpretability experiments.

Analysis across multiple architectures and modalities is thorough.

Causal ablations offer meaningful insights into attention head specialization.

### Weaknesses
The proposed evaluation does not measure symbol grounding:

The setup relies purely on two linguistically implemented tokens that correspond to the same object. The “environmental tokens” are still just symbols with no perceptual grounding.

Thus, the model’s task reduces to:

learning a mapping between symbol A (\<ENV\>) and symbol B (\<LAN\>)

rather than mapping a symbol to real-world referents.
This fails to demonstrate grounding as defined in Harnad (1990), which requires sensorimotor anchoring:

| Aspect | True Symbol Grounding | Paper’s Setup |
|--------|----------------------|---------------|
| Source of Meaning | Perception / world interaction | Another text token |
| Representation Modality | Non-symbolic | Symbolic |
| Link Type | Causal grounding | Token association |


While the reported effects are intriguing, it remains unclear whether the current setup demonstrates symbol grounding per se, as opposed to learning an association between two linked tokens. In particular, the “image-grounded” condition relies on pretrained ViT features that may already encode semantic regularities, which complicates claims of grounding from raw perceptual input. We encourage the authors to provide a more detailed argument for why the two-token design suffices to establish symbol grounding rather than symbol–symbol alignment, and to clarify the causal pathway from environmental evidence to linguistic meaning.

### Questions
1. Can the model still “ground” if you remove any shared lexical identity between \<ENV\> and \<LAN\> tokens (e.g., no 1–1 mapping)?

2. Does grounding transfer to unseen objects or contexts? Specifically, information between token and intrinsicly visualized object?

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
This paper investigates the emergence of symbol grounding in large language models (LLMs) and vision–language models (VLMs)—the process by which models learn to bind sensory experience to linguistic meaning. Specifically, the authors examine how sensory information in one modality (e.g., visual or environmental) is encoded and represented in the tokenized form of another modality (e.g., textual or linguistic), and how attention heads learn to associate corresponding tokens across modalities. The first part of the paper presents a set of behavioral analyses designed to measure the strength of symbolic grounding. The authors quantify grounding by testing whether a model trained from scratch can predict the appropriate linguistic token given environmental tokens. They further assess how prediction accuracy (or equivalently, surprisal) changes when the matching environmental token is present versus when alternative cues are provided. 
The authors also considered potential confounds from simple co-occurrence statistics and conducted extra experiments to show that symbol grounding cannot be fully explained by simple co-occurrence statistics.
The second part of the paper investigates the mechanistic emergence of symbol grounding within attention heads of VLMs and LLMs during training. The authors find that grounding representations tend to concentrate in the middle layers, implemented through an aggregate attention mechanism.
By perturbing specific attention heads, they demonstrate that symbol grounding is causally mediated by these components—perturbations lead to a measurable drop in their proposed grounding metric.

### Strengths
* The study is systematic, covering both behavioral and mechanistic analyses, tested on a wide range of LMs.
* The writing is clear and logically structured, making the argument easy to follow.
* The combination of causal interventions with behavioral probing provides convincing evidence for grounding-related mechanisms.

### Weaknesses
* Currently, the discussion of how symbol grounding relates to model failure modes (e.g., hallucination) is limited.
* It might also help to connect this work more explicitly to theoretical accounts of symbol grounding in cognitive science and linguistics.

### Questions
* It is unclear how attention heads act as a mechanistic pathway for symbol grounding. What is the precise mechanism introduced by these heads? How do earlier-layer heads influence the residual streams that, in turn, affect later layers? Why does grounding typically emerge in intermediate rather than early or late layers?
* Choice of mismatch tokens:  More discussion is needed on how mismatch tokens are selected, especially within the same semantic category, as this could strongly affect the interpretability of the surprisal effects.
* Surprisal distribution:  I am intrigued that the VLMs exhibit non-deterministic surprisal. It would be informative to visualize the distribution of surprisal values across the token dictionary and to analyze how they relate to different semantic categories or levels of abstraction.
* Abstract category representation:  How does grounding behave for hierarchical semantic categories (e.g. elephant -> mammal -> animal)? For example, if the visual scene depicts an elephant but the text says “the animal,” does the attention head for the abstract category (“animal”) emerge earlier or later than the one for the specific concept (“elephant”)? 
* Behavioral observation:  it is puzzling that symbol grounding does not emerge in LSTMs. I am curious to hear the author's explanation on this observation.
* What is the relation of this work to the work by Pavlick discussing symbolic grounding in LLMs?

Reference:
_Pavlick E. Symbols and grounding in large language models. Philos Trans A Math Phys Eng Sci. 2023_

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
The paper is focused on how various neural architectures solve the symbol grounding problem, that is, the problem raised by Hanard 1990 which asks how to symbols in an purely formal symbolic system come to refer to their real-world referents. The authors train several models from scratch on a variety of corpora (including CHILDES, and a visual dialog corpus). The authors manipulate the context provided to the model (either relevant or irrelevant to the utterance to be predicted) and show that during training, Transformers and SSMs see a sudden drop in surprisal in the relevant setting, showing that they develop a mechanism for using the context to reduce prediction uncertainty whereas LSTMs do not. The authors further perform some mechanistic analyses to demonstrate which attention heads are highly involved with this ability.

Overall, I lean against accepting this paper, with the primary reason being the framing. I just don't think the results here really have much to do with symbol grounding (I elaborate in Weaknesses). That said, I do think there are some interesting results--ones I might even share with colleagues--but I think they have more to do with architectures and their ability to handle long-context and binding operations. There is a lot to like about the study (the careful experimental design, the fact that models are trained from scratch, the variety of complementary analyses). But as the paper pitches its primary contribution and novelty as being about grounding per se, I think we are obligated to make sure that this pitch is framed appropriately and accurately. These days, its quite plausible that many readers will learn about the problem the first time in reading this work, and we have an obligation to make sure that we frame and interpret results accordingly.

### Strengths
* I appreciate that the authors train models from scratch in a careful way. This permits more conclusive takeaways from the results.
* The variety of datasets chosen is interesting, and adds interesting dimensions beyond standard VQA style experiments
* It is nice to see a combination of types of analyses - behavioral, traning dynamics, mechanistic studies. These combine to give a fuller picture of what we are looking at and form the ingredients of an overall nice paper.
* The comparison across different architectures is interesting. Its a nice clear illustration of the importance of an architectural binding mechanism. I know many people (esp. in cog and neuro science) who would like to see this result.

### Weaknesses
I believe the biggest issue with the paper is the framing. I think the results here are not "about" what the authors say the paper is about--i.e., I don't really see these results as being "about" grounding. I view framing in science as all-important, its not mere marketing, it actually dictates what the contribution of the paper is, and what future work it can/should influence. So at a reputable conference like ICLR, there is an obligation to quality control the framing at least as much as the experiments and results, if not more. That is why I recommend rejecting, but with the hope that the authors resubmit with new framing but largely the same results.

To elaborate: The authors study the "grounding" problem by analyzing how well LMs incorporate context to predict the next word in a sequence. The data is always represented as sequences of tokens (the exact source of these tokens varies, but they are always tokens of some kind). The mechanism the authors describe is one in which LMs attend backward to tokens in context and use that to increase/decrease the probability of the next word. 

My concern is that this mechanism described is extremely generic, and in fact describes how LMs do most of what they do. The fact that the tokens represent environment vs. language is irrelevant here, and I'd expect the exact same mechanism to apply in other contexts in which the model must retrieve from context....e.g., in a long text passage in which the model needs to produce any co-referential to syntactically related phrases which require looking back in the context. (Actually, I'd recommend that the authors run some controls of this type to convince readers that the mechanism is somehow specific to 'grounding' and not general to 'binding'.)

The above matters even more than it would otherwise because the authors hang heavily on the fact that their work is more faithful to Hanard's definition of symbol grounding. But Hanard's whole point was about how, once you have a system that represents the world as symbols, how do you know you aren't just studying relationships between meaningless symbols. I think his criticism could apply exactly to the work described here (or to most of modern LLMs, its not just this paper). I am pasting the abstract of Hanard 1990 for reference, I think it is pretty apparent why modeling the mechanisms for relating one symbol to another in an LM does not get at what he is going for. In fact, the related work that the authors point to as being merely about correlation/geometry is arguably closer to what Hanard was asking for, rather than something about the (quasi-symbolic) operations carried out by attention heads.

Hanard: "There has been much discussion recently about the scope and limits of purely symbolic models of the mind and about the proper role of connectionism in cognitive modeling. This paper describes the "symbol grounding problem": How can the semantic interpretation of a formal symbol system be made intrinsic to the system, rather than just parasitic on the meanings in our heads? How can the meanings of the meaningless symbol tokens, manipulated solely on the basis of their (arbitrary) shapes, be grounded in anything but other meaningless symbols? The problem is analogous to trying to learn Chinese from a Chinese/Chinese dictionary alone. A candidate solution is sketched: Symbolic representations must be grounded bottom-up in nonsymbolic representations of two kinds: (1) "iconic representations" , which are analogs of the proximal sensory projections of distal objects and events, and (2) "categorical representations" , which are learned and innate feature-detectors that pick out the invariant features of object and event categories from their sensory projections. Elementary symbols are the names of these object and event categories, assigned on the basis of their (nonsymbolic) categorical representations. Higher-order (3) "symbolic representations" , grounded in these elementary symbols, consist of symbol strings describing category membership relations (e.g., "An X is a Y that is Z"). Connectionism is one natural candidate for the mechanism that learns the invariant features underlying categorical representations, thereby connecting names to the proximal projections of the distal objects they stand for. In this way connectionism can be seen as a complementary component in a hybrid nonsymbolic/symbolic model of the mind, rather than a rival to purely symbolic modeling. Such a hybrid model would not have an autonomous symbolic "module," however; the symbolic functions would emerge as an intrinsically "dedicated" symbol system as a consequence of the bottom-up grounding of categories' names in their sensory representations. Symbol manipulation would be governed not just by the arbitrary shapes of the symbol tokens, but by the nonarbitrary shapes of the icons and category invariants in which they are grounded.

tldr: I think the authors have something interesting to say about a generic mechanism via which LMs manage and retrieve relevant information from context to support next word prediction, and how certain architectures (those with a mechanism for content-adressable retrieval) are more capable of this than others. That is a very interesting finding that many would like to read about. But when framed as a paper about grounding, I think it it is misleading and makes higher level claims that are not actually supported by the studies.

### Questions
Can you explain why the gather head doesn’t really matter. What is happening instead then? I.e., what is having the actual effect of gathering, in your proposed mechanism, if the gatherer heads aren't causally important?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the mechanistic basis of symbol grounding in multi-modal language models. The authors introduce a controlled evaluation framework designed to trace how abstract linguistic symbols acquire meaning by being connected to environmental content. The authors construct examples with two distinct representations for each lexical item: "environmental" tokens (ENV) representing its non-verbal (e.g., visual or situational) presence, and "linguistic" tokens (LAN) representing its use in a spoken utterance. The authors measure grounding using "grounding information gain," a surprisal-based metric that quantifies the model's ability to predict LAN tokens when its corresponding ENV tokens are present in the context, compared to when a mismatched ENV tokens are present. The authors show that grounding, as measured by information gain, emerges during training in Transformer and state-space models (Mamba-2) but fails to emerge in unidirectional LSTMs. Using saliency analysis and a "TunedLens" analysis, the paper traces the computations critical for grounding to the middle layers of the network, and then show that these peaks correspond to the location in the models that gather-and-aggregate mechanisms emerge, implicating these mechanisms in symbol grounding. Then using a series of head-level ablations, they reveal that aggregate heads are involved in the grounding effect, while gather heads play a less significant role.

### Strengths
1. *Methodology:* The paper uses a minimal, controlled testbed that allows them to distinguish between the "symbol" from its "ground" and allows for more precise, causal analysis.
2. *Confound control:* The authors demonstrate that the emergent grounding is not simply a signature of the models learning shallow co-occurrence statistics. In the analysis presented in Section 4.2, they show that grounding information gain continues to rise as the correlation with co-occurrence statistics ($R^2$) falls.
3. *Strong Mechanistic Account:* The authors successfully build a clear argument for the mechanistic circuit underlying symbol grounding. They first localizes the effect to the middle layers using saliency and probing, then hypothesizes a specific circuit (gather-and-aggregate), and finally confirms the hypothesis with causal interventions.. 
4. *Generalizability:* The authors demonstrate that their findings are not brittle, and that the grounding effects replicated across different model architectures (Transformers, Mamba-2), model scales (4, 12, 18-layer Transformers), and across different grounding modalities, including text-based dialogue and image-grounded dialogue.

### Weaknesses
1. *Hallucination prevention*: Though the authors do mention that such analyses are more difficult to conduct in larger scale VLMs and mention the relevance of this work to hallucination mitigation in VLMs in their discussion, the strength of their contribution would be greatly strengthened if they could demonstrate how interventions on the circuits described here may help to mitigate hallucinations in more naturalistic use cases (e.g., with larger, open-source VLMs).
2. *Gaps in Related Work:* One minor weakness is the papers related work, which misses a highly relevant line of inquiry. The "gather-and-aggregate" mechanism is analogous to mechanisms identified in recent work on symbol processing, variable binding, and abstract reasoning in LLMs. These works identify circuit components analogous to the "gather" and "aggregate" operators discussed in this paper, and situating the work within this literature may help increase the broad appeal of the paper (see Questions).

### Questions
1. *Connecting to Symbol Binding Literature:* The related work on the gather-and-aggregate mechanism and symbol grounding overlooks recent work on symbol processing and variable binding in Transformers. Could the authors comment on the relationship between their findings and the mechanisms identified in the following (or similar) papers? These seem highly relevant:
	* `https://arxiv.org/abs/2310.17191`
	* `https://arxiv.org/abs/2406.19501`
	* `https://arxiv.org/pdf/2502.20332`
	* `https://arxiv.org/abs/2409.05448`
2. *Asymmetry in Multimodal Settings*: The paper finds a strong asymmetry in the text-only (CHILDES) domain, where aggregate heads are causally critical but gather heads are not. The authors hypothesize this is because the "input template is semantically light". This makes sense. However, this asymmetry seems less likely to hold in the image-grounded setting in which the "ground" is distributed across many visual tokens, and "gathering" information from relevant patches would seem to be a crucial prerequisite for aggregating it. Do the authors have any preliminary results or speculation on whether this aggregate > gather effect decreases or reverses in the image-grounded dialogue setting?

### Soundness
3

### Presentation
4

### Contribution
3

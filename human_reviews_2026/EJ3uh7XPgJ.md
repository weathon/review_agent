# On Non-interactive Evaluation of Animal Communication Translators

- Decision: Reject
- Scores: 8, 4, 4, 2

## Abstract
If you had an AI Whale-to-English translator, how could you validate whether or not it is working?
Does one need to interact with the animals or rely on grounded observations such as temperature? We provide theoretical and proof-of-concept experimental evidence suggesting that interaction and even observations may not be necessary for sufficiently complex languages. One may be able to evaluate translators solely by their English outputs, offering potential advantages in terms of safety, ethics, and cost. This is an instance of machine translation quality evaluation (MTQE) without any reference translations available. A key challenge is identifying "hallucinations," false translations which may appear fluent and plausible. We propose using segment-by-segment translation together with the classic NLP shuffle test to evaluate translators. The idea is to translate animal communication, turn by turn, and evaluate how often the resulting translations make more sense in order than permuted. Proof-of-concept experiments on data-scarce human languages and constructed languages demonstrate the potential utility of this evaluation methodology. These human-language experiments serve solely to validate our reference-free metric under data scarcity.
It is found to correlate highly with a standard evaluation based on reference translations, which are available in our experiments. We also perform a theoretical analysis suggesting that interaction may not be necessary nor efficient in the early stages of learning to translate.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes and evaluates a method on how to rate the translation quality of translations which don’t contain any reference.
The proposed method is to split the given source into segments (using a LLM) and translate each segment independently (using a LLM). Afterwards a LLM is used to judge if the ordering of the target sequences makes sense.

It’s a inneressing approach to solve an unusual problem. The main weakness seems to be that it’s overly relying on wikipedia and models from the same LLM company.

### Strengths
The method allows to give some insights of whether the translation makes any sense or not when no reference is available.

For animal communication it allows one to get some impression if the method works or not without requiring additional communication, observing the animals is enough.

They confirmed that the approach works on low resource languages and on constructed languages.

### Weaknesses
I think the results shown in Figure 5 are most likely exaggerated. While it’s very plausible that later models simply perform better because they are strong, it is also very likely that later models saw the given wikipedia articles in its latest form during training. These models would know the correct ordering which in turn would inflate their results.

Since the parallel text were extracted from wikipedia articles in different languages there is a high likelihood that the first paragraphs are not really translations of each other. The authors acknowledge this, but not having proper translations still adds a lot of noise into the evaluation.

We can assume that all LLMs saw the latest wikipedia version available at training time. It’s not unlikely that they learned the correct paragraph ordering across languages. Also since the approach to treat the first paragraphs as parallel data seems to work, we can assume that the ordering is similar across languages meaning the LLMs saw the right ordering during training.

Every step in this work involves LLMs from OpenAI. This increases the likelihood that the LLM actually knows the correct sentence since it was created by a similarly trained LLM, again inflating the results. It would be good to see the results of at least one other LLM as well. I don’t expect a different result (thanks to wikipedia), but it would be more valuable information than having everything from related LLMs.

The constructed languages were created using LLMs. Given that LLMs are trained on human languages, I’m not really convinced that these languages are really so different to human languages as claimed. For the given purpose I think the described method should work well enough.

### Questions
End of Line 255: I think you forgot the “=0” and wanted to write “p(T, T’) = 0 indicates T is more plausible than T’”.

Was the cut off date June 1st, 2024 only checked on the non English side of the wikipedia article, or also on the English side?

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
3

### Summary
This submission derives some theoretical results on learning based on interactions vs. (less invasive and easier) observations. This is provided in the context of reference-free evaluation of low-resource or no-resource translation, with the particular use case of the translation of animal communication. Experiments show that the proposed shuffling-based metric is potentially useful to evaluate translation, without requiring costly or impossible to acquire references.

### Strengths
The core problem of evaluating "translations" in the context where references are not available is highly relevant. The application of evaluating animal communication is interesting and exciting. The proposed technique, relying on comparing the coherence of a translation with a shuffled version of itself, is novel (in this context). Experiments on both low-resource languages and constructed languages show that the proposed shuffle test yields significant correlation with a reference-based evaluation, which is promising.

### Weaknesses
A large part of the paper is occupied by a theoretical analysis. The theory it presents derives from well-established learning theory results. More importantly, the setup of interactive vs. observational learning is very loosely connected to the actual, highly interesting application to animal communication. In fact, there is essentially no reference to the results in Section 2 in the rest of the paper. A much tighter and clearer connection between that theory and the shuffle test would be highly appreciated. Otherwise, more methodological and experimental details on ShuffleEval and its evaluation (now exiled in the appendix) would be welcome.

Experiments are limited to about 100 test examples in each language/conlang. This may reflect operational constraints of animal communications studies, but is fairly low by machine translation standards. In that context, error bars or uncertainty evaluation would greatly help qualify how variable actual results and assess confidence.

Although experiments involve many "translators", they are all flavours of OpenAI's GTP. In addition, all evaluation is done using GPT. MT metrics (esp. reference-free) is a lively field of research, it is surprising that none of these metrics was included as reference. Minimally, the use of GPT5/4 are references for coherence & MT quality could be manually validated on a sample of examples.

### Questions
Recent work on MT metrics suggest that novel LLM-based metrics have strength for high-performing (high-resource) languages, but struggle to estimate mid-to-low performance systems (such as typically the case for low-resource languages, and one would assume conlangs).

(l.177) Why is the empirical risk minimization infeasible? Do you mean because of practical (e.g. multiple minima) or theoretical reasons?

(l.201) How is the translator family growing with the number of interactive experiments? Would it not be fixed for a choice of parameterization?

(l.230-238) Paragraph is not super convincing as the argument relies on ad hoc parameter choices (c, eps).

(l.249) Why would translators be different for paragraph-level (f) and segment-level (\phi)? Esp. with LLMs one would expect that they are the same.

(l.366) Why is the date of June 1, 2024 chosen?

(l.343) "prior work has validated": a reference would be nice here. Presumably you mean the refs. in l.052.

(l.357) How is 99.9% estimated from 100 examples?

(l.481-483) Could you expand?

Typos:

l.091: Set 'F' is only introduced later (l.144)

l.140: the the

l.222: Is opt_n the same as {f}^*_n?

l.255: missing value before "indicates T"?

l.343: we highly -> were highly

l.352: missing 'of'?

l.354: while model the -> while the?

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
This paper proposes an unsupervised evaluation for machine translation in a relatively high error stage. The motivation is to extend MT and its evaluation to animal language, where `interactive testing` as they call is, is more expensive or sometimes even infeasible with current resources.
The core contribution is a formal motivation and simulated experiments where ShufflEval is deployed as unsupervised MT metric.

### Strengths
- Unusual topic, adds diversity.
- Formal derivations and proof of the approach.

### Weaknesses
- Title is misleading. The motivation stems from animal sound translation, but the paper does not actually perform any experiments with whales.
- Artificial language setups might not be appropriately mimicking working with animals.
- Novelty: ShufflEval is not new and this evaluation doesn't add much to the adoption or success of it.
- Interesting discussion on the trade-off between cost and interactivity of feedback.

### Questions
- Why did you not use actual parallel, sentence-aligned data for the simulated? It would have removed some of the confounding factors/challenges.
- The "whalebreak" term is fun, but I'm not entirely sure if I understand it correctly? What does it entail?
- Can you motivate your data generation protocol? 
- What other simple MTQE could you apply?

### Soundness
3

### Presentation
3

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
The authors propose a translation quality metric that is purely candidate-based, meaning it has no access to the source or any reference translations. Their metric, which they call ShufflEval, compares the coherence of translating the source as-is versus cutting the source into smaller segments and translating the permuted segments. Naturally, one would expect the original source to have much higher coherence. 

The authors justify their proposed metric by appealing to basic statistical learning theory. They argue that although active learning/interactive systems can learn better models from less data than systems trained purely on observational data, this advantage essentially vanishes when collecting observational data is much cheaper than active learning, as one can just train systems on a lot more observational data. This is good news for the author's metric, as it relies solely on observational data.

Finally, the authors conduct experiments on low-resource human languages as well as so-called conlangs, which are artificial languages that they fabricated via prompting some powerful language model.

### Strengths
Overall, I found the original premise of the paper really intriguing and refreshing. I commend the authors on the first sentence of the abstract; it immediately drew me in and made me want to know everything about their work. I'm also a big fan of the terminology, e.g., the "whalebreak" model.

Though the applications of the authors' method seem a bit hypothetical or far-fetched at present, I found it relevant and interesting even as a purely thought-experiment.

### Weaknesses
I have two main issues with the paper: one regarding the theory/theoretical framing and one regarding the methodology.

First, the authors appeal to a fairly basic PAC bound (Eq 3) to argue that the expected risk of a system trained via empirical risk minimisation with respect to some loss $\ell$ will be close to the expected risk of a system trained via active learning. The result itself is fine, but using it to justify the use of the ShufflEval loss (defined in the equation in Section 2.4) is unsound for two reasons:
 1. It is easy to construct a system that minimises $\ell_{ShufflEval}$, which I call the "independent natural translator": it either ignores its input completely and outputs a piece of fluent English text, or (to make things more interesting) hashes the input and uses the hash as a seed to a random number generator to sample a piece of fluent English text from some dataset (e.g. English Wikipedia). As such, Eq 3 is vacuous for $\ell_{ShufflEval}$, and I cannot immediately see how the function class can be restricted to exclude these examples.
 2. This framing doesn't address the real problem: that we don't have reference translations. Given the story the authors tell in the rest of the paper, I would have expected Eq. 3 to connect the shufflEval loss to a loss that incorporates reference translations. As such, the authors should at least make it explicit that this is not what Eq. 3 represents.

As a more minor point, I would prefer the authors model translators as conditional distributions rather than functions, since in almost all cases sources have multiple valid translations. (Though this should not change the theory much)

My methodological issue has two parts also. First, I was disappointed that, despite the paper's incredible opening sentence and the careful ethics discussion in the introduction and at the end of the paper, there are no experiments on animal-to-English translation. Given this situation, I would either reduce the emphasis on animal translation in the main text (it occupies over one page!) or include some actual animal translation experiments. 

Second, regarding the conlangs examples, the authors state: "As a result, one might expect our conlangs to be less human-like, which serves the purpose of stress-testing ShuffleEval beyond human languages." However, I randomly spot-checked the translations generated by language models in the supplementary material and found that most seem to produce excellent translations. As such, this calls into question whether these experiments are meaningful in the first place. At any rate, the author's statement above certainly is not borne out by this observation.

If the authors can elucidate if and how my reasoning is incorrect and address my concerns, I'll be happy to raise my score. However, if they find my concerns valid, then I'm afraid that I cannot recommend acceptance without significant modifications to the paper. 

Miscellaneous:
 - page 5: $\rho(T, T')$ should be $\rho(T, T') = 0$
 - "We use LMs for several purposes, increasingly common practices in MT (Bavaresco et al., 2025), including" -- needs fixing

### Questions
How do the authors propose to pronounce their method?
Shuff - LEE - val, Shuffle - Eval or ShuffL - EE - val or some other way?

Does the authors' method have some connection with minimum Bayes risk decoding?

### Soundness
2

### Presentation
3

### Contribution
2

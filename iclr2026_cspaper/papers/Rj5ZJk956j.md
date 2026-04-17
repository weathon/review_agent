000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Weakening Neurons: A Newly Discovered Read-Write Functionality In Transformers With Outsize Influence

Anonymous authors Paper under double-blind review

## Abstract

We introduce a new mechanistic interpretability method for gated neurons, based on the cosine similarities between their weight vectors, and use it to gain a number of novel insights into the inner workings of transformer models. First, our method allows us to discover a class of neurons - *weakening* neurons - with surprising behavior: even though there are few, they activate extremely often and have a large influence on model behavior. Second, we show that nine different LLMs have similar patterns with respect to weakening neurons: weakening neurons appear mostly in late layers whereas their counterparts, *(conditional) strengthening* neurons, are very frequent in early-middle layers. Third, weakening neurons have a strong effect on model output when gate values are negative - which is surprising since negative gate values are not expected to encode functionality. Thus, for the first time, we observe a mechanism important for transformer functionality that involves negative gate values. 1

## 1 Introduction

Mechanistic interpretability research attempts to reverse-engineer the *mechanisms* inside neural networks, such as transformer-based (Vaswani et al., 2017) large language models (LLMs). Some of this work has addressed the interpretation of MLP sublayers, and we follow this line of research. Unlike previous work, we focus on gated activation functions (Shazeer, 2020), which are used in recent LLMs like OLMo, Llama and Gemma, but so far lack an extensive analysis from the interpretability perspective.

Much previous work analyzes neurons2 based only on the contexts in which they activate (Voita et al., 2024) or based only on their output weights3(Gurnee et al., 2024). However, neither of these fully captures the *mechanisms* that neurons implement: we also have to understand the relationship between input and output behavior of neurons, what we call their read-write (RW) functionality. We put this in the center of our analysis, and propose a simple method to investigate RW functionalities: computing cosine similarities between input (reading) and output (writing) weights. This new approach allows us to gain a number of striking novel insights into the inner workings of LLMs. In particular, we discover a small class of neurons - *weakening* neurons - with outsize influence and often surprising behavior. Our contributions are as follows: (i) We are the first to investigate read-write behavior of gated neurons, using cosine similarities of weight vectors. (ii) Applying this method to nine LLMs, we observe universal patterns: Early-middle layers contain many *conditional strengthening* neurons, and late layers tend more towards *weakening*. (iii) Thanks to the RW perspective, we discover a small class of neurons (*weakening* neurons), that is highly influential in often surprising ways: they activate often (in the sense of having a gate value above zero), and they influence various metrics, even in earlier layers where they are very rare. (iv) We introduce a new method of conditional ablation that enables us to find *which activations* of a given neuron are responsible for a certain behavior. (v)
1We publish code at https://anonymous.4open.science/r/RW_functionalities-4D32. 2We use "neuron" to refer to a hidden dimension inside the MLP layer. 3We use "weight" to refer to a weight vector, not a scalar.

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 Applying this method to weakening neurons, we find that some of their effect is due to cases in which their gate value is negative. Thus, for the first time, we observe a mechanism involving negative values of the Swish activation function.

## 2 Related Work

There is a large body of work on interpretability of transformer-based LLMs. Elhage et al. (2021) introduce the notion of residual stream. nostalgebraist (2020), Belrose et al. (2023) propose to interpret residual stream states as intermediate guesses about the next token; Rushing & Nanda (2024) discuss this as the *iterative inference hypothesis*. On a similar note, many works hypothesize that directions in model space can correspond to concepts; Park et al. (2024) discuss this as the linear representation hypothesis. Lad et al. (2024) define *stages of inference*. Similar to our work, Elhelo & Geva (2024) investigate input-output functionality of heads (instead of neurons). Much research has attempted to understand individual neurons. Geva et al. (2021) present them as a key-value memory. Other neuron analysis work includes (Miller & Neo, 2023; Niu et al., 2024). The focus on individual neurons has been criticized. Morcos et al. (2018) find that in good models, neurons are not monosemantic (but for image models, not LLMs). Millidge & Black (2022) find interpretable directions that do not correspond to individual neurons. Elhage et al. (2022) argue that interpretable features are non-orthogonal directions in model space and can be superposed. This corresponds to sparse linear combinations of neurons in MLP space. This has inspired a series of work on sparse autoencoders (SAEs), starting with Sharkey et al. (2022).

The focus on SAEs has been criticized: recent studies indicate that they do not always outperform baselines (Kantamneni et al., 2025; Leask et al., 2025; Mueller et al., 2025; Wu et al., 2025). A middle ground is possible: Gurnee et al. (2023) argue that interpretable features correspond to sparse combinations of neurons; this includes 1-sparse combinations, i.e., individual neurons. Accordingly, there is recent work on new classes of interpretable neurons, e.g. Ali et al. (2025); Zhao et al. (2025). Several works classify neurons based on the **contexts** in which they activate (Voita et al., 2024; Gurnee et al., 2024). For example, Voita et al. (2024) find *token detectors* that suppress repetitions. Gurnee et al. (2024) also define *functional roles* of neurons based on their **output** weight vector, such as *suppression neurons* that suppress a specific set of tokens. Stolfo et al. (2024) also investigate some output-based neuron classes. There has been less focus on the input-output perspective. Gurnee et al. (2024) compute cosine similarities between input and output weights for GPT-2 (Radford et al., 2019), but do not interpret their results. Elhage et al. (2021) mention the idea of input-output analysis (footnote 7), but do not follow up. Note that input-output analysis for gated activation functions adds complexity because, in addition to input and output weight vectors, the gating mechanism is crucial for RW functionality.

## 3 Preliminaries 3.1 Gated Activation Functions

We work on *gated activation functions* like SwiGLU or GEGLU (Shazeer, 2020). Gated activation functions are used widely, e.g., OLMo (Groeneveld et al., 2024) and Llama (Touvron et al., 2023) use SwiGLU, and Gemma (Gemma, 2024) uses GEGLU. Here we briefly describe SwiGLU. GEGLU
replaces Swish with GELU, but is otherwise identical. Traditional activation functions like ReLU require one weight matrix on the input side and one on the output side: The MLP outputs WoutReLU(Winxnorm),
where ReLU is applied element-wise to each neuron (it takes a single scalar as argument).4 Unlike ReLU, gated activation functions can output arbitrary positive or negative values. For example, if xgate > 0 and xin ≪ 0, then SwiGLU(xgate, xin) ≪ 0.

## 3.2 Weight Preprocessing

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 In this framework, SwiGLU can be described as a function of two scalars:
SwiGLU(xgate, xin) := Swish(xgate) · xin, Our code uses TransformerLens (Nanda & Bloom, 2022). When a model is loaded, certain preprocessing steps are applied to the weights, in order to make the weights more interpretable without changing model behavior (see their documentation for details).

We propose an additional preprocessing step specific to gated activation functions: For each neuron, we multiply win and wout by the sign of cos(wgate, win). See section C for our argument on why we do this and why it does not change model behavior.

## 4 Method 4.1 Approach

Other traditional activation functions are Swish(x) := x/(1 + exp(−x)) (Ramachandran et al., 2017) and GELU(x) := xΦ(x)
5(Hendrycks & Gimpel, 2016). Both of these can be seen as smooth approximations of ReLU. They are known to work better than ReLU, which is widely believed to be because of their good differentiability (e.g. Lee, 2023), i.e. better training dynamics. In contrast to these traditional functions, a *gated activation function* like SwiGLU requires two weight matrices on the input side: The MLP outputs Wout (Swish(W*gate*xnorm) ⊙ (Winxnorm)), (1)
where ⊙ denotes element-wise multiplication (a.k.a. Hadamard product).6 We find it more intuitive to separately consider each neuron: The neuron adds the vector Swish(⟨wgate, xnorm⟩) · ⟨win, xnorm⟩ · wout (2)
to the residual stream. Here wgate, win are each one of the dMLP *rows* of Wgate,Wout, and wout is one of the dMLP *columns* of Wout.

7 These weight vectors, as well as xnorm, all have dimensionality dmodel.

8 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215

| Table 1:            | Six prototypical read-write (RW) functionalities. See section 4.2 for details. | cos(wgate, wout)| ≈ 1 ≈ 0 (or > 0.5) (or < 0.5)   |             |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------|-------------|
| cos(win, wout) ≈ +1 | strengthening                                                                                                                      | conditional |
| (or > +0.5)         | strengthening                                                                                                                      |             |
| ≈ −1                | weakening                                                                                                                          | conditional |
| (or < −0.5)         | weakening                                                                                                                          |             |
| ≈ 0                 | proportional                                                                                                                       | orthogonal  |
| (or ∈ [−0.5, +0.5]) | change                                                                                                                             | output      |

Neuron-based. This cosine similarity method could in principle also be applied to SAE-style features instead of neurons, as long as each feature has well-defined input and output weights. However, for this paper we decided to just investigate neurons, and defer a possible investigation of SAEs to future work. We do so for the following reasons: (i) As argued in section 2, individual neurons can still be a promising research direction today. (ii) For any given LLM, neurons are readily available and clearly defined, whereas researchers may have published several SAEs with different sizes and architectures
- or, for less popular models, no SAE at all. (iii) We expect that findings from neurons will, to some extent, carry over to linear combinations of neurons. See section D for more on this. In section 5 we will see that, despite being "only" weight-based and neuron-based, our method already yields striking results.

## 4.2 Taxonomy Of Rw Functionalities

We now think through what different combinations of weight cosine similarities would mean for neuron RW functionality, and introduce our terminology. For the moment we focus on the prototypical cases, in which cosine similarities are approximately ±1 or 0. We present a taxonomy of these prototypical RW functionalities in table 1. Generally, when the output weight is similar enough to (one of) the detected directions, we speak of input manipulation, as opposed to **orthogonal output** neurons which write to directions not detected in the input. Intuitively, input manipulator neurons *manipulate* the concept that they detect. As special cases of input manipulation, we define: (i) **Strengthening** and **weakening** neurons: all three weight vectors are roughly collinear, and specifically cos(win, wout) ≈ ±1. The neuron detects a direction and then adds it to / removes it from the residual stream. (ii) **Conditional strengthening /**
weakening neurons: win and wout are roughly collinear and wgate is orthogonal to them. The neuron also strengthens / weakens the direction detected by its win vector, but will only activate *conditional* on wgate *being present in the residual stream*. (iii) **Proportional change** neurons: wout is collinear to wgate, but is orthogonal to win. If wgate is present in the residual stream, then the neuron writes a positive or negative multiple of this direction to the residual stream. This multiple is proportional to the presence of win in the residual stream.

These prototypical classes are limited in scope: Many cosines will not be close to 0 or ±1. For this general case, this paper explores three options to understand neuron RW functionalities at different levels of granularity: (1) Classify neurons according to the closest prototypical case (we choose a threshold τ = ±0.5). (2) Plot the marginal distributions of the three cosine similarities. (3) Place neurons in a scatter plot, based on their three weight cosines.

In option 1 (threshold-based classification), cos(win, wgate) may not always "match" the other two cosine similarities. Consider for example the case of strengthening: In the prototypical case with exact equalities (cos(win, wout) = cos(wgate, wout) = 1), all three weight vectors are collinear, so we also have cos(win, wgate) = 1. But without exact equalities (if we just know cos(win, wout) and cos(wgate, wout) are both above 0.5), it does not follow that cos(win, wgate) is also above 0.5.

9 When such a "mismatch" occurs, we prepend *atypical* to the category's name: In this example, we will 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

## 5 Where To Find Weakening Neurons

Figure 1: (a) Median of cos(win, wout) by layer (x-axis) for 9 models of 2B to 9B parameters. For all models, the value is positive in the beginning and negative in the end, indicating that early-middle layers "strengthen" directions they find in the residual stream whereas later layers tend more towards
"weakening" them. (b) Distribution of neurons by layer and category.

![4_image_1.png](4_image_1.png)

Figure 2: Fine-grained analysis of neuron RW behavior in three layers of Llama-3.2-3B, based on the configuration of their three weight vectors in parameter space. Each subplot represents a layer, each dot a neuron. The red lines mark the 95% randomness regions for each of the three cosine values. (There is a dotted line for variant (i) and a dashed line for variant (ii) in section 4.3, but they are almost the same.) We see that many neurons are outside the randomness regions, indicating that they manipulate their input in some way. Purple dots at the top of the plots are conditional strengthening neurons. Lighter dots in the bottom left corner are weakening neurons. speak of an atypical strengthening neuron. In figure 1(b) we will see that such neurons exist, but are quite rare overall.

## 4.3 Random Baselines

Given a cosine similarity between weight vectors, we test if it is significantly different from random.

To do so, we consider two random baselines: (i) i.i.d. Gaussian vector entries (as in a randomly initialized model); (ii) a layer-specific baseline based on "mismatched cosines". See section E for details. In practice, we find that both baselines give quite similar 95% randomness ranges.

In this section we compute cosine similarities of neuron weights as described in section 4, to investigate which RW functionalities actually appear in LLMs, and in which layers. Strikingly, our results are **consistent across models**. Across models, we find that there is a small but (as we will see later) influential number of **weakening neurons**, mostly in late layers. Other RW functionalities appear in other ranges: in particular, early-middle layers of all models contain a lot of conditional strengthening neurons.

![4_image_0.png](4_image_0.png) 

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 Concretely, we apply our method to 12 LLMs: Gemma-2-2B, Gemma-2-9B (Gemma, 2024), Llama2-7B, -3.1-8B, -3.2-1B, -3.2-3B (Touvron et al., 2023), OLMo-1B, OLMo-7B-0424 (Groeneveld et al., 2024), Mistral-7B (Jiang et al., 2023), Qwen2.5-0.5B, Qwen2.5-7B (Yang et al., 2024), Yi-6B (01.AI et al., 2025). These models use SwiGLU, except for Gemma, which uses GeGLU. To demonstrate our finding in more detail, we present three representative plots. (See section J for more.) Figure 1(a) shows the median value of cos(win, wout) across all layers of the nine larger models. The common pattern is clearly visible: In early-middle layers of all models, a majority of neurons has a cos(win, wout) high above zero, indicating strengthening; in late layers, this median cosine similarity goes slightly below zero, indicating a relative majority of weakening neurons. The other two plots focus on **Llama-3.2-3B**, but the patterns we describe are general: see section J for other models. Figure 1(b) shows RW class distribution across layers. In figure 2, we plot the distribution of neurons in a few selected layers, by displaying each neuron as a point with cos(wgate, wout) indicated on the x-axis, cos(win, wout) on the y-axis and cos(wgate, win) as its color. Input manipulation. First, we see that a large proportion of neurons are input manipulators (i.e., they are not orthogonal output neurons): In figure 1(b), these are 25% of all neurons, and as much as 50% in early-middle layers (layers 7–1110). What is more, figure 2 shows that even the neurons classified as "orthogonal output" often belong to clusters that are centered above/below the horizontal line. Their weight cosine similarities often exceed the significance threshold. E.g., in layer 14, there are many neurons whose cos(win, wout) (y-axis) is below 0.5 but above the significance threshold.

This suggests that even the "orthogonal output" neurons perform input manipulation to some extent. Different RW functionalities. Weakening neurons represent a large share of the (relatively few) input manipulators in late layers. They form a somewhat separate cluster in figure 2 (in the bottomleft corner of the rightmost subplot). Another important input manipulator class in late layers is proportional change. In contrast, across all models, early middle layers are dominated by conditional strengthening. In fact, the majority of input manipulators (more than 80% in Llama) belong to just this one class. This general pattern of strengthening-then-weakening holds across models, as figure 1(a) shows at one glance. In figure 2 (and figure 52 in the appendix), the pattern manifests as a large cluster of neurons, centered clearly above the x-axis in most layers, but moving below it in the last few layers. In summary, we find across models that conditional strengthening dominates in early-middle layers, but in late layers we find more weakening neurons.

## 6 Ablation Experiments

Since model training produced so many input manipulator neurons, we hypothesize that they must contribute to model performance in an important way. We now test this hypothesis by ablating neurons based on their RW functionality. We find that weakening neurons have the highest effect on the metrics that we tested - this is completely unexpected since weakening neurons are a very small class of a few hundred neurons. In the rest of the paper, we run a model on a dataset - whereas in section 5 we just applied our weight-based method from section 4. Therefore, to save resources, we focus on a single model: We choose **OLMo-7B**, because its training dataset, Dolma (Soldaini et al., 2024), is publicly available and its RW functionalities mostly follow the typical patterns. As a dataset, we use a random subset of 20M tokens from Dolma,11 except for attribute rate, where we follow the setup of Geva et al. (2023).

## 6.1 Exploring Rw Classes And Metrics

We run the model on our dataset and record various metrics, such as the loss. In each run we ablate a number of neurons from a different RW class, or (as a baseline) the same number of *random* neurons from the same layers. This enables us to observe the effect of various RW classes on these metrics.

The baseline verifies if effects are due to the layers rather than RW classes.

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

![6_image_1.png](6_image_1.png)

![6_image_0.png](6_image_0.png)

## Figure 3: Effect Of Zero-Ablating Weakening Neurons.

The main metrics we consider are attribute rate (Geva et al., 2023) and entropy of the output distribution. We justify these choices in section F. We try two types of ablation: zero ablation (setting activations to zero), and mean ablation (setting them to the mean activation of the given neuron).

We find that **ablating weakening neurons has a large effect** on both metrics, and this effect is not seen with other classes or with other neurons from the same layers. The effect is clearest with zero ablation, but also present with mean ablation (see section F.4 for mean ablation results). For attribute rate, the effect is most visible in layers ≈ 10 and onward. See figure 3(a). This is particularly interesting since there are very few weakening neurons in these early-middle layers. The case of entropy is also striking: Figure 3(b) shows that ablating weakening neurons often makes the output distribution flatter; in other words weakening neurons make the output distribution *sharper*. We would expect the opposite: removing information from the residual stream should make it less informative and therefore flatten the output distribution.

## 6.2 Conditional Ablations

We now further investigate the effect of weakening neurons on entropy. We use **conditional ablations**:
We ablate only some activations of each neuron, based on the signs of the corresponding xgate and xin.

Specifically, we consider the following four conditions (using the preprocessing from section 3.2):
(i) xgate > 0, xin > 0, leading to xpost > 0; (ii) xgate > 0, xin < 0, leading to xpost < 0; (iii) xgate < 0, xin < 0, leading to xpost > 0; (iv) xgate < 0, xin > 0, leading to xpost < 0. We find that a large part of the sharpening effect of weakening neurons is due to case (iii): In figure 3(b), case (iii) (bottom left subplot) shows entropy effects similar to those of weakening neurons as a whole, whereas this is much less the case for the other subplots. This is surprising, but also solves the mystery we encountered earlier (i.e., we expected weakening neurons to flatten the distribution, but in reality they often sharpen it).

It is *surprising* for two reasons: First, these negative xgate activations are relatively rare in weakening neurons (as we will see in section 7). Second, because of the Swish function, **negative gate values** are relatively small (whereas positive values can be arbitrarily large), and it was often assumed they were 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

## 8 Case Study Of A Weakening Neuron 6.3 Case Study Of Entropy Reduction

To understand this phenomenon further, we study a particular text example, namely where the entropy reduction by case (iii) activations of weakening neurons was most extreme (with zero ablation). The input text is: Yesterday (21 December) the Government announced a package of support for hospitality and leisure businesses that are losing trade because of the O and the correct next token is mic (as in *Omicron*). The model predicts this next token correctly. Which tokens have the largest score difference between clean and ablated runs? We find that, in the clean run, mic and similar tokens get a massive boost (of up to 12 points) compared to the ablated run, whereas no token gets its score reduced by nearly as much. Thus, at least in this case, the case (iii) activations of weakening neurons sharpen the output distribution by boosting the correct next token.

We further investigate whether any single weakening neuron has a wout similar to mic. This is not the case. This suggests that in this case weakening neurons work together (in superposition, cf. Elhage et al., 2022) to achieve the observed effect.

## 7 Weakening Neurons Activate Often

Our findings from section 6 raise the question of how often weakening neurons activate, i.e., how often their gate value is positive. In fact, Gurnee et al. (2024) found a negative correlation between activation frequency and cos(win, wout)
- but in a GELU model. We now investigate whether a similar phenomenon occurs with gated activation functions. We show the most striking result in figure 4. We show other results in section J and discuss some of them in section G.

Consistent with Gurnee et al. (2024), we find that the many (conditional) strengthening neurons activate very rarely, and (conditional) weakening neurons activate very often. In fact, in most layers there is an almost linear negative relationship between cos(win, wout) and activation frequency: correlations are at least −0.71 in all layers except the last two (which have −0.29 and +0.29). It seems that each conditional strengthening neuron is responsible only for a narrow domain, perhaps a specific set of tokens. This result is another indication that weakening neurons have a disproportionately large influence on model behavior. Note however that activation frequencies do not fully explain their effect, since we found that even their negative gate values are influential (section 6).

Figure 4: Two-dimensional histogram of activation frequency (x-axis) vs. cos(win, wout) (y-axis).

We now qualitatively examine two neurons in more detail: a strengthening and a weakening neuron. We will see that weakening neurons can have a quite complex behavior. In section I we detail the only useful for training dynamics (see section 3.1). Our results show for the first time (concurrently with Kong et al. (2025) who focus on a different phenomenon) that negative gate values have a strong effect on model mechanisms (not just training). This shows that, for mechanistic interpretability research, Swish is not reducible to ReLU.

This is also *explanatory*: When xgate <0, the usual neuron behavior gets a minus sign in front, so that weakening neurons take on a strengthening behavior. E.g., if a neuron usually detects "minus *again*"
(wgate) and writes "*again*" (wout), in case (iii) it detects "*again*" (−wgate) and writes "again" (wout),
which indeed makes the output distribution sharper. We will analyze such a neuron in section 8.

![7_image_0.png](7_image_0.png) 

methods (including how we chose the neurons), present this analysis in more detail, and present many more case studies from various RW functionalities. To analyze the neurons, we combine the RW perspective with two well-established neuron analysis methods: projecting weights to vocabulary space (nostalgebraist, 2020; Geva et al., 2022; Dar et al., 2023; Gurnee et al., 2024; Voita et al., 2024), and finding text examples which strongly activate the neuron (Dalvi et al., 2019; Geva et al., 2021; Nanda, 2022; Voita et al., 2024; Gurnee et al., 2024).

We choose neuron **28.4737** for strengthening and **31.9634** for weakening.12 Both of them are also a prediction neuron in the sense of Gurnee et al. (2024), which indicates that they directly promote a specific set of tokens and are thus likely to be monosemantic. We can see that **strengthening neuron 28.4737** has a straightforward read-write behavior: It further promotes *review* when the residual stream already indicates that this is the obvious next token. In contrast, **weakening neuron 31.9634** is much harder to interpret: The weights indicate that this neuron produces "*again*" when the residual stream contains "minus *again*"; but the examples strongly activating the neuron do not have an obvious semantic relationship to *again*. The most interpretable activations were some weaker positive activations when *again* is a plausible continuation, e.g., on the token *once* (as in *once again*). These are cases with negative xgate values (and also xin < 0, hence positive activations) - a case that we found to be important in section 6.2.

In these cases, *again* is already weakly present in the residual stream before the last MLP, and the neuron reinforces *again*. Thus the behavior of this particular weakening neuron is interpretable in the xgate < 0 case, echoing our finding from section 6.2 that this case is surprisingly relevant to model behavior. These two case studies show that even when the output weights are highly interpretable, strengthening and weakening have a very different overall behavior, and the weakening behavior is much more complex. We think that this is due to the nature of weakening: it inherently involves (an apparent) conflict between the intermediate model prediction and what the neuron promotes.

## 9 Conclusion

We have explored a new method for analyzing gated neurons in LLMs: computing the cosine similarities of their weights to understand their read-write functionality. Our method complements prior interpretability approaches and, though quite simple, provides striking new insights into the inner working of LLMs. We have found that a large share of neurons exhibit strong RW interactions: early-middle layers are dominated by conditional strengthening neurons; weakening neurons are fewer and appear mostly in late layers. This finding is particularly noteworthy since it is universal across models, and is all the more striking since it could be observed with such a simple method. Focusing on weakening neurons, a relatively small RW class, we have discovered that they have an outsize impact on model behavior, including aspects as different as attribute rate (part of factual recall), and next-token entropy. We have also introduced a new analysis method, conditional ablation, which enables to find out which activations of a neuron are responsible for a given behavior. This method has shown that part of the impact of weakening neurons is due to a mechanism involving negative gate values; we are the first to observe such a mechanism. Our findings open up new research questions in mechanistic interpretability, and we hope that our study will inspire further investigations. In particular, a better understanding of weakening neurons is crucial for interpreting LLMs overall. Investigating the many conditional strengthening neurons in more detail could also lead to valuable insights. In upcoming work, we plan to investigate the evolution of RW functionalities during model training. Later on, we would also like to go beyond the analysis of single neurons and address questions such as how neurons work together within and across RW classes, or whether a similar analysis also works for SAE latents.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

## References

01.AI, :, Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Guoyin Wang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, Kaidong Yu, Peng Liu, Qiang Liu, Shawn Yue, Senbin Yang, Shiming Yang, Wen Xie, Wenhao Huang, Xiaohui Hu, Xiaoyi Ren, Xinyao Niu, Pengcheng Nie, Yanpeng Li, Yuchi Xu, Yudong Liu, Yue Wang, Yuxuan Cai, Zhenyu Gu, Zhiyuan Liu, and Zonghong Dai. Yi: Open foundation models by 01.ai, 2025. URL https://arxiv.org/abs/2403.04652.

Ameen Ali, Shahar Katz, Lior Wolf, and Ivan Titov. Detecting and pruning prominent but detrimental neurons in large language models. *COLM*, 2025. URL https://arxiv.org/pdf/2507.09185.

Nora Belrose, Zach Furman, Logan Smith, Danny Halawi, Igor Ostrovsky, Lev McKinney, Stella Biderman, and Jacob Steinhardt. Eliciting latent predictions from transformers with the tuned lens, 2023. URL https://arxiv.org/pdf/2303.08112.

Fahim Dalvi, Nadir Durrani, Hassan Sajjad, Yonatan Belinkov, Anthony Bau, and James Glass. What is one grain of sand in the desert? analyzing individual neurons in deep NLP models. *AAAI*, 2019. doi: https://doi.org/10.1609/aaai.v33i01.33016309.

Guy Dar, Mor Geva, Ankit Gupta, and Jonathan Berant. Analyzing transformers in embedding space. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pp. 16124–16170, Toronto, Canada, July 2023. Association for Computational Linguistics. doi:
10.18653/v1/2023.acl-long.893. URL https://aclanthology.org/2023.acl-long.893.

Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. LLM.int8(): 8-bit matrix multiplication for Transformers at scale. *NeurIPS*, 2022. URL https://proceedings.neurips. cc/paper_files/paper/2022/file/c3ba4962c05c49636d4c6206a97e9c8a-Paper-Confere nce.pdf.

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A mathematical framework for transformer circuits, 2021. URL https://transformer-circuits. pub/2021/framework/index.html.

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy models of superposition, 2022. URL https://transformer-circuits.pub/2022/toy_model/index.html.

Amit Elhelo and Mor Geva. Inferring functionality of attention heads from their parameters, 2024.

URL https://arxiv.org/abs/2412.11965.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Kawin Ethayarajh. How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing
(EMNLP-IJCNLP), pp. 55–65, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1006. URL https://aclanthology.org/D19-1006.

Team Gemma. Gemma. 2024. doi: 10.34740/KAGGLE/M/3301. URL https://www.kaggle.com
/m/3301.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wentau Yih (eds.), Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 5484–5495, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.446. URL https://aclanthology.org/2021.emnlp-main.446.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Mor Geva, Avi Caciularu, Kevin Wang, and Yoav Goldberg. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 30–45, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.3. URL https://aclanthology.org/2022.emnlp-main.3.

Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. Dissecting recall of factual associations in auto-regressive language models. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pp. 12216–12235, Singapore, December 2023. Association for Computational Linguistics. doi: 10 .18653/v1/2023.emnlp-main.751. URL https://aclanthology.org/2023.emnlp-main.751.

Dirk Groeneveld, Iz Beltagy, Evan Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, Tushar Khot, William Merrill, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew Peters, Valentina Pyatkin, Abhilasha Ravichander, Dustin Schwenk, Saurabh Shah, William Smith, Emma Strubell, Nishant Subramani, Mitchell Wortsman, Pradeep Dasigi, Nathan Lambert, Kyle Richardson, Luke Zettlemoyer, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah Smith, and Hannaneh Hajishirzi. OLMo: Accelerating the science of language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), *Proceedings of the 62nd Annual Meeting of the* Association for Computational Linguistics (Volume 1: Long Papers), pp. 15789–15809, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-lon g.841. URL https://aclanthology.org/2024.acl-long.841.

Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, and Dimitris Bertsimas.

Finding neurons in a haystack: Case studies with sparse probing, 2023. URL https://arxiv.or g/pdf/2305.01610.

Wes Gurnee, Theo Horsley, Zifan Carl Guo, Tara Rezaei Kheirkhah, Qinyi Sun, Will Hathaway, Neel Nanda, and Dimitris Bertsimas. Universal neurons in gpt2 language models, 2024. URL https://arxiv.org/pdf/2401.12181.

Dan Hendrycks and Kevin Gimpel. Bridging nonlinearities and stochastic regularizers with gaussian error linear units. *CoRR*, 2016. URL http://arxiv.org/abs/1606.08415.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023. URL https://arxiv.org/ab s/2310.06825.

Subhash Kantamneni, Joshua Engels, Senthooran Rajamanoharan, Max Tegmark, and Neel Nanda.

Are sparse autoencoders useful? a case study in sparse probing. *arXiv*, 2025. URL https:
//arxiv.org/pdf/2502.16681.pdf.

Linghao Kong, Angelina Ning, Micah Adler, and Nir Shavit. Negative pre-activations differentiate syntax. *arXiv*, 2025. URL https://arxiv.org/pdf/2509.24198.pdf.

Olga Kovaleva, Saurabh Kulshreshtha, Anna Rogers, and Anna Rumshisky. BERT busters: Outlier dimensions that disrupt transformers. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.), *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, pp. 3392–3405, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/ 2021.findings-acl.300. URL https://aclanthology.org/2021.findings-acl.300.

Vedang Lad, Wes Gurnee, and Max Tegmark. The remarkable robustness of llms: Stages of inference?,
2024. URL https://arxiv.org/abs/2406.19384.

Patrick Leask, Bart Bussmann, Michael Pearce, Joseph Bloom, Curt Tigges, Noura Al Moubayed, Lee Sharkey, and Neel Nanda. Sparse autoencoders do not find canonical units of analysis. *arXiv*, 2025. URL https://arxiv.org/pdf/2502.04878.pdf.

Minhyeok Lee. Mathematical analysis and performance evaluation of the GELU activation function in deep learning. *Journal of Mathematics*, 2023. doi: https://doi.org/10.1155/2023/4229924.

Joseph Miller and Clement Neo. We found an neuron in gpt-2, 2023. URL https://www.lesswron g.com/posts/cgqh99SHsCv3jJYDS/we-found-an-neuron-in-gpt-2.

Beren Millidge and Sid Black. The singular value decompositions of transformer weight matrices are highly interpretable, 2022. URL https://www.lesswrong.com/posts/mkbGjzxD8d8XqKHzA/ the-singular-value-decompositions-of-transformer-weight.

Ari S. Morcos, David G.T. Barrett, Neil C. Rabinowitz, and Matthew Botvinick. On the importance of single directions for generalization, 2018. URL https://arxiv.org/pdf/1803.06959.pdf.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Aaron Mueller, Atticus Geiger, Sarah Wiegreffe, Dana Arad, Iván Arcuschin, Adam Belfki, Yik Siu Chan, Jaden Fiotto-Kaufman, Tal Haklay, Michael Hanna, Jing Huang, Rohan Gupta, Yaniv Nikankin, Hadas Orgad, Nikhil Prakash, Anja Reusch, Aruna Sankaranarayanan, Shun Shao, Alessandro Stolfo, Martin Tutek, Amir Zur, David Bau, and Yonatan Belinkov. MIB: a mechanistic interpretability benchmark. *arXiv*, 2025. URL https://arxiv.org/pdf/2504.13151.

Neel Nanda. Neuroscope. Website, 2022. URL https://neuroscope.io.

Neel Nanda and Joseph Bloom. Transformerlens. https://github.com/TransformerLensOrg/Tr ansformerLens, 2022.

Jingcheng Niu, Andrew Liu, Zining Zu, and Gerald Penn. What does the knowledge neuron thesis have to do with knowledge?, 2024. URL https://arxiv.org/pdf/2405.02421.

nostalgebraist. Interpreting gpt: The logit lens, 2020. URL https://www.lesswrong.com/posts/
AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens.

Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of *Proceedings of Machine Learning Research*, pp. 39643–39666. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/park2 4c.html.

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.

Prajit Ramachandran, Barret Zoph, and Quoc V. Le. Searching for activation functions. *arXiv*, 2017.

URL https://arxiv.org/pdf/1710.05941.

Cody Rushing and Neel Nanda. Explorations of self-repair in language models. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pp. 42836–42855. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/rushing24a.html.

Lee Sharkey, Dan Braun, and Beren Millidge. [interim research report] taking features out of superposition with sparse autoencoders, 2022. URL https://www.alignmentforum.org/posts
/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposi tion.

Noam Shazeer. Glu variants improve transformer, 2020. URL https://arxiv.org/pdf/2002.052 02.

Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Evan Walsh, Luke Zettlemoyer, Noah Smith, Hannaneh Hajishirzi, Iz Beltagy, Dirk Groeneveld, Jesse Dodge,
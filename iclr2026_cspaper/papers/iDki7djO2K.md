000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 A fundamental challenge in developing general learning algorithms is their tendency to forget past knowledge when adapting to new data. Addressing this problem requires a principled understanding of forgetting; yet, despite decades of study, no unified definition has emerged that provides insights into the underlying dynamics of learning. We propose an algorithm- and task-agnostic theory that characterises forgetting as a lack of self-consistency in a learner's predictive distribution over future experiences, manifesting as a loss of predictive information. Our theory naturally yields a general measure of an algorithm's propensity to forget. To validate the theory, we design a comprehensive set of experiments that span classification, regression, generative modelling, and reinforcement learning. We empirically demonstrate how forgetting is present across all learning settings and plays a significant role in determining learning efficiency. Together, these results establish a principled understanding of forgetting and lay the foundation for analysing and improving the information retention capabilities of general learning algorithms.

## 1 Introduction

Forgetting is a ubiquitous yet poorly understood phenomenon in machine learning (McCloskey & Cohen, 1989). When a learner updates its beliefs based on new observations, it often forgets prior knowledge. This leads to a degradation in performance on previous observations. Although this behaviour is well documented in continual learning (CL; Kirkpatrick et al., 2016) and reinforcement learning (RL; Atkinson et al., 2021; Khetarpal et al., 2022), forgetting also occurs in independent and identically distributed (i.i.d.) settings (Lee & Storkey, 2023). Most studies of forgetting come from the CL literature. Here, metrics usually track how performance degrades on earlier tasks after training on later tasks (Chaudhry et al., 2018a). Although widely adopted, these measures poorly capture forgetting and often conflate two distinct phenomena: backward transfer, where new learning improves performance on past tasks (Benavides-Prado & Riddle, 2022), and *forgetting*, where updates erode prior knowledge (Jagielski et al., 2022). This makes it challenging to distinguish between constructive and destructive adaptation.

In contrast, we study forgetting as a *fundamental property of learning*. It is a consequence of how any adaptive system updates its beliefs. CL, RL, and neural networks are not our focus. Instead, we treat them as instances of a broader phenomenon. We believe that before effective learning algorithms for CL can be developed, a precise conceptual understanding of forgetting must first be formulated. Therefore, our aim is to provide a general formalism that characterises forgetting in learning systems. This motivates a new conceptual foundation of forgetting, built on the following insight:
If a learner updates its predictions on data it already expects, that update cannot represent the acquisition of new information. Instead, it must represent the loss of previously acquired knowledge.

This allows us to give a precise and general definition of forgetting. We ask: *What is forgetting?*
When and why does it occur? How does forgetting impact learning?

We address these questions through the following contributions: 1. **Formulation of learning over time**: Inspired by Hutter (2005); Dong et al. (2022); Fortini &
Petrone (2019); Fong et al. (2023), we define a *general theoretical formulation* for reasoning about how learners acquire, retain, and lose capabilities during learning (§3).

## Anonymous Authors

Paper under double-blind review

## Abstract

1

# Forgetting Is Everywhere

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 2. **Conceptualisation of forgetting**: We formulate forgetting from a *predictive perspective*. Specifically, we define forgetting as a violation of *self-consistency* in the learner's predictive distribution, providing a unified probabilistic foundation that generalises prior conceptions of forgetting (§4.2).

3. **Operational measure**: This conceptualisation naturally yields a measure of the propensity to forget, which we propose to assess the validity of our formalism (Definition 4.6).

4. **Empirical characterisation**: We empirically study the propensity to forget in diverse environments and learning paradigms. These include regression, classification, generative modelling, CL, and RL. Our results confirm that forgetting dynamics conform to expected characteristics. They also reveal a trade-off between forgetting and training efficiency (§5).

## 2 Related Work

In this paper, we study the prevalence of forgetting in general learning algorithms. While terminology varies across fields, all phenomena involving the loss of previously acquired knowledge reflect the same underlying process. For clarity, we refer to all such phenomena collectively as *forgetting*.

Forgetting in CL. Forgetting is often studied in the context of CL (De Lange et al., 2021; Wang et al., 2024), where various proxy metrics measure forgetting in sequential learning settings, often as the loss of performance on previous tasks (Kirkpatrick et al., 2016; Chaudhry et al., 2018b;a). However, this approach cannot distinguish between two distinct effects: *backwards transfer*, where learning on new data improves performance on past tasks (Benavides-Prado & Riddle, 2022), and forgetting, where learning on new data degrades previously acquired knowledge (McCloskey & Cohen, 1989; Jagielski et al., 2022). Performance on previous tasks depends on both how current data informs the learner about past data (*backwards transfer*) and the extent to which updating on new tasks results in a loss of previous knowledge. Furthermore, most metrics are limited to CL settings and do not generalise to all learners or domains, motivating the need for a more general definition of forgetting. Various approaches have sought to quantify forgetting. In toy settings with known data distributions, forgetting can be measured relative to an oracle (Lee et al., 2021). Others examine changes in internal representations (Kim et al., 2025) or model the trade-off between generalisation and forgetting (Raghavan & Balaprakash, 2021). Mechanistic accounts in associative memory models demonstrate how interference between correlated patterns leads to retrieval errors (Hopfield, 1982; Amit et al.,
1985). A related line of work studies models trained on their own generated outputs, typically in the context of generative model collapse (Alemohammad et al., 2023; Shumailov et al., 2023; Bertrand et al., 2023; Scholten et al., 2025). These studies implicitly touch on the phenomenon we formalise as forgetting: updating a model on self-generated targets can degrade previously encoded knowledge.

Forgetting in RL. Forgetting has also been studied in RL. Early work in continual RL noted the risk of knowledge degradation when agents adapt over long time horizons (Ring, 1994; 1997). Recent surveys highlight that forgetting is a persistent challenge in RL (Khetarpal et al., 2022). Empirical studies document several forms of forgetting, with forgetting measured in a manner similar to the CL counterpart, either by tracking the degradation of performance on previously learned tasks or by quantifying the distributional shift in the agent's policy or value function over time (Shenfeld et al., 2025). Value-based methods with function approximation often lose performance on earlier estimates (Mnih et al., 2015; van Hasselt et al., 2018). This is closely related to the phenomenon of *policy churn* identified by (Schaul et al., 2022) in which the greedy policy of a value-based learner changes in a large portion of the input space after just a few updates. Policy gradient methods are also prone to overwriting earlier strategies during continual adaptation (Ring, 1994; Kirkpatrick et al., 2017). Replay buffers are widely used to mitigate these effects by reintroducing past experiences during training (Rolnick et al., 2019; Fedus et al., 2020).

Misconceptions of forgetting. Throughout the literature, we identify a general trend: forgetting is often characterised through a mechanism-specific lens. Model-centric views equate forgetting with parameter drift (McCloskey & Cohen, 1989; French, 1999; Rusu et al., 2016; Kirkpatrick et al., 2017; Zenke et al., 2017; Aljundi et al., 2018; Masse et al., 2018; Li et al., 2024; Zhao et al., 2023), or policy drift (Shenfeld et al., 2025), while accuracy-centric views characterise forgetting as performance decay on earlier tasks (Kemker et al., 2018; Parisi et al., 2019; Jagielski et al., 2022; Van de Ven et al., 2022). This fragmentation has hindered the development of coherent principles for understanding and mitigating forgetting. Moreover, these definitions mischaracterise forgetting and are constrained by specific models or tasks. Not all parameter or policy changes imply forgetting: not all learners have parameters, yet in those that do, parameters can change without the learner necessarily forgetting anything. We demonstrate this in §5.1 with a learner whose parameters change without causing forgetting. Motivated by these limitations, we treat the predictive distribution as the object of interest. We propose a principled conceptualisation of forgetting based on predictive self-consistency that applies broadly across learning paradigms. We further demonstrate the generality of our conceptualisation and show how it supports an empirical measure of forgetting that aligns with theoretical expectations.

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 3.1 PRELIMINARIES: LEARNING INTERACTION We present a general framework in which supervised learning, RL, and generative modelling are all specific cases of a single stochastic interaction process. The formalism draws inspiration from the agent-environment perspective from general RL (Hutter, 2005; Lattimore, 2014; Dong et al., 2022; Abel et al., 2023; Hutter et al., 2024; Kumar et al., 2025), but is adapted to emphasise conventions standard in machine learning (Bishop, 2006; Goodfellow et al., 2016; Fong et al., 2023).

$$p:{\mathcal{X}}\times{\mathcal{Y}}\to{\mathcal{D}}({\mathcal{Z}})$$

Notation. We let capital calligraphic letters denote measurable spaces (X )
1, lowercase letters denote elements (x) or functions (f), and uppercase italics denote random variables (X). For any measurable space X , we let D(X ) denote the set of probability distributions over X . For example, a mapping p : *X × Y → D*(Z) (1)
is interpreted as a family of conditional distributions p(· | *x, y*) on Z, indexed by (x, y) *∈ X × Y*.

We formalise learning as an ongoing interaction between a *learner* and an *environment*, evolving over discrete time steps t ∈ N0 = {0, 1*, . . .* }. We distinguish between two sources of stochasticity:
- *External probabilities* pe(·) describe the stochasticity inherent to the environment. - *Predictive distributions* qf (·) describe the predictive uncertainty of the learner.

To formalise this interaction, we define a structure that describes the data exchanged between the learner and the environment.

Definition 3.1 (Interface). An *environment-learner interface* is a pair (X , Y) of measurable spaces. Elements of X are *observations*, encompassing the signals the environment emits (e.g., observations, targets, etc.), and elements of Y are *outputs*, encompassing the signals emitted by the learner (e.g.,
actions, classifications, regression values, etc.).

Definition 3.2 (Histories). The set of *histories* relative to an interface (X , Y) is

$${\mathcal{H}}=\bigcup_{t=0}^{\infty}({\mathcal{X}}\times{\mathcal{Y}})^{t+1}.$$

$$(2)$$

t+1. (2)
We refer to an element of H as a history H ∈ H, which is a sequence H0:t = ((X0, Y0), . . . ,(Xt, Yt))
of t + 1 observation-output pairs (Xi, Yi). For a sequence A = (A0, A1*, . . . , A*t) and indices 0 ≤ i ≤ j ≤ t, we denote the subsequence between indices i and j as Ai:j = (Ai, Ai+1*, . . . , A*j ).

Definition 3.3 (Environment). An *environment* relative to interface (X ,Y) is a pair (*e, p*X0
), where
- e : H, *Y → D*(X ) assigns to each history-current output pair (*H, Y* ) ∈ (H, Y) a conditional distribution pe(· | *H, Y* ) over the next observation,
- pX0 ∈ D(X ) specifies the distribution of the initial observation X0.

1We assume that all measurable spaces (X, Y, Z*, . . .*) are standard Borel (Borel σ-algebras of Polish spaces),
such that conditional probability kernels p(· | *h, y*) and q(· | *z, x*) exist and are measurable.

## 3 Learning And Inference Processes

Definition 3.4 (Learner). A *learner* relative to interface (X , Y) is a tuple (Z, f, u, u′, pZ0), where:
- a measurable space Z, called the *learner state space*; - a prediction function f : *Z × X → D*(Y), giving conditional distributions qf (· | *z, x*); - a learning-mode state update function u : *Z × X × Y → D*(Z);
- an *inference-mode state update function* u
′: *Z × X × Y → D*(Z);
- an initial learner state distribution pZ0 ∈ D(Z).

Although a learner's entire state could, in principle, be updated by a single function (as in work by Dong et al., 2022 and Kumar et al., 2025), we distinguish between two update functions to capture different modes of learner evolution. During interaction with the environment, the learning-mode update u governs the evolution of the entire state, including predictive parameters and auxiliary components, such as replay buffers. In contrast, the inference-mode update u
′allows auxiliary components of the state, such as buffers or counters, to evolve while keeping the predictive parameters fixed. This distinction allows an observer to analyse the learner's behaviour in both dynamic conditions, where beliefs are updated over time, and in static conditions, where beliefs remain fixed.

The interaction process. The interaction between a learner and an environment defines a joint stochastic process over observations X ∈ X , outputs Y ∈ Y, and states Z ∈ Z.

Definition 3.5 (Interaction Process). The *(learning) interaction process* between an environment
(*e, p*X0) and a learner (Z, f, u, u′, pZ0) relative to an interface (X , Y) is defined by:
Initialisation: Z0 ∼ pZ0, X0 ∼ pX0, Interaction: Yt ∼ qf (· | Zt−1, Xt−1), (learner samples output)
Xt ∼ pe(· | H0:t−1, Yt), (environment samples current observation) Zt ∼ u(Zt−1, Xt, Yt). (learner updates state)
This interaction process generates the stochastic processes Xt (t ≥ 0), Yt (t > 0), and Zt (t ≥ 0).

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215

$\tau$ = $\tau$. 
$$\begin{array}{l}{{X_{t}\sim p_{e}(\cdot\mid H_{0:t-1},Y_{t}),}}\\ {{Z_{t}\sim u(Z_{t-1},X_{t},Y_{t}).}}\end{array}$$

## 3.2 Predictive Distributions

At any time during learning, the learner's state encodes expectations about how future interactions will unfold. We can make these expectations explicit by asking the learner to *internally simulate the future*. Rather than waiting for new observations, we let the learner "roll forward" in inference mode using update u
′to project how the sequence of inputs and outputs may continue. This is very similar to placing a learner into inference mode, where it projects how the sequence of outputs might continue. We refer to such a simulation as an induced future or *predictive distribution*. It represents a hypothetical rollout of the learner's predictive model under its current beliefs, independent from the environment. To distinguish this hypothetical evolution from real learning progression, we introduce a future time index s ∈ N and denote steps within the simulated trajectory as superscripts on X, Y , and Z.

During the rollout, the learner generates targets from its own predictive distribution, while any components not represented by the learner are *borrowed* from the environment. Formally, given a history H0:t and state Zt, the predictive distribution evolves according to Y
s ∼ qf (· | Z
s−1 t, Xs−1), Xs ∼ qe(· | H0:s−1, Y s), Zs t ∼ u
′(Z
s−1 t, Xs, Y s), (3)
where qe is a *hybrid distribution* that treats the learner's predictions as targets while borrowing components from the environment as needed, and s = (t + 1, t + 2*, . . .*). Progression along the futures index s provides a principled means to analyse the learner independently of new observations. We emphasise that this futures distribution is entirely isolated from the interaction process.

We refer to an element of future histories as H ∈ H, which is a sequence of future observation-output pairs (Xi, Y i) induced by the learner Ht+1:∞ = ((Xt+1, Y t+1),(Xt+2, Y t+2)*, . . .*).

Definition 3.6 (Predictive Distributions). The *predictive distribution* of the learner at state Zt with realised history H0:t is the joint distribution obtained by rolling out the inference-mode process in (3):
q(Ht+1:∞ | Zt, H0:t). (4)
Thus, an induced future is a probability distribution over entire infinite sequences of future inputs and outputs (*X × Y*)
N. Because the learner's state Zt and subsequent observations are random, the predictive distribution at each step q(· | Zt, H0:t) is a random variable taking values in D((*X × Y*)
N).

![4_image_0.png](4_image_0.png)

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 Predictive-Bayesian perspective. This construction motivates a *predictive perspective of learning*,
inspired by predictive Bayesianism (Fortini & Petrone, 2019; 2025; Fong et al., 2023). Here, the learner's state Zt is characterised by its predictive distribution over futures. This perspective offers several advantages: it allows predictive statements to be *validated* against realised outcomes (Fong et al., 2023) and provides an interpretable representation of the learner's knowledge at any time - this is valuable in deep learning because parameters are not interpretable. By tracking the evolution of the predictive distribution, we yield a general formulation of learning that allows us to define forgetting.

## 3.3 Learning As A Stochastic Process

The stochasticity of the environment induces stochasticity in the learner's state evolution; as the learner interacts with the environment, the sequence {Zt} forms a stochastic process. Since each state Zt defines a predictive distribution, the sequence of predictive distributions {q(Ht+1:∞ | Zt, H0:t)}
also forms a stochastic process. Thus, the distribution over learner states at time t defines a distribution over predictive distributions. This is illustrated in Figure 1.

Interpretation across learning paradigms. The abstract variables (Xt, Yt, Zt) admit natural interpretations across learning settings. In supervised learning, Xt denotes (current input, previous target) pairs and Yt the learner's output in response to the previous input; in RL, Xt contains states and rewards while Yt are actions; in generative modelling, Yt directly models Xt. Across these paradigms, the learner state Zt encapsulates all contents of the learner (including parameters, latent variables, buffers, etc.). Distinct modes of operation correspond to different update rules: in training (update rule u), both the learner's belief and auxiliary structures may adapt, whereas an inference update (u
′) leaves its beliefs fixed while transient components continue to evolve.

For example, in a supervised classification setting, Xt consists of a (current input, previous target) pair, Yt is the label predicted by the learner given Xt−1, and Zt may represent parameters and momenta. The learning-mode update u performs gradient steps on a loss that compares the observed and predicted labels. The inference-mode update u
′ does not change the parameters and momenta.

Thus, while the same symbols are used throughout, their interpretation shifts depending on the paradigm. What remains consistent is the structural principle: the learner generates Yt conditioned on (Zt−1, Xt−1), the environment supplies the observation Xt, and the state Zt evolves via u. This defines a single stochastic-process formalism that encompasses general learning processes.

## 4 Forgetting

We consider an observer monitoring the interaction between a learner and its environment. The observer evaluates how the learner's capabilities evolve over time, with particular emphasis on quantifying the extent to which the learner forgets during the interaction.

## 4.1 Characterising Forgetting

Before formalising forgetting, we first outline desiderata that any valid notion of forgetting should satisfy. These desiderata are motivated by thought experiments, detailed in §C. Desideratum 4.1. A forgetting measure should quantify the loss of learned information over time.

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

## 4.2 Forgetting

Forgetting is distinct from measurable success on a task, such as accuracy or cumulative reward. It is a property of the learner itself, independent of the environment. A learner can maintain outdated or incorrect beliefs yet still perform well, or lose relevant knowledge without an immediate change in task outcomes. Conventional metrics, including backward transfer, may fail to capture these underlying changes in knowledge, highlighting the need to disentangle forgetting from task performance. Desideratum 4.2. A characterisation of forgetting must not conflate forgetting with the correctness of outputs or with justified updates that change beliefs. When a learner incorporates new observations, its beliefs (and therefore its state) will change. A change in belief does not necessarily imply anything is forgotten. Therefore, conceptualisations based on changes to beliefs (or parameters) can misidentify information-preserving updates as forgetting.

Desideratum 4.3. Forgetting should characterise the learner's loss of prior information and capabilities, not just the retention of previously observed data.

Forgetting encompasses the loss of general capabilities, not only individual observations. For example, a learner may fail to generalise to unobserved examples that it previously handled correctly. Conceptualisations of forgetting that prioritise memorisation overlook this broader notion. Desideratum 4.4. Forgetting is a property of the learner, not of the environment in which it operates. An environment cannot forget; however, it can influence the rate or magnitude of forgetting. These desiderata provide a principled foundation for developing and evaluating conceptualisations of forgetting. The thought experiments in §C justify the desiderata and formalism developed below.

$$Z_{t}\sim u(Z_{t-1},X_{t},Y_{t}).$$

Learning is a process of both *absorption and loss*. Each new observation provides the learner with new information while simultaneously displacing or overwriting previously learned information. Therefore, every update reshapes what the learner can represent and, consequently, what it may forget. If a learner's behaviour changes after observing information that it has already incorporated from previous observations, this cannot reflect the acquisition of new information. Therefore, it indicates a loss of prior knowledge. The learner's expectations are formalised by its predictive distribution; consequently, simulating training updates on samples drawn from this predictive distribution provides a principled perspective on the learner's likelihood of forgetting at any point in time. Recall that the learner maintains a state that evolves recursively:
Zt ∼ u(Zt−1, Xt, Yt). (5)
At each time t, the state Zt induces a predictive distribution q(Ht+1:∞ | Zt, H0:t), which characterises both the learner's current predictive capabilities and its expectations about futures. We therefore define forgetting in terms of *induced futures*, rather than the learner's state. Different states–whether parametric, non-parametric, or otherwise–may induce identical futures:
q(Ht+1:∞ | Zt, H0:t) = q(Ht+1:∞ | Z
′
t, H0:t), (6)
where Zt ̸= Z
′t. Furthermore, predictions about already-seen observations may shift without indicating that forgetting has occurred. By grounding our definition in induced futures, we ensure that updates are evaluated with respect to the learner's *expected predictions*, thereby distinguishing between constructive and destructive adaptation. This satisfies Desideratum 4.2 and Desideratum 4.3. Consistency as non-forgetting. Intuitively, a learner is *unforgetful* if, in expectation, its predictive distribution is invariant to updates on targets that are consistent with its own predictions.

Formally, let q(Ht+1:∞ | Zt−1, H0:t−1) denote the predictive distribution before updates on targets consistent with the learner's expectations, and let q(Ht+1:∞ | Zt, H0:t) denote the distribution after updating on targets consistent with the learner's expectations. Non-forgetting requires:
q(Ht+1:∞ | Zt−1, H0:t−1) = EYt,Xt,Zt
-q(Ht+1:∞ | Zt, H0:t), (7)
where Yt ∼ qf (· | Zt−1, Xt−1), Xt ∼ qe(· | H0:t−1, Yt), and Zt ∼ u(· | Zt−1, Xt, Yt).

In these updates, Xt is sampled from the same hybrid distribution qe introduced in §3.2. Updates are performed on these learner-consistent targets to ensure a separation between forgetting and backward 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

$$p(\theta\mid X_{1:t})=\int p(\theta\mid X_{1:t+1})p(X_{t+1}\mid X_{1:t})\,\mathrm{d}X_{t+1}.$$
Zp(θ | X1:t+1)p(Xt+1 | X1:t) dXt+1. (10)
transfer. In this formulation, the expectation in (7) is taken over both the stochasticity of the learner's target generation and the environmental inputs, yielding a description of predictive consistency. This notion extends to multiple updates. Non-forgetting requires that predictive distributions remain compatible with those induced after updates. Formally, predictive distributions must be recoverable by marginalising over all k-step interaction paths, yielding the *generalised consistency condition*.

Definition 4.5 (Consistency Condition). For k ≥ 1, a learner is k-step consistent *if and only if* q
∗k(Ht+k:∞ | Zt−1, H0:t−1) = EXt:t
′ ,Yt:t
′ ,Zt:t
′-q(Ht+k:∞ | Zt
′ , H0:t
′ ), (8)
where t
′ = t + k − 1, and for i = *t, . . . , t*′the expectation is taken over Xi ∼ qe(· | H0:i−1, Yi),
Yi ∼ qf (· | Zi−1, Xi−1), Zi ∼ u(· | Zi−1, Xi, Yi).

Definition 4.5 shows why replay is often essential. When the update u(Zt−1, Xt, Yt) depends on the history H0:t−1, then the consistency condition requires access to past data during updates. Replay mechanisms provide this access, offering a clear mathematical justification for their role (see §B.3). Propensity to forget. In practice, most learning algorithms do forget. To quantify *how much* the learner is likely to forget at any time, we introduce a concrete measure grounded in our formalism. When the consistency condition is violated, the learner's updated predictive distribution diverges from its initial predictive distribution. Measuring this divergence yields a natural notion of a learner's propensity to forget. This operationalises our conceptual definition, allowing us to validate the formalism by ensuring that the measure aligns with intuitive expectations about forgetting in practice. Definition 4.6 (Propensity to Forget). The k-step propensity to forget incurred at time t, using a suitable divergence measure D(·∥·), is given by Γk(t) := D q(Ht+k:∞ | Zt−1, H0:t−1) ∥ q
∗
k(Ht+k:∞ | Zt−1, H0:t−1). (9)
Takeaway 1. Forgetting occurs when the consistency condition is violated; the predictive distribution after k *updates is no longer recoverable from those before the updates.*
Scope and boundary of validity. Our formalism applies whenever the learner's predictive distribution accurately represents the learner's state, Zt 7→ q(Ht+1:∞ | Zt, H0:t). Only information used to generate predictions contributes; state components that do not influence predictions (e.g., unused buffer entries) are excluded. Typically, the predictive distribution reflects the state, but this may not be the case during transitory phases such as buffer reinitialisation, target-network lag, or other mechanisms that temporarily decouple the state from predictions. In these intervals, forgetting is undefined, not because the formalism fails, but because the learner temporarily lacks a predictive model of its behaviour. Some algorithms may never produce a predictive mapping and thus fall outside the scope of this formalism. In most cases, however, the predictive distribution is representative of the state.

## 5 Empirical Analysis

Our theoretical account provides a general conceptualisation of forgetting. To illustrate its utility, we empirically study Definition 4.6 across multiple environments and learning algorithms. 5.1 UNFORGETFUL LEARNERS Exact Bayesian learners are unforgetful because they satisfy the k-step self-consistency condition: marginalising the posterior after a hypothetical future observation recovers the current posterior, Conditioning on a hypothetical future observation and then marginalising over its prior predictive distribution returns the same belief as not conditioning at all; conditioning and marginalising commute. In exchangeable settings, this self-consistency further implies permutation-invariance.

Let X1:t denote the observations up to time t, and θ the model parameters. The Bayesian posterior is

$$p(\theta\mid X_{1:t})\propto p(\theta)\prod_{i=1}^{n}p(X_{i}\mid\theta).$$
$\left(10\right)^2$
$$(11)$$
p(Xi| θ). (11)
7 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Takeaway 2. *Parameter changes alone do not imply forgetting.*

![7_image_0.png](7_image_0.png)

Figure 2: **Self-consistent learners do not forget.** Axes showing **observations** (*x, y*) are shaded white, and **parameter** axes (w0, w1) are shaded grey. *Top row:* The same four observations are presented to a linear regression learner in different orders. *Second row:* A full Bayesian posterior represents a complete summary of all observations and satisfies the k-step consistency condition, implying that no forgetting has occurred. The permutation invariance of the learner in exchangeable settings follows from the commutativity of Bayesian updates. *Bottom rows:* Two constrained learners - a Gaussian variational posterior with diagonal covariance and a point estimate updated by gradient descent - violate self-consistency. Their updates alter induced futures; consequently, previously supported capabilities may be lost. The final column shows the resulting posterior predictive distributions. Since multiplication commutes, the posterior is invariant to the order of observations:
p(θ | X1:t) = p(θ | Xπ(1)*, . . . , X*π(t)) ∀ permutations π. (12)
Therefore, exact Bayesian updates are *permutation-invariant* in exchangeable settings. Approximate learners, such as those using diagonal Gaussian variational inference or gradient-based point estimates, violate self-consistency. Their updates depend on the order of observations, and their beliefs cannot be recovered by marginalising over future predictions. Consequently, the learner's current predictive distribution may exclude previously supported predictions, which by Definition 4.5 constitutes forgetting. The contrast between exact and approximate learners is shown in Figure 2.

Across deep learning settings, forgetting is consistently non-zero, with both its magnitude and dynamics varying substantially (Figure 3). Even in i.i.d. settings, where forgetting is often overlooked, forgetting fluctuates throughout training, reflecting how neural networks continually update and overwrite their knowledge. Although absolute Γk(t) values are domain-specific, the forgetting trajectory reveals how the learner's beliefs evolve over time. These results show that forgetting is functionally meaningful in all tasks, highlighting the importance of a general conceptualisation of forgetting. This empirical observation motivates our paper title, "Forgetting is Everywhere".

## 5.2 Forgetting In Deep Learning

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

![8_image_0.png](8_image_0.png)

Figure 3: **Forgetting occurs across all deep learning scenarios.** *Left:* Forgetfulness dynamics of a shallow neural network trained on regression, classification, and generative modelling tasks. The solid line shows the k-step forgetfulness (where k varies from 1 to 40) over the normalised training step. Regression and classification tasks use KL divergence, while the generative task uses the maximum mean discrepancy (MMD). Forgetting dynamics vary throughout training, even without any distribution shift. *Right:* A class-incremental learning example using a single-layer neural network on a two-moons classification task. We show the k-step forgetting profile over four seeds, with the shaded area indicating the spread of Γk(t) over k from 1 to 40. The plot illustrates the abrupt increase in forgetting at the task boundary. See §F for details on the experimental implementation.

## 5.3 Approximate Learners Can Benefit From Forgetting

At each update, an approximation-based learner incorporates new information from current observations while discarding parts of its existing state (§4.2). Because approximate updates yield imperfect representations, a learner's performance depends on striking a balance between adapting to new information and retaining useful prior information. To study this effect, we investigate how modifications to the learner influence the propensity to forget.

Across experiments, a consistent pattern emerges for approximate learners: a moderate amount of forgetting improves learning efficiency. Here, we quantify training efficiency using the inverse of the normalised area under the training loss curve, a practical proxy for learning speed and convergence quality. Empirically, the forgetting-efficiency relationship shows an "elbow" (Figure 4), indicating that optimal training efficiency occurs at a non-zero level of forgetting. This suggests that effective approximate learners utilise forgetting as a mechanism for adaptive and efficient learning.

Takeaway 3. *Forgetting is ubiquitous in deep learning. The trade-off between training efficiency* and forgetting determines the optimal amount to forget–in deep learning, this is rarely zero.

![8_image_1.png](8_image_1.png)

Figure 4: **Approximate learners can benefit from non-zero forgetting.** We analyse how training efficiency and forgetfulness co-vary across learning algorithms. Training efficiency is measured as the inverse of the normalised area under the training loss curve, an approximate but informative proxy for learning speed and convergence quality in the setting we study. Forgetfulness is quantified as the mean 40-step propensity to forget, Γ40(t), across training steps. *Left:* In a regression task, varying stochastic gradient descent's momentum parameter shows that higher momentum increases forgetfulness, with maximum training efficiency at 0.9 momentum. *Right:* Varying model size shows that maximum efficiency occurs at 20 parameters. In both cases, the most efficient learners exhibit some forgetting– too little slows adaptation, too much destabilises learning–highlighting a fundamental trade-off.

![9_image_0.png](9_image_0.png)

## 5.4 Implications Of Distribution Shift

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Distribution shift and stochasticity strongly influence forgetting dynamics. In i.i.d. environments, the interaction process is stationary and the predictive distributions are stable. When learner hyperparameters are well-tuned and training conditions are stable, the learner effectively balances adaptation and retention. In such settings, training efficiency is high because the learner's updates operate under consistent conditions, and the underlying data distribution provides consistent feedback for improvement.

In CL, abrupt shifts in the observation distribution cause discontinuous changes in state, Zt, and the predictive distribution. Consequently, the magnitude of consistency violation abruptly increases at task boundaries (Figure 3) as the learner must rapidly adapt to a new task. All learning involves balancing the integration of new information with the retention of current knowledge. RL presents this challenge in an extreme form. Here, the learner's policy influences future observations, inducing continual non-stationarity. In DQN, for example, as the agent experiences new transitions, the TD loss rises because the agent incorporates new information (Figure 5). As the agent consolidates this information, the TD loss declines and the rate of information acquisition plateaus. The forgetting curve follows the TD loss because forgetting information is the mechanism by which the agent manages this process, demonstrating that forgetting is an essential component of RL.

Takeaway 4. Forgetting is an integral component of learning: effective learning requires selectively forgetting outdated knowledge to integrate new information.

## 6 Conclusion

In this work, we introduced a general, algorithm- and task-agnostic formulation of forgetting, describing it as the *temporal inconsistency* of a learner's predictive distribution. To our knowledge, this is the first generalised definition of forgetting. Unlike previous definitions, our approach encompasses generalisation forgetting, disentangles forgetting from backward transfer, and separates forgetting from parameter updates. This shows that learners *can adapt without forgetting* (§5.1). We also introduced the *propensity to forget* as an operational measure, allowing empirical validation of our definition.

Our empirical analysis across diverse learning algorithms and task settings shows that forgetting is pervasive in deep learning and shaped by interactions between the learner and the environment.

Interestingly, optimal training efficiency does not always correspond to minimal forgetting; in some cases, an *intermediate amount of forgetting maximises efficiency*, highlighting the importance of considering forgetting when designing learning algorithms. Overall, our work reframes forgetting as a *fundamental property of learning dynamics* rather than a failure mode limited to continual or non-stationary regimes. We hope our work provides a clear conceptual basis for understanding how a learner's capabilities emerge, persist, and deteriorate, guiding the design of algorithms that can adapt while retaining previously acquired knowledge.

## References

David Abel, Andre Barreto, Benjamin Van Roy, Doina Precup, Hado P van Hasselt, and Satin- ´
der Singh. A definition of continual reinforcement learning. *Advances in Neural Information* Processing Systems, 36:50377–50407, 2023.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Sina Alemohammad, Josue Casco-Rodriguez, Lorenzo Luzi, Ahmed Imtiaz Humayun, Hossein Babaei, Daniel LeJeune, Ali Siahkoohi, and Richard Baraniuk. Self-consuming generative models go mad. In *The Twelfth International Conference on Learning Representations*, 2023.

Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, and Tinne Tuytelaars.

Memory aware synapses: Learning what (not) to forget. In *Proceedings of the European conference* on computer vision (ECCV), pp. 139–154, 2018.

Daniel J Amit, Hanoch Gutfreund, and Haim Sompolinsky. Spin-glass models of neural networks.

Physical Review A, 32(2):1007, 1985.

Craig Atkinson, Brendan McCane, Lech Szymanski, and Anthony Robins. Pseudo-rehearsal: Achieving deep reinforcement learning without catastrophic forgetting. *Neurocomputing*, 428:291–307, 2021.

Diana Benavides-Prado and Patricia Riddle. A theory for knowledge transfer in continual learning.

In *Conference on Lifelong Learning Agents*, pp. 647–660. PMLR, 2022.

Christopher M Bishop. *Pattern recognition and machine learning*, volume 4. Springer, 2006. Shi Dong, Benjamin Van Roy, and Zhengyuan Zhou. Simple agent, complex environment: Efficient reinforcement learning with agent states. *Journal of Machine Learning Research*, 23(255):1–54, 2022.

William Fedus, Prajit Ramachandran, Rishabh Agarwal, Yoshua Bengio, Hugo Larochelle, Mark Rowland, and Will Dabney. Revisiting fundamentals of experience replay. In International conference on machine learning, pp. 3061–3071. PMLR, 2020.

Edwin Fong, Chris Holmes, and Stephen G Walker. Martingale posterior distributions. Journal of the Royal Statistical Society Series B: Statistical Methodology, 85(5):1357–1391, 2023.

Sandra Fortini and Sonia Petrone. Quasi-bayes properties of a recursive procedure for mixtures.

arXiv preprint arXiv:1902.10708, 2019.

Sandra Fortini and Sonia Petrone. Exchangeability, prediction and predictive modeling in bayesian statistics. *Statistical Science*, 40(1):40–67, 2025.

Robert M French. Catastrophic forgetting in connectionist networks. *Trends in cognitive sciences*, 3
(4):128–135, 1999.

Ian Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua Bengio. *Deep learning*, volume 1.

MIT Press, 2016.

Arslan Chaudhry, Puneet K Dokania, Thalaiyasingam Ajanthan, and Philip HS Torr. Riemannian walk for incremental learning: Understanding forgetting and intransigence. In *Proceedings of the* European conference on computer vision (ECCV), pp. 532–547, 2018a.

Arslan Chaudhry, Marc'Aurelio Ranzato, Marcus Rohrbach, and Mohamed Elhoseiny. Efficient lifelong learning with a-gem. *arXiv preprint arXiv:1812.00420*, 2018b.

Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Ales Leonardis, Gregory ˇ
Slabaugh, and Tinne Tuytelaars. A continual learning survey: Defying forgetting in classification tasks. *IEEE transactions on pattern analysis and machine intelligence*, 44(7):3366–3385, 2021.

Quentin Bertrand, Avishek Joey Bose, Alexandre Duplessis, Marco Jiralerspong, and Gauthier Gidel.

On the stability of iterative retraining of generative models on their own data. *arXiv preprint* arXiv:2310.00429, 2023.

J. J. Hopfield. Neural networks and physical systems with emergent collective computational abilities.

Proceedings of the National Academy of Sciences of the United States of America: Biological Sciences, 79(8):2554–2558, 1982.

Marcus Hutter, David Quarel, and Elliot Catt. *An introduction to universal artificial intelligence*.

Chapman and Hall/CRC, 2024.

Marcus Hutter. Universal artificial intelligence: Sequential decisions based on algorithmic probability. Springer Science & Business Media, 2005.

Matthew Jagielski, Om Thakkar, Florian Tramer, Daphne Ippolito, Katherine Lee, Nicholas Carlini, Eric Wallace, Shuang Song, Abhradeep Thakurta, Nicolas Papernot, et al. Measuring forgetting of memorized training examples. *arXiv preprint arXiv:2207.00099*, 2022.

Ronald Kemker, Marc McClure, Angelina Abitino, Tyler Hayes, and Christopher Kanan. Measuring catastrophic forgetting in neural networks. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.

Khimya Khetarpal, Matthew Riemer, Irina Rish, and Doina Precup. Towards continual reinforcement learning: A review and perspectives. *Journal of Artificial Intelligence Research*, 75:1401–1476, 2022.

Joonkyu Kim, Yejin Kim, and Jy-yong Sohn. Understanding the behavior of representation forgetting in continual learning. *arXiv preprint arXiv:2505.20970*, 2025.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Saurabh Kumar, Henrik Marklund, Ashish Rao, Yifan Zhu, Hong Jun Jeon, Yueyang Liu, Benjamin Van Roy, et al. Continual learning as computationally constrained reinforcement learning.

Foundations and Trends in Machine Learning, 18(5):913–1053, 2025.

Tor Lattimore. *Theory of general reinforcement learning*. PhD thesis, Australian National University, 2014.

Sebastian Lee, Sebastian Goldt, and Andrew Saxe. Continual learning in the teacher-student setup:
Impact of task similarity. In *International Conference on Machine Learning*, pp. 6109–6119. PMLR, 2021.

Thomas L Lee and Amos Storkey. Chunking: Forgetting matters in continual learning even without changing tasks. *arXiv preprint arXiv:2310.02206*, 2023.

James Kirkpatrick, Razvan Pascanu, Neil C. Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114:3521 - 3526, 2016.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A
Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. *Proceedings of the national academy of sciences*, 114 (13):3521–3526, 2017.

Qinglang Li, Jing Yang, Xiaoli Ruan, Shaobo Li, Jianjun Hu, and Bingqi Hu. Spirf-cta: Selection of parameter importance levels for reasonable forgetting in continuous task adaptation. Knowledge- Based Systems, 305:112575, 2024.

Nicolas Y Masse, Gregory D Grant, and David J Freedman. Alleviating catastrophic forgetting using context-dependent gating and synaptic stabilization. Proceedings of the National Academy of Sciences, 115(44):E10467–E10475, 2018.

Michael McCloskey and Neal J Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. In *Psychology of learning and motivation*, volume 24, pp. 109–165. Elsevier, 1989.
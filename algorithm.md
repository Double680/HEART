# GRPO Training Algorithm for Query Augmentation

This section describes the training procedure used to optimize the query augmentor. Given a question, the augmentor generates a rewritten retrieval query. The generated query is evaluated by an external evidence retriever, and the resulting retrieval quality is used as the reward signal for GRPO training.

## Notation

Let each training example be $x = (q, D_{\text{text}}, D_{\text{table}}, E_{\text{text}}, E_{\text{table}})$, where $q$ is the original question, $D_{\text{text}}$ is the list of textual retrieval units, $D_{\text{table}}$ is the list of table-cell descriptions, and $E_{\text{text}}$, $E_{\text{table}}$ are the annotated text and table evidence indices. Either evidence set may be empty.

The query augmentor is a causal language model $\pi_\theta$. For each prompt, it generates $G$ candidate outputs. The generated output is parsed into an augmented query $z$. If the output contains `<query>...</query>`, the inner text is used; otherwise, the raw generated text is used as a fallback query.

## Evidence Rewards

For each enabled evidence type $t \in \{\text{text}, \text{table}\}$, the algorithm computes a retrieval reward independently. If the gold evidence set $E_t$ is empty, the reward for that evidence type is undefined and is returned as `None`; this masks the sample from that reward function rather than treating it as a failed retrieval.

For non-empty evidence, two reward variants are supported.

Hard reward:

$$
\mathcal{I}_{t,K}(z)
= \operatorname{TopK}\left(\{s_j(z)\}_{j=1}^{|D_t|}, K\right)
$$

$$
r_t^{\mathrm{hard}}(z)
=
\frac{
  |\mathcal{I}_{t,K}(z) \cap E_t|
}{
  |E_t|
}
$$

Here, $s_j(z)$ is the retriever score between the generated query $z$ and the $j$-th document in $D_t$. $\mathcal{I}_{t,K}(z)$ denotes the indices of the top-$K$ retrieved documents.

Soft reward:

$$
r_t^{\mathrm{soft}}(z)
=
\frac{
  \sum_{j \in E_t} s_j(z)
}{
  |E_t|
}
$$

The soft reward is the average retriever score assigned to the gold evidence units. The implementation min-max normalizes retriever scores before computing this reward, so dense, BM25, and hybrid retrievers share the same soft-reward scale.

The final evidence reward for type $t$ is therefore

$$
r_t(z) =
\begin{cases}
r_t^{\mathrm{hard}}(z), & \text{if hard reward is selected}, \\
r_t^{\mathrm{soft}}(z), & \text{if soft reward is selected}.
\end{cases}
$$

In the implementation, text evidence defaults to hard reward and table evidence defaults to soft reward. The default retrieval depth is $K = 10$. No additional reward scaling is applied.

## Reward Aggregation

The implementation supports two aggregation modes. Let $\mathcal{T}$ denote the set of enabled evidence types, e.g., text and/or table. For a prompt, the query augmentor samples $G$ completions and obtains generated queries $\{z_i\}_{i=1}^{G}$. For each evidence type $t$, the reward $r_t(z_i)$ may be undefined when the corresponding gold evidence set is empty; such entries are masked.

### Sum-Then-Advantage

This is the standard GRPO-style aggregation. For each generated query $z_i$, enabled evidence rewards are first combined into a single scalar reward:

$$
r(z_i)
=
\sum_{t \in \mathcal{T}_i} w_t r_t(z_i)
$$

where $\mathcal{T}_i \subseteq \mathcal{T}$ contains the evidence types whose rewards are defined for $z_i$. GRPO then computes the group-level reward mean and standard deviation over the $G$ generations:

$$
\mu
=
\frac{1}{G}
\sum_{i=1}^{G} r(z_i)
$$

$$
\sigma
=
\sqrt{
  \frac{1}{G}
  \sum_{i=1}^{G}
  \left(r(z_i) - \mu\right)^2
}
$$

The final advantage used by GRPO is

$$
A(z_i)
=
\frac{r(z_i) - \mu}{\sigma}
$$

If $\sigma = 0$, the normalized advantage is set to $0$ by the implementation. This mode lets different evidence rewards interact before normalization: a high text reward can compensate for a low table reward before the group advantage is computed.

### Advantage-Then-Sum

This mode first normalizes each evidence reward separately within the generation group. For each evidence type $t$, let $V_t$ be the subset of generations whose reward $r_t(z_i)$ is defined:

$$
V_t
=
\{i \mid r_t(z_i) \text{ is defined}\}
$$

The per-type reward mean and standard deviation are

$$
\mu_t
=
\frac{1}{|V_t|}
\sum_{j \in V_t} r_t(z_j)
$$

$$
\sigma_t
=
\sqrt{
  \frac{1}{|V_t|}
  \sum_{j \in V_t}
  \left(r_t(z_j) - \mu_t\right)^2
}
$$

The per-type advantage is then

$$
A_t(z_i)
=
\frac{
  r_t(z_i) - \mu_t
}{
  \sigma_t
},
\quad i \in V_t
$$

If $\sigma_t = 0$, the per-type advantage is set to $0$. If $i \notin V_t$, then $A_t(z_i)$ is undefined and masked from the corresponding reward function.

The final value passed to GRPO is the weighted sum of the defined per-type advantages:

$$
A(z_i)
=
\sum_{t \in \mathcal{T}_i} w_t A_t(z_i)
$$

In this mode, the trainer-side reward normalization is disabled to avoid normalizing the already normalized per-type advantages again. Compared with sum-then-advantage, this mode prevents one evidence type from dominating the group normalization of another evidence type and naturally supports samples with missing text or table evidence.

## Algorithm

```text
Input:
  Training set S
  Query augmentor pi_theta
  Evidence retriever R
  Prompt template T
  Enabled evidence reward types C_text, C_table
  Number of generations G
  Retrieval depth K
  Reward weights w_text, w_table

For each training example x in S:
  1. Build the user prompt by inserting the original question q into T.
  2. Append the no-thinking suffix to suppress explicit reasoning.
  3. Sample G completions from pi_theta.
  4. For each completion y_i:
       a. Extract the augmented query z_i from y_i.
       b. For each enabled evidence type t:
            i.   If E_t is empty, return None for this reward.
            ii.  If C_t is hard, retrieve top-K documents and compute recall.
            iii. If C_t is soft, compute the retriever's soft relevance score.
  5. Aggregate rewards using either:
       a. sum-then-advantage, or
       b. advantage-then-sum.
  6. Update pi_theta with the GRPO objective.

Output:
  Trained query augmentor pi_theta
```

## Implementation Notes

- Text evidence uses `paragraphs` as retrieval documents and `qa.text_evidence` as gold indices.
- Table evidence uses `table_description` as retrieval documents and `qa.table_evidence_id` as gold indices.
- Samples with empty evidence for one modality can still contribute rewards for the other modality.
- The reward evaluator can use dense, BM25, or hybrid retrieval.
- During multi-GPU training, the reward retriever is placed on the local device of each process when `retriever_device=auto`.

# Cora-Node-Classification 

This repository contains a compact pipeline that assigns one of **seven Cora subject classes**  
(`Case_Based`, `Genetic_Algorithms`, `Neural_Networks`, `Probabilistic_Methods`, `Reinforcement_Learning`, `Rule_Learning`, `Theory`) to every node in the Cora citation graph.

Four successively stronger models are available, allowing the user to trade runtime for accuracy
with a single command-line flag.

---

## 1 · Rationale behind the chosen models

### Model rationale & comparative notes

| Model               | Why it is included                                                                                                                      | Strengths on Cora                                                                                                                            | Speed class   | Main limitations                                                                               |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ---------------------------------------------------------------------------------------------- |
| **LogReg + TF-IDF** | Serves as a pure‐text baseline that sets the lower bound for graph-aware gains; popular reference in graph-ML literature.               | • Captures token–class correlations efficiently.<br>• Weights are directly interpretable.                                                    | **Very fast** | Ignores citation structure and non-linear feature interactions.                                |
| **2-layer GCN**     | Minimal graph neural network: two convolutions diffuse features across the 2-hop citation neighbourhood.                                | • Exploits “papers cite similar papers” signal.<br>• Few parameters → low risk of over-fitting on a small graph.                             | **Fast**      | Over-smooths if stacked deeper; limited to short-range context.                                |
| **APPNP**           | Decouples feature learning (MLP) from a fixed 10-step personalised-PageRank propagation.                                                | • Provides mid-range (≈10-hop) context without adding trainable weights per hop.<br>• Less prone to over-smoothing than deeper vanilla GCNs. | **Medium**    | Propagation depth is fixed; may under-perform if the optimal hop count differs across classes. |
| **GCN-II**          | Adds identity mapping and residual connections, enabling very deep message passing (32 layers here) while preserving original features. | • Looks dozens of hops away yet keeps local details.<br>• Regularised with DropEdge to curb over-fitting.                                    | **Slower**    | More training epochs required; accuracy gains taper off beyond \~32 layers.                    |

These four checkpoints form a deliberate accuracy ladder:

1. **Text-only benchmark** (LogReg) → establishes a reference without graph cost.
2. **Local graph context** (2-layer GCN) → quick accuracy boost via 2-hop information.
3. **Mid-range diffusion** (APPNP) → balances depth and stability, typically the best trade-off.
4. **Deep residual network** (GCN-II) → highest accuracy achievable without resorting to heavyweight attention or ensemble methods.


Comparison with alternative families:

* Tree ensembles and linear SVMs were discarded due to inferior graph‐agnostic performance.  
* Transformer-style language models are ill-suited because Cora tokens lack word order.  
* Graph Attention Networks provide marginal gains but require materially longer CPU training times.

---

## 2 · Installation and single-command execution

```bash
git clone <repository-url>
cd cora-node-cls
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt            
# copy cora.content and cora.cites into ./cora/
python train_modular.py                    # choose 80 / 85 / 87 / 89 at the prompt
predictions.tsv is produced in the required
<paper_id> <class_label> format, and overall 10-fold accuracy is displayed.

A CUDA device is auto-detected and reduces times by a factor of 3–4.

The raw Cora data are not committed; place cora.content and cora.cites inside ./cora/.

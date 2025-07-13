# Official Code for "Investigating Mechanisms for In-Context Vision Language Binding"
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2505.22200)


Selected for Oral presentation at Mechanistic Interpretability for Vision Workshop at CVPR 2025

This codebase is built based on the repository for [Binding ID Mechanism](https://github.com/jiahai-feng/binding-iclr), proposed in ICLR 2024.

## Environment set-up

```
pip install -r requirements.txt

```

## Experiment Details

- This code works with `python 3.10.14`

- All experiments use the [llava-onevision-qwen2-7b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf) model from HuggingFace.

- The TransformerLens Library was used for all the experiments. Since it only supports LLMs, the Qwen2-7B LLM from the HuggingFace model was loaded into a HookedTransformer version of Qwen2-7B.

## Results

Use the following commands to reproduce the results from the paper.

- Factorizability

```
python factorizability.py --cache_dir '/hf/hub'
```

- Position Independence
  
```
python pos_ind.py --cache_dir '/hf/hub'
```

- Mean Interventions
  
    - Compute difference of binding vectors
    ```
    python mean_est.py --cache_dir '/hf/hub'
    ```

    - Intervene with the computed vectors
    ```
    python mean_ab.py --cache_dir '/hf/hub'
    ```

    - Intervention with random vectors
    ```
    python mean_ab.py --cache_dir '/hf/hub' --random_means
    ```


Plot the results using `plots.ipynb`
import os
import random
import subprocess
import numpy as np
import scanpy as sc
import anndata as ad


def fix_seed(seed=42, verbose=True):
    if verbose:
        print(f'Global random seed set to: {seed}.')

    random.seed(seed)
    np.random.seed(seed)
    sc.settings.seed = seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_multigpu_cluster(min_free_ratio=0.5, verbose=True):
    try:
        smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )

        free_memories, total_memories = [], []
        for line in smi_output.strip().split('\n'):
            free, total = line.split(',')
            free_memories.append(int(free.strip()))
            total_memories.append(int(total.strip()))

        available_gpus = [
            str(i) for i, (free, total) in enumerate(zip(free_memories, total_memories)) 
            if (free / total) >= min_free_ratio
        ]
        if not available_gpus:
            best_gpu = str(free_memories.index(max(free_memories)))
            available_gpus = [best_gpu]

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(available_gpus)

        if verbose:
            gpu_str_print = ', '.join(available_gpus)
            print(f'Multi-GPU cluster configured. JAX will use devices: cuda:{gpu_str_print}.')

    except Exception as e:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        if verbose:
            print(f'nvidia-smi check failed: {e}. Falling back to default cuda:0.')


def optimize_leiden(adata, n_clusters=10, used_obsm='emb', add_obs='leiden',
                    res_min=1e-3, res_max=2.0, max_tries=20, random_state=42, verbose=True):
    feat_mat = adata.obsm[used_obsm]
    all_same = np.allclose(feat_mat, feat_mat[[0], :])
    if all_same:
        adata.obs[add_obs] = np.zeros(adata.n_obs, dtype=int)
        adata.obs[add_obs] = adata.obs[add_obs].astype('category')
        if verbose:
            print(f"Warning: All rows in obsm['{used_obsm}'] are identical, assign all cluster label 0.")
        return

    best_labels = None
    tmp_adata = ad.AnnData(np.zeros((adata.shape[0], 1)))
    tmp_adata.obsm['X_feat'] = adata.obsm[used_obsm]
    sc.pp.neighbors(tmp_adata, use_rep='X_feat', n_neighbors=20, random_state=random_state)

    for i in range(max_tries):
        res = (res_min + res_max) / 2
        sc.tl.leiden(tmp_adata, resolution=res, random_state=random_state, 
                     directed=False, n_iterations=2, flavor='igraph')
        current_clusters = len(tmp_adata.obs['leiden'].cat.categories)

        if verbose:
            print(f'Current resolution is {res:.4f}, found {current_clusters} clusters.')

        if current_clusters == n_clusters:
            best_labels = tmp_adata.obs['leiden'].values
            if verbose:
                print(f'Success: Leiden found {n_clusters} clusters at resolution {res:.4f} (Attempt {i+1}).')
            break
        elif current_clusters < n_clusters:
            res_min = res
        else:
            res_max = res

    if best_labels is None:
        best_labels = tmp_adata.obs['leiden'].values
        if verbose:
            print(f'Warning: Leiden failed to find exactly {n_clusters} clusters '
                  f'(found {current_clusters} clusters at resolution {res:.4f}).')

    adata.obs[add_obs] = np.array(best_labels).astype(int)
    adata.obs[add_obs] = adata.obs[add_obs].astype('category')
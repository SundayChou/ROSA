import mygene
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from goatools.obo_parser import GODag
from sklearn.neighbors import NearestNeighbors


def read_and_preprocess_data(dataset_name, all_slice_names, target_slice_name, min_cells=50,
                             min_counts=10, n_top_genes=2000, target_sum=1e4, verbose=True):
    gene_map, adata_list = None, []
    for s in all_slice_names:
        tmp_adata = sc.read_h5ad(f'../data/{dataset_name}/{s}.h5ad')
        sc.pp.filter_genes(tmp_adata, min_cells=min_cells)
        sc.pp.filter_genes(tmp_adata, min_counts=min_counts)
        adata_list.append(tmp_adata)
        if gene_map is None:
            gene_map = dict(zip(tmp_adata.var_names, tmp_adata.var['gene_symbol']))

    adata = sc.concat(adata_list, join='inner', label='time_point',
                      keys=all_slice_names, index_unique='-', merge='same')

    aligned_coords = np.zeros_like(adata.obsm['spatial'])
    scaled_coords = np.zeros_like(adata.obsm['spatial'])
    for s in all_slice_names:
        mask = adata.obs['time_point'] == s
        coords = adata.obsm['spatial'][mask]
        center_coords = coords - coords.mean(axis=0)
        aligned_coords[mask] = center_coords
        mu = center_coords.mean()
        sigma = center_coords.std()
        if sigma < 1e-8:
            sigma = 1.0
        scaled_coords[mask] = (center_coords - mu) / sigma
    adata.obsm['spatial'] = aligned_coords
    adata.obsm['scaled_spatial'] = scaled_coords

    adata_source = adata[adata.obs['time_point'] != target_slice_name].copy()
    sc.pp.highly_variable_genes(adata_source, flavor='seurat_v3', n_top_genes=n_top_genes)
    adata = adata[:, adata_source.var['highly_variable']].copy()
    adata.var['gene_symbol'] = adata.var_names.map(gene_map)

    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    adata.uns['all_slice_names'] = all_slice_names
    adata.uns['target_slice_name'] = target_slice_name
    adata.uns['source_slice_names'] = [s for s in all_slice_names if s != target_slice_name]

    if verbose:
        print('Source and target slices have been preprocessed.')

    return adata


def compute_goef_weights(adata, species='human', obo_path='../data/go-basic.obo'):
    hvg_indices = adata.var_names.tolist()
    hvg_symbols = adata.var['gene_symbol'].tolist()
    N = len(hvg_indices)

    if species == 'human':
        hvg_symbols = [str(sym).upper() for sym in hvg_symbols]
    elif species == 'mouse':
        hvg_symbols = [str(sym).capitalize() for sym in hvg_symbols]

    symbol2indices = {}
    for idx_pos, sym in enumerate(hvg_symbols):
        symbol2indices.setdefault(sym, []).append(idx_pos)

    mg = mygene.MyGeneInfo()
    queries = mg.querymany(list(symbol2indices.keys()), scopes=['symbol'],
                           fields=['go'], species=species, returnall=False, verbose=False)

    index2go = {}
    for q in queries:
        sym = q.get('query')
        gos = set()
        for d in ['BP', 'MF', 'CC']:
            terms = q.get('go', {}).get(d, [])
            terms = [terms] if isinstance(terms, dict) else terms
            gos.update(t['id'] for t in terms)
        if gos and sym in symbol2indices:
            for idx_pos in symbol2indices[sym]:
                index2go[idx_pos] = gos

    unique_gos = set()
    godag = GODag(obo_path, prt=None)
    for idx_pos, terms in index2go.items():
        valid_terms = {t for t in terms if t in godag}
        index2go[idx_pos] = valid_terms
        unique_gos.update(valid_terms)

    unique_gos_list = sorted(list(unique_gos))
    K = len(unique_gos_list)
    go2idx = {go: i for i, go in enumerate(unique_gos_list)}

    rows, cols = [], []
    for idx_pos, terms in index2go.items():
        for go in terms:
            rows.append(idx_pos)
            cols.append(go2idx[go])

    d_go = np.array([godag[go].depth for go in unique_gos_list])
    if len(rows) > 0:
        T = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, K))
        c_gene = T.dot(d_go)
    else:
        c_gene = np.zeros(N)

    w_goef = np.log1p(c_gene)
    max_val = np.max(w_goef)
    if max_val > 1e-8:
        w_goef = w_goef / max_val

    return w_goef


def compute_single_slice_dlmmd(coords, X_slice, w_goef, n_neighbors, cv_scale_factor, trunc_ratio, alpha_go):
    N = coords.shape[0]
    actual_k = min(n_neighbors, N - 1)

    nbrs = NearestNeighbors(n_neighbors=actual_k + 1, algorithm='kd_tree', n_jobs=-1).fit(coords)
    knn_dists, _ = nbrs.kneighbors(coords)
    knn_dists = knn_dists[:, 1:]

    mean_knn_dists = np.mean(knn_dists, axis=1)

    dist_std = np.std(mean_knn_dists)
    dist_mean = np.mean(mean_knn_dists) + 1e-8
    cv = dist_std / dist_mean
    w_dense = np.tanh(cv_scale_factor * cv)

    sigma = np.median(mean_knn_dists)
    if sigma < 1e-8:
        sigma = 1.0
    raw_density = np.exp(- (mean_knn_dists ** 2) / (2 * (sigma ** 2)))
    max_d = np.max(raw_density)
    raw_density = np.maximum(raw_density, trunc_ratio * max_d)

    p_uniform = np.ones(N) / N
    p_density = raw_density / (raw_density.sum() + 1e-8)
    p_samp = (1 - w_dense) * p_uniform + w_dense * p_density

    if alpha_go > 0.0 and w_goef is not None and X_slice is not None and np.any(w_goef > 0):
        score_goef = X_slice.log1p().dot(w_goef)
        max_val = np.max(score_goef)

        truncated_score_goef = np.maximum(score_goef, max_val * trunc_ratio)
        norm_score_goef = truncated_score_goef / (np.sum(truncated_score_goef) + 1e-8)
        factor_goef = np.power(norm_score_goef, alpha_go)

        joint_mass = p_samp * factor_goef
        dlmmd_vector = joint_mass / (np.sum(joint_mass) + 1e-8)
    else:
        dlmmd_vector = p_samp / (np.sum(p_samp) + 1e-8)

    return dlmmd_vector


def compute_dlmmd(adata, species='human', obo_path='../data/go-basic.obo',
                  n_neighbors=30, cv_scale_factor=5.0, trunc_ratio=0.5, alpha_go=1.5, verbose=True):
    target_slice_name = adata.uns['target_slice_name']
    source_slice_names = adata.uns['source_slice_names']
    counts_mat = adata.layers['counts']

    if alpha_go > 0.0:
        w_goef = compute_goef_weights(adata, species=species, obo_path=obo_path)
    else:
        w_goef = None

    dlmmd_array = np.zeros(adata.n_obs)
    mask_target = (adata.obs['time_point'] == target_slice_name).values
    coords_target = adata.obsm['spatial'][mask_target]
    X_target = counts_mat[mask_target]
    b_target = compute_single_slice_dlmmd(coords_target, X_target, w_goef, n_neighbors, 
                                           cv_scale_factor, trunc_ratio, alpha_go)
    dlmmd_array[mask_target] = b_target

    for s in source_slice_names:
        mask_source = (adata.obs['time_point'] == s).values
        coords_source = adata.obsm['spatial'][mask_source]
        X_source = counts_mat[mask_source]
        a_source = compute_single_slice_dlmmd(coords_source, X_source, w_goef, n_neighbors, 
                                               cv_scale_factor, trunc_ratio, alpha_go)
        dlmmd_array[mask_source] = a_source

    adata.obs['dlmmd'] = dlmmd_array

    if verbose:
        print('Dual-layer marginal mass distribution have been computed.')
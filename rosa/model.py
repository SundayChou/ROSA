import jax
import time
import numpy as np
import jax.numpy as jnp
import concurrent.futures

from .solver import solve_unbalanced_gw


class ROSA:
    def __init__(
        self,
        adata,
        epsilon=2e-2,
        tau_a=0.999,
        tau_b=0.999,
        max_gw_iterations=100,
        max_sinkhorn_iterations=1000,
        gw_threshold=1e-2,
        sinkhorn_threshold=1e-2,
        verbose=True,
    ):
        self.adata = adata
        self.epsilon = epsilon
        self.tau_a = tau_a
        self.tau_b = tau_b
        self.max_gw_iterations = max_gw_iterations
        self.max_sinkhorn_iterations = max_sinkhorn_iterations
        self.gw_threshold = gw_threshold
        self.sinkhorn_threshold = sinkhorn_threshold
        self.verbose = verbose

        self.target_slice_name = self.adata.uns['target_slice_name']
        self.source_slice_names = self.adata.uns['source_slice_names']
        self.all_slice_names = self.adata.uns['all_slice_names']
        self.has_dlmmd = 'dlmmd' in self.adata.obs
        self.pi_dict = {}
        self.is_solved = False

        mask_target = (self.adata.obs['time_point'] == self.target_slice_name).values
        self.coords_target = jnp.array(self.adata.obsm['scaled_spatial'][mask_target])
        self.b_target = jnp.array(self.adata.obs['dlmmd'].values[mask_target]) if self.has_dlmmd else None

        self.coords_source_dict = {}
        self.a_source_dict = {}
        for s in self.source_slice_names:
            mask_source = (self.adata.obs['time_point'] == s).values
            self.coords_source_dict[s] = jnp.array(self.adata.obsm['scaled_spatial'][mask_source])
            self.a_source_dict[s] = jnp.array(self.adata.obs['dlmmd'].values[mask_source]) if self.has_dlmmd else None

        if self.verbose:
            print('The OT problems are ready for solving.')

    def solve(self):
        start_time = time.time()
        devices = jax.local_devices()

        def _solve_single_slice(s, coords_source, a_source, device):
            if self.verbose:
                print(f'Computing the OT plan for {s} ({coords_source.shape[0]} spots) ' \
                      f'-> {self.target_slice_name} ({self.coords_target.shape[0]} spots)...')

            source_on_device = jax.device_put(coords_source, device)
            target_on_device = jax.device_put(self.coords_target, device)
            a_on_device = jax.device_put(a_source, device) if a_source is not None else None
            b_on_device = jax.device_put(self.b_target, device) if self.b_target is not None else None

            pi = solve_unbalanced_gw(
                coords_source=source_on_device,
                coords_target=target_on_device,
                a=a_on_device,
                b=b_on_device,
                epsilon=self.epsilon,
                tau_a=self.tau_a,
                tau_b=self.tau_b,
                max_gw_iterations=self.max_gw_iterations,
                max_sinkhorn_iterations=self.max_sinkhorn_iterations,
                gw_threshold=self.gw_threshold,
                sinkhorn_threshold=self.sinkhorn_threshold
            )

            return s, pi.block_until_ready()

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = []
            for i, s in enumerate(self.source_slice_names):
                device = devices[i % len(devices)]
                coords_source = self.coords_source_dict[s]
                a_source = self.a_source_dict[s]
                futures.append(executor.submit(_solve_single_slice, s, coords_source, a_source, device))
            temp_pi_dict = {}
            for future in concurrent.futures.as_completed(futures):
                s, pi = future.result()
                temp_pi_dict[s] = pi

        for s in self.source_slice_names:
            self.pi_dict[s] = temp_pi_dict[s]

        self.total_time = (time.time() - start_time) / 60
        self.total_peak_memory = sum(dev.memory_stats()['peak_bytes_in_use'] for dev in devices) / (1024 ** 3)

        if self.verbose:
            print('All OT problems have been solved.')
            print(f'Total training time: {self.total_time:.4f} minutes.')
            print(f'Total peak GPU memory usage: {self.total_peak_memory:.4f} GB.')

        self.is_solved = True

    def impute(self):
        if not self.is_solved:
            raise ValueError('The OT problems are not solved yet! Call the `.solve()` method first.')

        if self.verbose:
            print('Executing barycentric projection for target slice imputation...')

        time_weights = []
        target_idx = self.all_slice_names.index(self.target_slice_name)
        for s in self.source_slice_names:
            source_idx = self.all_slice_names.index(s)
            distance = abs(source_idx - target_idx)
            time_weights.append(1.0 / distance)
        time_weights = np.array(time_weights)
        time_weights /= time_weights.sum()

        self.imputed_X = np.zeros((self.coords_target.shape[0], self.adata.shape[1]))

        def _impute_single_tp(i, s):
            pi = np.array(self.pi_dict[s]) 
            pi_T = pi.T 
            pi_T_norm = pi_T / (pi_T.sum(axis=1, keepdims=True) + 1e-8)

            mask = (self.adata.obs['time_point'] == s).values
            X_source = self.adata.X[mask]
            pred_from_source = pi_T_norm @ X_source
            partial_imputed_X = time_weights[i] * pred_from_source

            return partial_imputed_X

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.source_slice_names)) as executor:
            futures = []
            for i, s in enumerate(self.source_slice_names):
                futures.append(executor.submit(_impute_single_tp, i, s))

            for future in concurrent.futures.as_completed(futures):
                self.imputed_X += future.result()

        if self.verbose:
            print('The target slice has been imputed.')
import os
from absl import logging
import numpy as np

from jax_dft import scf

# pytype: disable=attribute-error

_TEST_DISTANCE_X100 = {
    'h2_plus': set([
        64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 248, 256,
        264, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464,
        480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688,
        704, 720, 736, 752, 768, 784, 800, 816, 832, 848]),
    'h2': set([
        40, 56, 72, 88, 104, 120, 136, 152, 184, 200, 216, 232,
        248, 264, 280, 312, 328, 344, 360, 376, 392, 408, 424,
        456, 472, 488, 504, 520, 536, 568, 584, 600]),
    'h4': set([
        104, 120, 136, 152, 168, 200, 216, 232, 248, 280, 296,
        312, 344, 360, 376, 392, 408, 424, 440, 456, 472, 488,
        520, 536, 552, 568, 584, 600]),
    'h2_h2': set([
        16, 48, 80, 112, 144, 176, 208, 240, 272, 304, 336, 368,
        400, 432, 464, 496, 528, 560, 592, 624, 656, 688, 720, 752,
        784, 816, 848, 880, 912, 944, 976]),
}


class Dataset(object):
    def __init__(self, path=None, data=None, num_grids=None, name=None):
        if data is None and path is None:
            raise ValueError("path and data cannot both be None.")
        self.name = name
        if data is None and path is not None:
            data = self._load_from_path(path)
        for name, array in data.items():
            setattr(self, name, array)
        self._set_num_grids(num_grids)
        self.total_num_samples = self.distances.shape[0]

    def _load_from_path(self, path):
        file_open = open
        data = {}
        with file_open(os.path.join(path, 'num_electrons.npy'), 'rb') as f:
            # make sure this is a scalar not an array
            data['num_electrons'] = int(np.load(f))
        with file_open(os.path.join(path, 'grids.npy'), 'rb') as f:
            data['grids'] = np.load(f)
        with file_open(os.path.join(path, 'locations.npy'), 'rb') as f:
            data['locations'] = np.load(f)
        with file_open(os.path.join(path, 'nuclear_charges.npy'), 'rb') as f:
            data['nuclear_charges'] = np.load(f)
        with file_open(os.path.join(path, 'distances_x100.npy'), 'rb') as f:
            data['distances_x100'] = np.load(f).astype(int)
        with file_open(os.path.join(path, 'distances.npy'), 'rb') as f:
            data['distances'] = np.load(f)
        with file_open(os.path.join(path, 'total_energies.npy'), 'rb') as f:
            data['total_energies'] = np.load(f)
        with file_open(os.path.join(path, 'densities.npy'), 'rb') as f:
            data['densities'] = np.load(f)
        with file_open(os.path.join(path, 'external_potentials.npy'), 'rb') as f:
            data['external_potentials'] = np.load(f)
        return data

    def _set_num_grids(self, num_grids):
        original_num_grids = getattr(self, 'num_grids', None)
        if original_num_grids is None:
            self.num_grids = num_grids
            logging.info(f"This dataset has {num_grids} grids")
        else:
            if num_grids > original_num_grids:
                raise ValueError(f"num_grids {num_grids} cannot be greater "
                                 f"than original number {original_num_grids}")
            self.num_grids = num_grids
            diff = original_num_grids - num_grids
            if (num_grids % 2) == 0:
                left_grids_removed = (diff - 1) // 2
                right_grids_removed = (diff + 1) // 2
            else:
                left_grids_removed = diff // 2
                right_grids_removed = diff // 2

            self.grids = self.grids[:, left_grids_removed: original_num_grids - right_grids_removed]
            self.densities = self.densities[:, left_grids_removed: original_num_grids-right_grids_removed]
            self.locations = self.external_potentials[
                             :, left_grids_removed: original_num_grids - right_grids_removed]
            logging.info(
                f"The original number of grids {original_num_grids} "
                f"are trimmed into {self.num_grids} grids"
            )

    def get_mask(self, selected_distance_x100):
        """Get mask for distance_x100"""
        if selected_distance_x100 is None:
            mask = np.ones(len(self.distances_x100))
        else:
            selected_distance_x100 = set(selected_distance_x100)
            mask = np.array([distance in selected_distance_x100
                    for distance in self.distances_x100])
            if len(selected_distance_x100) != np.sum(mask):
                raise ValueError("selected_distances_x100 contains distance "
                                 "that is not in the dataset")
        return mask

    def get_test_mask(self):
        """Gets mask for test set"""
        return self.get_mask(_TEST_DISTANCE_X100[self.name])

    def get_subdataset(self, selected_distance_x100=None, downsample_step=None):
        mask = self.get_mask(selected_distance_x100)
        if downsample_step is not None:
            sample_mask = np.zeros(self.total_num_samples, dtype=bool)
            sample_mask[::downsample_step] = True
            mask = np.logical_and(mask, sample_mask)
        return Dataset(
            data=dict(
                    num_electrons=self.num_electrons,
                    grids=self.grids,
                    locations=self.locations[mask],
                    nuclear_charges=self.nuclear_charges[mask],
                    distances_x100=self.distances_x100[mask],
                    distances=self.distances[mask],
                    total_energies=self.total_energies[mask],
                    densities=self.densities[mask],
                    external_potentials=self.external_potentials[mask]
            )
        )

    def get_molecules(self, selected_distance_x100=None):
        mask = self.get_mask(selected_distance_x100)
        num_samples = np.sum(mask)

        return scf.KohnShamState(
            density=self.densities[mask],
            total_energy=self.total_energies[mask],
            locations=self.locations[mask],
            nuclear_charges=self.nuclear_charges[mask],
            external_potentials=self.external_potentials[mask],
            grids=np.tile(np.expand_dims(self.grids, axis=0), reps=(num_samples, 1)),
            num_electrons=np.repeat(self.num_electrons, repeats=num_samples),
            converged=np.repeat(True, repeats=num_samples)
        )

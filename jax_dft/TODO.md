## `utils`
- [x] test_shift
- [x] test_get_dx
- [x] test_get_dx_incorrect_ndim
- [x] test_gaussian
- [x] test_soft_coulomb
- [x] test_exponential_coulomb
- [x] test_get_atomic_chain_potential_soft_coulomb
- [x] test_get_atomic_chain_potential_exponential_coulomb
- [x] test_get_atomic_chain_potential_incorrect_ndim
- [x] test_get_nuclear_interaction_energy
- [x] test_get_nuclear_interaction_energy_batch
- [x] test_get_nuclear_interaction_energy_incorrect_ndim
- [x] test_float_value_in_array_true
- [x] test_float_value_in_array_false
- [x] test_flip_and_average_the_front_of_array_center_on_grids
- [x] test_flip_and_average_the_back_of_array_center_on_grids
- [x] test_flip_and_average_the_front_of_array_center_not_on_grids
- [x] test_flip_and_average_the_back_of_array_center_not_on_grids
- [x] test_flip_and_average_location_not_on_grids
- [x] test_location_center_at_grids_center_point_true
- [x] test_location_center_at_grids_center_point_false
- [x] test_compute_distances_between_nuclei
- [x] test_compute_distances_between_nuclei_wrong_locations_ndim
- [x] test_compute_distances_between_nuclei_wrong_nuclei_indices_size

## `scf`
- [x] test_discrete_laplacian
- [x] test_get_kinetic_matrix
- [x] test_wavefunctions_to_density
- [x] test_get_total_eigen_energies
- [x] test_get_gap
- [x] test_solve_noninteracting_system
- [x] test_get_hartree_energy
- [x] test_get_hartree_potential
- [x] test_get_external_potential_energy
- [x] test_get_xc_energy
- [x] test_get_xc_potential
- [x] test_get_xc_potential_hartree
- [ ] test_save_and_load_state --- this function is not there
- [x] test_kohn_sham_iteration
- [x] test_kohn_sham_iteration_neural_xc
- [x] test_kohn_sham_iteration_neural_xc_energy_loss_gradient
- [x] test_kohn_sham_iteration_neural_xc_density_loss_gradient
- [x] test_kohn_sham_iteration_neural_xc_density_loss_gradient_symmetry
- [x] test_kohn_sham
- [x] test_kohn_sham_convergence
- [x] test_kohn_sham_neural_xc
- [x] test_kohn_sham_neural_xc_energy_loss_gradient
- [x] test_kohn_sham_neural_xc_density_loss_gradient
- [x] test_get_initial_density_exact
- [x] test_get_initial_density_noninteracting
- [x] test_get_initial_density_unknown

## `neural_xc`
- [x] test_negativity_transform
- [x] test_exponential_function_normalization
- [x] test_exponential_function_channels
- [x] test_exponential_global_convolution
- [x] test_self_interaction_weight
- [x] test_self_interaction_layer_one_electron
- [x] test_self_interaction_layer_large_num_electrons
- [x] test_wrap_network_with_self_interaction_layer_one_electron
- [x] test_wrap_network_with_self_interaction_layer_large_num_electrons
- [x] test_downsampling_block
- [x] test_linear_interpolation
- [x] test_linear_interpolation_transpose
- [x] test_upsampling_block
- [x] test_build_unet
- [x] test_build_sliding_net
- [x] test_build_sliding_net_invalid_window_size
- [x] test_local_density_approximation
- [x] test_local_density_approximation_wrong_output_shape
- [x] test_spatial_shift_input
- [x] test_reverse_spatial_shift_output
- [x] test_is_power_of_two_true
- [x] test_is_power_of_two_false
- [x] test_global_functional_with_unet
- [x] test_global_functional_with_sliding_net
- [x] test_global_functional_wrong_num_spatial_shift
- [x] test_global_functional_wrong_num_grids
- [x] test_global_functional_with_unet_wrong_output_shape
- [x] test_global_functional_with_sliding_net_wrong_output_shape

## `np_utils`
- [x] test_flatten
- [x] test_get_exact_h_atom_density
- [x] test_get_exact_h_atom_density_wrong_shape
- [x] test_spherical_superposition_density

## `jit_scf`
- [x] test_flip_and_average_on_centre
- [x] test_flip_and_average_on_centre_fn
- [x] test_connection_weights
- [x] test_kohn_sham_iteration_neural_xc
- [x] test_kohn_sham_neural_xc_density_mse_converge_tolerance
- [x] test_kohn_sham_neural_xc_num_mixing_iterations

## `losses`
- [x] test_trajectory_mse_wrong_predict_ndim
- [x] test_trajectory_mse_wrong_predict_target_ndim_difference
- [x] test_density_mse
- [x] test_energy_mse
- [x] test_get_discount_coefficients
- [x] test_trajectory_mse_on_density
- [x] test_trajectory_mse_on_energy


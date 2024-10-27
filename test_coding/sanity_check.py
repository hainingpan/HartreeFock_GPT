import numpy as np

def map_lattice_geom_to_valid_symmetries(model):
  if model.lattice == 'square':
    valid_sym_rot = ['C4', 'C2']
    valid_sym_mirror = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
  elif model.lattice == 'triangular':
    valid_sym_rot = ['C3', 'C6']
    valid_sym_mirror = [np.pi / 3, 2 * np.pi / 3]
  else:
    raise ValueError('Invalid symmetry: %s' % model.lattice)
  return valid_sym_rot, valid_sym_mirror
  
rot_mat = lambda theta: np.array(
    [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
)

# Rotational symmetry check
def rotate_2D(k_space, theta):
  """
  Args:
    k_space: k-mesh, np.ndarray(N,2)
    theta: rotate angle (counterclockwise)
  """
  assert k_space.shape[-1] == 2, (
      'Only 2D lattice is supported, the current dimension of lattice is'
      f' {k_space.shape[-1]}'
  )

  return k_space @ rot_mat(theta).T

def check_rotation(model, kind):
  """R_θ H(k) R_θ.T=H(R_θ(k))=H(k)"""
  theta = {'C' + str(n): np.pi * 2 / n for n in [2, 3, 4, 6]}
  assert kind in theta, f'Symmetry should be among f{theta.keys()}'
  ek = model.generate_non_interacting()
  model_1 = copy.copy(model)
  model_1.k_space = rotate_2D(model_1.k_space, theta=theta[kind])

  ek_1 = model_1.generate_non_interacting()
  # print(np.max(np.abs(ek-ek_1)))
  return np.allclose(ek, ek_1)

# Mirror symmetry check

def mirror_2D(k_space, theta):
  """k_space: k-mesh, np.ndarray(N,2)

  theta: angle of mirror axis
  mirror= R(-θ) @ M @ R(θ)
  """
  assert k_space.shape[-1] == 2, (
      'Only 2D lattice is supported, the current dimension of lattice is'
      f' {k_space.shape[-1]}'
  )
  inv_mat = np.diag([1, -1])
  mirror = rot_mat(theta) @ inv_mat @ rot_mat(-theta)
  return k_space @ mirror.T



def check_mirror(model, axis, theta):
  """y=tan(θ)x

  M_θ= R(θ) M_y R(-θ)
  M_θ H_{σ,σ'}(k) M_θ^(-1) = \tilde{R}_θ H_{σ,σ'}(k) \tilde{R}_θ^(-1), where
  \tilde{R}_θ = R(θ) (-iσ_y) R(-θ)

  ---
  axis= (s,s'), representing the position (index) of which two axis are for the spin
  """
  ek = model.generate_non_interacting()
  model_1 = copy.copy(model)
  model_1.k_space = mirror_2D(model_1.k_space, theta=theta)
  i_pauli_y = np.array([[0, 1], [-1, 0]])
  R_theta = rot_mat(theta) @ (-i_pauli_y) @ rot_mat(-theta)
  print(R_theta.shape)
  h0=model_1.generate_non_interacting()
  h0_before_idx = np.arange(len(h0.shape))+4 # by default the idx of h0 is (4,5,6,7,8,9...), the index of R_theta is (0,1), and the index of R_theta^T is (2,3)
  h0_before_idx[axis[0]], h0_before_idx[axis[1]]=1, 2 # then the two axis with spin should be (4,5,1,2,8,9,...) such that it can be contracted with the R_theta and R_theta^T, leading the new h with the new axis  being (4,5,0,3,7,8,...)
  h0_after_idx = np.arange(len(h0.shape))+4
  h0_after_idx[axis[0]], h0_after_idx[axis[1]]=0,3

  ek_1 = np.einsum(
      R_theta,
      [0, 1],
      model_1.generate_non_interacting(),
      h0_before_idx,
      R_theta.T,
      [2, 3],
      h0_after_idx,
  )
  return np.allclose(ek, ek_1)
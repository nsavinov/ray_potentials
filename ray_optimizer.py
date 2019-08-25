'''This code implements two modes of ray potentials from our paper
https://arxiv.org/pdf/1604.02885.pdf: convex and nonconvex.

As our simple example in Section A.5 shows, the convex mode fails even
in very simple cases, taking all-0.5 solution which is impossible to round
to a proper close-to-binary solution. Nonconvex mode adds
a special constraint: y_occ_i <= max(0, y_free_i-1 + x_occ_i - 1.0) -- see
equation (10) in our paper. We call it a visibility consistency constraint.
It prevents the algorithm from taking a bad solution.

In the code below we use a first-order primal-dual optimization algorithm to
optimize ray potentials.
For details on this algorithm, see equation (18) in Section 3.1 in
"Diagonal preconditioning for first order primal-dual algorithms
in convex optimization" by Pock & Chambolle. We use alpha = 1 in equation (10).
Basically, update to each variable x_i needs to be normalized by absolute
sum of coefficients in K for terms in <Kx, y> where x_i appears.
Extra primal variables (2 * x_k+1 - x_k) are used for updating
dual variables.'''


from functools import reduce
from copy import deepcopy


class RayOptimizer:
  def __init__(self,
               grid_sizes,
               rays,
               ray_costs_occ,
               ray_costs_free,
               nonconvex):
    self.grid_sizes = grid_sizes
    self.n_cells = reduce((lambda x, y: x * y), self.grid_sizes)
    self.rays = rays
    self.ray_costs_occ = ray_costs_occ
    self.ray_costs_free = ray_costs_free
    self.nonconvex = nonconvex
    self.init_variables()

  def clamp01(self, value):
    return max(0.0, min(1.0, value))

  def clamp_nonneg(self, value):
    return max(0.0, value)

  def get_zeros_along_rays(self):
    return [[0.0] * len(ray) for ray in self.rays]

  def get_ones_along_rays(self):
    return [[1.0] * len(ray) for ray in self.rays]

  def get_zeros_all_cells(self):
    return [0.0] * self.n_cells

  def init_variables(self):
    # primal variables
    self.y_occ = self.get_zeros_along_rays()
    self.y_free = self.get_ones_along_rays()
    self.x_occ = self.get_zeros_all_cells()
    # extra primal variables
    self.extra_y_occ = deepcopy(self.y_occ)
    self.extra_y_free = deepcopy(self.y_free)
    self.extra_x_occ = deepcopy(self.x_occ)
    # dual variables: correspond to constraints
    self.dual_y_occ_y_free = self.get_zeros_along_rays()
    self.dual_y_free_y_free = self.get_zeros_along_rays()
    self.dual_y_occ_x_occ = self.get_zeros_along_rays()
    self.dual_y_free_x_occ = self.get_zeros_along_rays()
    # in particular, visibility consistency constraints
    if self.nonconvex:
      self.dual_vis_con = self.get_zeros_along_rays()

  def step(self):
    self.primal()
    self.dual()

  def primal(self):
    # deltas for primal variables
    self.dy_occ = self.get_zeros_along_rays()
    self.dy_free = self.get_zeros_along_rays()
    self.dx_occ = self.get_zeros_all_cells()
    # preconditioners: we will divide deltas by them
    self.pc_y_occ = self.get_zeros_along_rays()
    self.pc_y_free = self.get_zeros_along_rays()
    self.pc_x_occ = self.get_zeros_all_cells()
    for ray_ind, ray in enumerate(self.rays):
      for ray_pos, cell in enumerate(ray):
        # costs
        self.dy_occ[ray_ind][ray_pos] += self.ray_costs_occ[ray_ind][ray_pos]
        if ray_pos >= 1:
          # y_occ_y_free
          self.dy_occ[ray_ind][ray_pos] += (
              self.dual_y_occ_y_free[ray_ind][ray_pos])
          self.pc_y_occ[ray_ind][ray_pos] += 1.0
          self.dy_free[ray_ind][ray_pos - 1] -= (
              self.dual_y_occ_y_free[ray_ind][ray_pos])
          self.pc_y_free[ray_ind][ray_pos - 1] += 1.0
          # y_free_y_free
          self.dy_free[ray_ind][ray_pos] += (
              self.dual_y_free_y_free[ray_ind][ray_pos])
          self.pc_y_free[ray_ind][ray_pos] += 1.0
          self.dy_free[ray_ind][ray_pos - 1] -= (
              self.dual_y_free_y_free[ray_ind][ray_pos])
          self.pc_y_free[ray_ind][ray_pos - 1] += 1.0
          # visibility consistency
          if self.nonconvex:
            linear_branch = (self.y_free[ray_ind][ray_pos - 1] +
                             self.x_occ[cell] - 1.0)
            self.dy_occ[ray_ind][ray_pos] += (
                self.dual_vis_con[ray_ind][ray_pos])
            self.pc_y_occ[ray_ind][ray_pos] += 1.0
            if linear_branch > 0: # linear branch is active in the constraint
              self.dy_free[ray_ind][ray_pos - 1] -= (
                  self.dual_vis_con[ray_ind][ray_pos])
              self.pc_y_free[ray_ind][ray_pos - 1] += 1.0
              self.dx_occ[cell] -= (
                  self.dual_vis_con[ray_ind][ray_pos])
              self.pc_x_occ[cell] += 1.0
        # y_occ_x_occ
        self.dy_occ[ray_ind][ray_pos] += self.dual_y_occ_x_occ[ray_ind][ray_pos]
        self.pc_y_occ[ray_ind][ray_pos] += 1.0
        self.dx_occ[cell] -= self.dual_y_occ_x_occ[ray_ind][ray_pos]
        self.pc_x_occ[cell] += 1.0
        # y_free_x_occ
        self.dy_free[ray_ind][ray_pos] += (
            self.dual_y_free_x_occ[ray_ind][ray_pos])
        self.pc_y_free[ray_ind][ray_pos] += 1.0
        self.dx_occ[cell] += self.dual_y_free_x_occ[ray_ind][ray_pos]
        self.pc_x_occ[cell] += 1.0
      # costs
      self.dy_free[ray_ind][-1] += self.ray_costs_free[ray_ind]
    self.divide_deltas_by_preconditioners()
    self.backup_primal()
    self.update_primal_by_deltas()
    self.clamp_primal()
    self.compute_extra_primal()

  def divide_deltas_by_preconditioners(self):
    for ray_ind, ray in enumerate(self.rays):
      for ray_pos, cell in enumerate(ray):
        self.dy_occ[ray_ind][ray_pos] /= max(1.0,
                                             self.pc_y_occ[ray_ind][ray_pos])
        self.dy_free[ray_ind][ray_pos] /= max(1.0,
                                              self.pc_y_free[ray_ind][ray_pos])
    for cell in range(self.n_cells):
      self.dx_occ[cell] /= max(1.0, self.pc_x_occ[cell])

  def backup_primal(self):
    self.prev_y_occ = deepcopy(self.y_occ)
    self.prev_y_free = deepcopy(self.y_free)
    self.prev_x_occ = deepcopy(self.x_occ)

  def update_primal_by_deltas(self):
    for ray_ind, ray in enumerate(self.rays):
      for ray_pos, cell in enumerate(ray):
        self.y_occ[ray_ind][ray_pos] -= self.dy_occ[ray_ind][ray_pos]
        self.y_free[ray_ind][ray_pos] -= self.dy_free[ray_ind][ray_pos]
    for cell in range(self.n_cells):
      self.x_occ[cell] -= self.dx_occ[cell]

  def clamp_primal(self):
    for ray_ind, ray in enumerate(self.rays):
      for ray_pos, cell in enumerate(ray):
        self.y_occ[ray_ind][ray_pos] = self.clamp01(
            self.y_occ[ray_ind][ray_pos])
        self.y_free[ray_ind][ray_pos] = self.clamp01(
            self.y_free[ray_ind][ray_pos])
    for cell in range(self.n_cells):
      self.x_occ[cell] = self.clamp01(self.x_occ[cell])

  def compute_extra_primal(self):
    for ray_ind, ray in enumerate(self.rays):
      for ray_pos, cell in enumerate(ray):
        self.extra_y_occ[ray_ind][ray_pos] = (
            2.0 * self.y_occ[ray_ind][ray_pos] -
            self.prev_y_occ[ray_ind][ray_pos])
        self.extra_y_free[ray_ind][ray_pos] = (
            2.0 * self.y_free[ray_ind][ray_pos] -
            self.prev_y_free[ray_ind][ray_pos])
    for cell in range(self.n_cells):
      self.extra_x_occ[cell] = (
          2.0 * self.x_occ[cell] - self.prev_x_occ[cell])

  def dual(self):
    # inline division by preconditioners because it's easier than for primal
    for ray_ind, ray in enumerate(self.rays):
      for ray_pos, cell in enumerate(ray):
        if ray_pos >= 1:
          self.dual_y_occ_y_free[ray_ind][ray_pos] += (
              self.extra_y_occ[ray_ind][ray_pos] -
              self.extra_y_free[ray_ind][ray_pos - 1]) / 2.0
          self.dual_y_free_y_free[ray_ind][ray_pos] += (
              self.extra_y_free[ray_ind][ray_pos] -
              self.extra_y_free[ray_ind][ray_pos - 1]) / 2.0
          # visibility consistency
          if self.nonconvex:
            linear_branch = (self.y_free[ray_ind][ray_pos - 1] +
                             self.x_occ[cell] - 1.0)
            if linear_branch > 0:
              self.dual_vis_con[ray_ind][ray_pos] += (
                  self.extra_y_occ[ray_ind][ray_pos] -
                  self.extra_y_free[ray_ind][ray_pos - 1] -
                  self.extra_x_occ[cell] + 1.0) / 3.0
            else:
              self.dual_vis_con[ray_ind][ray_pos] += (
                  self.extra_y_occ[ray_ind][ray_pos])
        self.dual_y_occ_x_occ[ray_ind][ray_pos] += (
            self.extra_y_occ[ray_ind][ray_pos] -
            self.extra_x_occ[cell]) / 2.0
        self.dual_y_free_x_occ[ray_ind][ray_pos] += (
            self.extra_y_free[ray_ind][ray_pos] +
            self.extra_x_occ[cell] - 1.0) / 2.0
    self.clamp_duals()

  def clamp_duals(self):
    for ray_ind, ray in enumerate(self.rays):
      for ray_pos, cell in enumerate(ray):
        self.dual_y_occ_y_free[ray_ind][ray_pos] = self.clamp_nonneg(
            self.dual_y_occ_y_free[ray_ind][ray_pos])
        self.dual_y_free_y_free[ray_ind][ray_pos] = self.clamp_nonneg(
            self.dual_y_free_y_free[ray_ind][ray_pos])
        self.dual_y_occ_x_occ[ray_ind][ray_pos] = self.clamp_nonneg(
            self.dual_y_occ_x_occ[ray_ind][ray_pos])
        self.dual_y_free_x_occ[ray_ind][ray_pos] = self.clamp_nonneg(
            self.dual_y_free_x_occ[ray_ind][ray_pos])
        if self.nonconvex:
          self.dual_vis_con[ray_ind][ray_pos] = self.clamp_nonneg(
              self.dual_vis_con[ray_ind][ray_pos])

  def get_solution(self):
    return self.x_occ

  def print_state(self):
    self.print_primal()
    self.print_dual()

  def print_primal(self):
    print('y_occ:\n', self.y_occ)
    print('y_free:\n', self.y_free)
    print('x_occ:\n', self.x_occ)

  def print_dual(self):
    print('dual_y_occ_y_free:\n', self.dual_y_occ_y_free)
    print('dual_y_free_y_free:\n', self.dual_y_free_y_free)
    print('dual_y_occ_x_occ:\n', self.dual_y_occ_x_occ)
    print('dual_y_free_x_occ:\n', self.dual_y_free_x_occ)
    if self.nonconvex:
      print('dual_vis_con:\n', self.dual_vis_con)

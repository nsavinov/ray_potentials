def create_task_from_section_a5(task_type):
  # Cell numbering: [[[0, 1, 2]]]
  # Desired solution: [[[0, 1, whatever]]]
  grid_sizes = [1, 1, 3] # 1D example represented in 3D for generality
  rays = [[0, 1, 2]] # linear indices of grid cells are listed for each ray
  gradient_step_size = 1. # implicitly specified via cost magnitudes
  ray_costs_occ = [[-2.0, -3.0, -2.0]] # one cost for each position of each ray
  ray_costs_occ = [[val * gradient_step_size for val in costs]
                   for costs in ray_costs_occ]
  ray_costs_free = [0.0] # one cost for every ray
  return {'grid_sizes' : grid_sizes,
          'rays' : rays,
          'ray_costs_occ' : ray_costs_occ,
          'ray_costs_free' : ray_costs_free,
          'nonconvex' : ('nonconvex' in task_type)}


def create_square_central_cell_task(task_type):
  # Cell numbering: [[[0, 1, 2],
  #                   [3, 4, 5],
  #                   [6, 7, 8]]]
  # Desired solution: [[[0, 0, 0],
  #                     [0, 1, 0],
  #                     [0, 0, 0]]]
  grid_sizes = [1, 3, 3]
  rays = [[0, 1, 2], [3, 4, 5], [6, 7, 8],
          [0, 3, 6], [1, 4, 7], [2, 5, 8],
          [2, 1, 0], [5, 4, 3], [8, 7, 6],
          [6, 3, 0], [7, 4, 1], [8, 5, 2]]
  ray_costs_occ = [[0., 0., 0.], [-1., -1., -1.], [0., 0., 0.],
                   [0., 0., 0.], [-1., -1., -1.], [0., 0., 0.],
                   [0., 0., 0.], [-1., -1., -1.], [0., 0., 0.],
                   [0., 0., 0.], [-1., -1., -1.], [0., 0., 0.]]
  ray_costs_free = [-1., 0., -1.,
                    -1., 0., -1.,
                    -1., 0., -1.,
                    -1., 0., -1.]
  return {'grid_sizes' : grid_sizes,
          'rays' : rays,
          'ray_costs_occ' : ray_costs_occ,
          'ray_costs_free' : ray_costs_free,
          'nonconvex' : ('nonconvex' in task_type)}


def create_square_corners_task(task_type):
  # Cell numbering: [[[0, 1, 2],
  #                   [3, 4, 5],
  #                   [6, 7, 8]]]
  # Desired solution: [[[1, 0, 1],
  #                     [0, 0, 0],
  #                     [1, 0, 1]]]
  grid_sizes = [1, 3, 3]
  rays = [[0, 1, 2], [3, 4, 5], [6, 7, 8],
          [0, 3, 6], [1, 4, 7], [2, 5, 8],
          [2, 1, 0], [5, 4, 3], [8, 7, 6],
          [6, 3, 0], [7, 4, 1], [8, 5, 2]]
  ray_costs_occ = [[-1., -1., -1.], [0., 0., 0.], [-1., -1., -1.],
                   [-1., -1., -1.], [0., 0., 0.], [-1., -1., -1.],
                   [-1., -1., -1.], [0., 0., 0.], [-1., -1., -1.],
                   [-1., -1., -1.], [0., 0., 0.], [-1., -1., -1.]]
  ray_costs_free = [0., -1., 0.,
                    0., -1., 0.,
                    0., -1., 0.,
                    0., -1., 0.]
  return {'grid_sizes' : grid_sizes,
          'rays' : rays,
          'ray_costs_occ' : ray_costs_occ,
          'ray_costs_free' : ray_costs_free,
          'nonconvex' : ('nonconvex' in task_type)}

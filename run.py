import argparse
from ray_optimizer import RayOptimizer


def create_task_from_section_a5(task_type):
  grid_sizes = [1, 1, 3]
  rays = [[0, 1, 2]]
  gradient_step_size = 0.01
  ray_costs_occ = [[-2.0, -3.0, -2.0]]
  ray_costs_occ = [[val * gradient_step_size for val in costs]
                   for costs in ray_costs_occ]
  ray_costs_free = [0.0]
  return {'grid_sizes' : grid_sizes,
          'rays' : rays,
          'ray_costs_occ' : ray_costs_occ,
          'ray_costs_free' : ray_costs_free,
          'nonconvex' : ('nonconvex' in task_type)}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-v', '--verbose', action='store_true')
  parser.add_argument('--task_type',
                      choices=['section_a5_convex',
                               'section_a5_nonconvex'],
                      default='section_a5_convex')
  args = parser.parse_args()
  task_specification = create_task_from_section_a5(args.task_type)
  ray_optimizer = RayOptimizer(**task_specification)
  for step in range(1000):
    if args.verbose:
      print('step:', step)
      ray_optimizer.print_state()
      print('-----------------')
    ray_optimizer.step()
  print('Final occupancy indicators:\n', ray_optimizer.get_solution())


if __name__ == '__main__':
  main()

'''This code tests our algorithm on the example from section A.5 in
our paper https://arxiv.org/pdf/1604.02885.pdf.'''


import argparse
from ray_optimizer import RayOptimizer
from tasks import create_task_from_section_a5


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-v', '--verbose', action='store_true')
  parser.add_argument('--task_type',
                      choices=['convex', 'nonconvex'],
                      default='nonconvex')
  args = parser.parse_args()
  task_specification = create_task_from_section_a5(args.task_type)
  ray_optimizer = RayOptimizer(**task_specification)
  for step in range(1000):
    if args.verbose:
      print('step:', step)
      ray_optimizer.print_state()
      print('-----------------------------------------------------------------')
    ray_optimizer.step()
  print('Final occupancy indicators:\n', ray_optimizer.get_solution())
  print('Desired occupancy indicators:\n', [0.0, 1.0, 'whatever'])


if __name__ == '__main__':
  main()

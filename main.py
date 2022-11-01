import os
import argparse
import sys
import numpy as np
import yaml
import random as rn
from GA import GA, write_log


def seed():
    np.random.seed(2)
    rn.seed(1)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    seed()
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        default='xgboost',
                        type=str,
                        help='Model')
    parser.add_argument('--pc',
                        default=0.8,
                        type=float,
                        help='Probability of Crossover')
    parser.add_argument('--pm',
                        default=0.2,
                        type=float,
                        help='Probability of Mutation')
    parser.add_argument('--population',
                        default=2,
                        type=int,
                        help='Population size')
    parser.add_argument('--gen',
                        default=2,
                        type=int,
                        help='Number of generation')
    parser.add_argument('--select_best_only',
                        default=True,
                        type=str2bool,
                        help='Select best individuals only')
    parser.add_argument('--percentage_split', default=50, type=int, help='')
    parser.add_argument('--percentage_back_test', default=0, type=int, help='')
    parser.add_argument('--fixed', default=False, type=str2bool, help='')
    parser.add_argument('--shuffle', default=True, type=str2bool, help='')

    args = parser.parse_args()

    ga = GA(args.model, args.percentage_split, args.percentage_back_test, args.fixed, args.shuffle)
    last_pop_fitness, fitness_gen, mae = ga.evolution(
                                                pc=args.pc,
                                                pm=args.pm,
                                                population_size=args.population,
                                                max_gen=args.gen,
                                                select_best_only=args.select_best_only)
    print("fitness: ", last_pop_fitness)
    print("gen: ", fitness_gen)
    print("MAE test: ", mae)
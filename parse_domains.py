import better_exceptions

import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', help='root of rddl directory: should contain domains, lib, parsed folders')
args = parser.parse_args()

root = args.root

domain_path = os.path.abspath(os.path.join(os.path.abspath(root), './domains')) + '/'
lib_path = os.path.abspath(os.path.join(os.path.abspath(root), './lib')) + '/'
parsed_path = os.path.abspath(os.path.join(os.path.abspath(root), './parsed')) + '/'

# domains = ['sysadmin', 'game_of_life', 'navigation', 'tamarisk', 'elevators', 'traffic', 'skill_teaching',
#            'recon', 'academic_advising', 'crossing_traffic', 'triangle_tireworld', 'wildfire']
# instances = ['1', '5', '10']

domains = ['wildfire']
instances = ['1_1', '1_2', '1_3', '1_4', '1_5', '5_1', '5_2', '5_3', '5_4', '5_5', '10_1', '10_2', '10_3', '10_4', '10_5']

rddl_parser_exec = lib_path + 'rddl-parser'

for domain in domains:
    domain_mdp = domain + '_mdp'
    for instance in instances:
        problem = domain + '_inst_mdp__' + instance
        print(problem)

        # Run rddl-parser executable
        command = rddl_parser_exec + ' ' + domain_path + domain_mdp + '.rddl ' + domain_path + problem + '.rddl ' + parsed_path
        # print(command)
        os.system(command)

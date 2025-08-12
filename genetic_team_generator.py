# -*- coding: utf-8 -*-
""" Varebu Algorithm V1

Gist
- Purpose: Genetic algorithm to form balanced volleyball teams from a player dataset.
- Core idea: Individuals are sets of teams; fitness rewards balanced per-team skills
  (Serve/Attack/Defense/Block) and penalizes duplicate players and poor gender distribution.
- Flow: ETL players -> initial population -> mutate top individuals -> select -> iterate.

Inputs
- CSV `DATA_SOURCE`.
  - If `PRECOMPUTED_SCORES = True`: expects columns
    ['Player','Gender','OverallService','OverallReception','OverallAttack','OverallBlock'].
  - Else: expects raw stats to compute the above (see code for required column names).

Key Parameters
- NUM_ITERATIONS, POPULATION_SIZE, NUM_PLAYERS, TEAM_SIZE, MUTATION_PROBABILITY.
- VARIANCE_THRESHOLD (skill balance), DUPLICATE_PLAYERS_PENALTY, BAD_GENDER_DISTRIBUTION_PENALTY.

Dependencies
- pandas, numpy, scikit-learn, matplotlib.

Run
- Python 3.8+ recommended.
- Command: `python tp_ia_adv_en.py`

Output
- Console logs of iteration progress and best fitness.
- Printed final individual (teams, average skills, gender counts).
- Plots: run fitness curves and per-skill bar charts by team.
"""

# Run parameters
NUM_ITERATIONS = 20  # @param{}
ACCEPTANCE_THRESHOLD = 0.93  # @param{}

POPULATION_SIZE = 100  # @param{}
NUM_PLAYERS = 60  # @param{}
# Percentage 1..100
MUTATION_PROBABILITY = 50  # @param{}
TEAM_SIZE = 6  # @param{}

# Fitness parameters
VARIANCE_THRESHOLD = 7  # @param{}
DUPLICATE_PLAYERS_PENALTY = 500  # @param{}
BAD_GENDER_DISTRIBUTION_PENALTY = 10  # @param{}

# DATA_SOURCE = '/content/drive/MyDrive/IAA 2C 2022/Entrega N°3/score-sabados-12.csv'
DATA_SOURCE = '/Users/franco.giannotti/Downloads/score-sabados-12.csv'
PRECOMPUTED_SCORES = True  # @param{}

# Use embedded CSV instead of reading from external file
USE_EMBEDDED_DATA = True

# Embedded CSV data (index,Player,Gender,OverallService,OverallReception,OverallAttack,OverallBlock,OverallSum)
EMBEDDED_CSV = """
,Player,Gender,OverallService,OverallReception,OverallAttack,OverallBlock,OverallSum
1,Yulse,H,90,100,100,100,390
2,Rakki,H,70,80,80,90,320
3,Ema,H,70,60,90,90,310
4,Martin,H,70,70,70,80,290
5,Boris,H,80,80,70,70,300
6,Tomi fleco,H,50,50,60,70,230
7,Mica,M,50,50,50,50,200
8,Octa joven,H,50,50,50,50,200
9,Octa,H,60,70,90,90,310
10,Ale,H,50,60,60,60,230
11,Jonas,H,80,70,70,80,300
12,cass,M,70,80,65,50,265
"""

# Imports
import pandas as pd
import numpy as np
from typing import List, Callable, Optional
import random
from sklearn import preprocessing
import math
from copy import copy
import matplotlib.pyplot as plt
import copy as builtin_copy
from io import StringIO


# Data structures
class Skills:
    def __init__(self, attack: int, defense: int, block: int, serve: int):
        self.Serve = serve
        self.Defense = defense
        self.Attack = attack
        self.Block = block

    def __str__(self) -> str:
        return f' [{self.Serve}, {self.Defense}, {self.Attack}, {self.Block}]\n'


class Player:
    def __init__(self, name: str, gender: str, skills: Skills):
        self.Name = name
        self.Gender = gender
        self.Skills = skills
        self.Overall = (
            skills.Attack + skills.Defense + skills.Serve + skills.Block
        )

    def __str__(self) -> str:
        return f'\tNAME: {self.Name}'

    def attack(self) -> int:
        return self.Skills.Attack

    def block(self) -> int:
        return self.Skills.Block

    def defense(self) -> int:
        return self.Skills.Defense

    def serve(self) -> int:
        return self.Skills.Serve


Players = List[Player]


class Team:
    def __init__(self, players: Players):
        self.Players = players

    def __str__(self) -> str:
        team_str = f'\n--------- TEAM: Players: {len(self.Players)} ---------'
        for player in self.Players:
            team_str += '\n' + str(player)
        return team_str

    def overall(self, ability_mapper: Callable) -> float:
        ability = list(map(lambda player: ability_mapper(player), self.Players))
        return round(np.mean(ability), 3)

    def overall_attack(self) -> float:
        return self.overall(lambda player: player.attack())

    def overall_defense(self) -> float:
        return self.overall(lambda player: player.defense())

    def overall_block(self) -> float:
        return self.overall(lambda player: player.block())

    def overall_serve(self) -> float:
        return self.overall(lambda player: player.serve())

    def count_women(self) -> int:
        return sum(1 for player in self.Players if player.Gender == 'M')


Teams = List[Team]


class Individual:
    def __init__(self, teams: Teams):
        self.Teams = teams
        self.__build_player_pool__()

    def __str__(self) -> str:
        indiv_str = f'\n\n######## INDIVIDUAL: Teams: {len(self.Teams)} ########'
        for team in self.Teams:
            indiv_str += '\n' + str(team)
        return indiv_str

    def __build_player_pool__(self) -> None:
        self.PlayerPool = list([])
        self.PlayerPoolSet = set([])

        for team in self.Teams:
            self.PlayerPool.extend(team.Players)
            self.PlayerPoolSet.update(team.Players)
        return

    def variance(self, ability_mapper: Callable) -> float:
        ability = list(map(lambda team: ability_mapper(team), self.Teams))
        return np.var(ability)

    def variance_attack(self) -> float:
        return self.variance(lambda team: team.overall_attack())

    def variance_defense(self) -> float:
        return self.variance(lambda team: team.overall_defense())

    def variance_serve(self) -> float:
        return self.variance(lambda team: team.overall_serve())

    def variance_block(self) -> float:
        return self.variance(lambda team: team.overall_block())

    def has_repeated_players(self) -> bool:
        return len(self.PlayerPool) != len(self.PlayerPoolSet)

    def gender_poorly_distributed(self) -> bool:
        women_count_total = sum(1 for player in self.PlayerPool if player.Gender == 'M')
        num_teams = len(self.Teams)
        women_needed_per_team = math.floor(women_count_total / num_teams)

        women_per_team = list(
            map(
                lambda team: sum(
                    1 for player in team.Players if player.Gender == 'M'
                ),
                self.Teams,
            )
        )

        return not (
            all(
                count in range(women_needed_per_team, women_needed_per_team + 2)
                for count in women_per_team
            )
        )

    def fitness_from_skills(self) -> float:
        # If variance is 0, it returns the maximum value: 4
        # If variance is between 0 and VARIANCE_THRESHOLD, it returns a value between 0 and 4
        # If variance is greater than VARIANCE_THRESHOLD, it returns a negative value (penalizing)
        return (
            4 * VARIANCE_THRESHOLD
            - (
                self.variance_block()
                + self.variance_attack()
                + self.variance_defense()
                + self.variance_serve()
            )
        ) / 4 * VARIANCE_THRESHOLD

    def fitness(self) -> float:
        fitness_value = self.fitness_from_skills()

        if self.has_repeated_players():
            fitness_value -= DUPLICATE_PLAYERS_PENALTY
        if self.gender_poorly_distributed():
            fitness_value -= BAD_GENDER_DISTRIBUTION_PENALTY

        return fitness_value

    def print_results(self) -> None:
        result = (
            f'\n\n######## INDIVIDUAL: Fitness = {self.fitness()} ########'
            + f'\nPlayers: {NUM_PLAYERS}'
            + f'\nTeams: {len(self.Teams)}'
        )

        for team in self.Teams:
            result += (
                f'\n\n--------- TEAM {self.Teams.index(team)}: Players: {len(team.Players)} ---------'
                + f'\n(S/A/D/B) = ({team.overall_serve()}, {team.overall_attack()}, {team.overall_defense()}, {team.overall_block()})'
                + f"\t\t(F/M) = ({sum(1 for player in team.Players if player.Gender == 'M')}, {sum(1 for player in team.Players if player.Gender == 'H')})"
            )

        print(result)


Individuals = List[Individual]


class Population:
    def __init__(self, individuals: Individuals):
        self.Individuals = individuals

    def __str__(self) -> str:
        pop_str = f'\n\n######## POPULATION: Individuals: {len(self.Individuals)} ########'
        for indv in self.Individuals:
            pop_str += '\n' + str(indv)
        return pop_str

    def best_individual(self) -> Individual:
        return sorted(
            self.Individuals, key=lambda individual: individual.fitness(), reverse=True
        )[0]


# Genetic algorithm
# ETL Dataset
def etl_players() -> pd.DataFrame:
    # Read dataset (embedded or external)
    if USE_EMBEDDED_DATA:
        df = pd.read_csv(StringIO(EMBEDDED_CSV), sep=',', index_col=0)
    else:
        df = pd.read_csv(DATA_SOURCE, sep=',', index_col=0)
    if not PRECOMPUTED_SCORES:
        # Raw-stats mode expects a different schema (pipe-separated). Only supported for external data.
        df = pd.read_csv(DATA_SOURCE, sep='|')

        # Compute overall scores for skills
        df['OverallService'] = np.where(
            df['Total Service Points'] == 0,
            0,
            (
                df['Total Service Points']
                + df['Ace']
                - 2 * df['Service Error']
            )
            / df['Total Service Points'],
        )
        df['OverallReception'] = np.where(
            df['Total Reception'] == 0,
            0,
            (
                df['Total Reception']
                + df['Excellent Reception']
                - 2 * (df['Error Reception'] + df['Negative Reception'])
            )
            / df['Total Reception'],
        )
        df['OverallAttack'] = np.where(
            df['Total Points Attack'] == 0,
            0,
            (
                2 * df['Excellent Attack']
                - df['Blocked Attack']
                - 2 * df['Error Attack']
            )
            / df['Total Points Attack'],
        )
        df['OverallBlock'] = np.where(
            df['Sets'] == 0, 0, (df['Points Block'] - df['Net']) / df['Sets']
        )

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))

    # Normalize scores to 0..100
    df[['OverallService', 'OverallReception', 'OverallAttack', 'OverallBlock']] = (
        scaler.fit_transform(
            df[['OverallService', 'OverallReception', 'OverallAttack', 'OverallBlock']]
        )
    )

    # Truncate to int for easier interpretation
    df['OverallService'] = np.floor(df.OverallService).astype(int)
    df['OverallReception'] = np.floor(df.OverallReception).astype(int)
    df['OverallAttack'] = np.floor(df.OverallAttack).astype(int)
    df['OverallBlock'] = np.floor(df.OverallBlock).astype(int)
    df['OverallSum'] = (
        df['OverallService']
        + df['OverallReception']
        + df['OverallAttack']
        + df['OverallBlock']
    )

    # Dataframe with relevant data for the problem
    return df[
        [
            'Player',
            'Gender',
            'OverallService',
            'OverallReception',
            'OverallAttack',
            'OverallBlock',
            'OverallSum',
        ]
    ]


def initialize_players() -> Players:
    # Read the CSV and apply transformations to obtain Overall Scores of S/A/D/B
    df_players = etl_players()

    # Transform df to a list of Player objects
    players: Players = list(
        map(
            lambda x: Player(
                name=x[0],
                gender=x[1],
                skills=Skills(
                    serve=x[2],
                    attack=x[3],
                    defense=x[4],
                    block=x[5],
                ),
            ),
            df_players.values.tolist(),
        )
    )

    return players


# Operators
# Generate initial population
def generate_individual(players: Players, num_teams: int) -> Individual:
    # Shuffle players
    random.shuffle(players)

    # Create as many empty lists as teams should be formed
    # Assign players to each team cyclically
    teams = [[] for _ in range(num_teams)]
    for player in players:
        corresponding_team_index = math.floor(players.index(player) % num_teams)
        teams[corresponding_team_index].append(player)

    # Map list of lists to list of Team and return Individual
    teams_mapped = list(map(lambda players_per_team: Team(players_per_team), teams))
    return Individual(teams_mapped)


def generate_initial_population(players: Players) -> Population:
    # Calculate number of teams based on players and TEAM_SIZE
    # Ensure at least 1 team
    num_teams = max(round(len(players) / TEAM_SIZE), 1)
    print(f'Creating {num_teams} teams for each individual...')

    # Generate the required number of individuals
    individuals: Individuals = []
    for i in range(POPULATION_SIZE):
        print(f'Generating teams for individual {i}')
        individuals.append(generate_individual(players, num_teams))

    return Population(individuals)


# Mutation
def swap_players_random(team1: Team, team2: Team) -> None:
    pos_swap_1 = random.randint(0, len(team1.Players) - 1)
    pos_swap_2 = random.randint(0, len(team2.Players) - 1)

    aux_player = team1.Players[pos_swap_1]
    team1.Players[pos_swap_1] = team2.Players[pos_swap_2]
    team2.Players[pos_swap_2] = aux_player


def swap_players_smart(
    good_team: Team,
    bad_team: Team,
    swap_criteria: Callable,
    rollback_criteria: Callable,
) -> None:
    # Order players of the good team by skill and get the top one (index 1 to avoid extreme?)
    best_player = sorted(
        good_team.Players, key=lambda player: swap_criteria(player), reverse=True
    )[1]

    # Order players of the bad team and get the worst (index 1 similarly)
    worst_player = sorted(bad_team.Players, key=lambda player: swap_criteria(player))[1]

    # Keep the ability difference for rollback decision
    ability_difference_before = np.absolute(
        rollback_criteria(good_team) - rollback_criteria(bad_team)
    )

    # Swap
    good_team.Players[good_team.Players.index(best_player)] = worst_player
    bad_team.Players[bad_team.Players.index(worst_player)] = best_player

    # Rollback if it gets worse
    ability_difference_after = np.absolute(
        rollback_criteria(good_team) - rollback_criteria(bad_team)
    )
    if ability_difference_after > ability_difference_before:
        print('Rollback swap')
        good_team.Players[good_team.Players.index(worst_player)] = best_player
        bad_team.Players[bad_team.Players.index(best_player)] = worst_player


def mutate(individual: Individual) -> Individual:
    # Shallow copy the individual to mutate
    mutated_individual = copy(individual)

    # We iterate half because we modify 2 teams per iteration
    num_teams = len(individual.Teams)
    mutations_per_individual = int(num_teams / 2)  # Truncated for odd numbers

    # Sort criteria for teams and players
    sort_criteria_list = [
        (lambda team: team.count_women(), lambda player: player.Gender, 'GENDER'),
        (lambda team: team.overall_attack(), lambda player: player.attack(), 'ATTACK'),
        (lambda team: team.overall_defense(), lambda player: player.defense(), 'DEFENSE'),
        (lambda team: team.overall_serve(), lambda player: player.serve(), 'SERVE'),
        (lambda team: team.overall_block(), lambda player: player.block(), 'BLOCK'),
    ]
    for sort_criteria in sort_criteria_list:
        # Sort teams by the given criterion, descending
        rank = sorted(individual.Teams, key=lambda team: sort_criteria[0](team), reverse=True)

        # For each pair of teams, swap the best player with the worst
        for i in range(mutations_per_individual):
            swap_players_smart(
                rank[i],
                rank[num_teams - i - 1],
                sort_criteria[1],
                sort_criteria[0],
            )

    mutated_individual.Teams = rank
    return mutated_individual


def mutation(population: Population) -> Population:
    # Takes a portion of the population and, for each individual, swaps some players between teams
    mutated_population = Population([])

    ranked_individuals = sorted(
        population.Individuals, key=lambda individual: individual.fitness(), reverse=True
    )
    num_individuals_to_mutate = int(len(ranked_individuals) / 4)
    print(f'Mutating the top {num_individuals_to_mutate} individuals...')
    for i in range(num_individuals_to_mutate):
        mutated_individual = mutate(ranked_individuals[i])
        mutated_population.Individuals.append(mutated_individual)

    print(
        f'Mutated {len(mutated_population.Individuals)} individuals of {len(population.Individuals)}'
    )
    return mutated_population


# Selection
def selection(population_obj: Population) -> Population:
    print(f'Selecting among {len(population_obj.Individuals)} individuals')

    ranked_individuals = sorted(
        population_obj.Individuals, key=lambda individual: individual.fitness(), reverse=True
    )

    selected_individuals = ranked_individuals[0:POPULATION_SIZE]
    selected_population = Population(selected_individuals)
    return selected_population


# Stop condition (unused)
def stop_condition(population_obj: Population, current_iteration: int) -> bool:
    return current_iteration >= NUM_ITERATIONS


# Best individual handling
def update_best_individual(population_obj: Population):
    global best_individual

    if best_individual is None:
        best_individual = population_obj.best_individual()
        print(
            f'No best individual defined, setting one with fitness {best_individual.fitness()}'
        )
    else:
        current_best_fitness = best_individual.fitness()
        new_best_candidate = population_obj.best_individual()
        new_best_fitness = new_best_candidate.fitness()

        if current_best_fitness < new_best_fitness:
            best_individual = new_best_candidate
            print(
                f'Found a new best individual with fitness {new_best_fitness} (previous fitness: {current_best_fitness})'
            )
    return


# Evolution strategy
best_individual: Optional['Individual'] = None

best_individual_of_run: Optional['Individual'] = None
best_individual_of_run_cycle = 0
cyclesIndividuals: list = []
cyclesMaxFitness: list = []
cyclesAvgFitness: list = []
cyclesMinFitness: list = []


def main() -> Individual:
    random.seed()

    players = initialize_players()
    random.shuffle(players)
    subset_players = players[0:NUM_PLAYERS]
    population_obj = generate_initial_population(subset_players)
    show_run_params_detail()

    for current_iteration in range(NUM_ITERATIONS):
        print(f'\nEntering loop, iteration number {current_iteration}')
        update_best_individual(population_obj)
        mutated_population_obj = mutation(builtin_copy.deepcopy(population_obj))
        population_obj = selection(
            Population(population_obj.Individuals + mutated_population_obj.Individuals)
        )
        print(f'Best individual of the run fitness: {best_individual.fitness()}')
        print(
            f'Best individual of current iteration fitness: {population_obj.best_individual().fitness()}'
        )
        compute_statistics(current_iteration, population_obj, show=True, best_is_max=True)

    validations(best_individual)
    return best_individual


# Validation and reporting
def compute_statistics(cycle, population_obj, show=True, best_is_max=True):
    global best_individual_of_run, best_individual_of_run_cycle
    global cyclesMaxFitness
    global cyclesAvgFitness
    global cyclesMinFitness
    if len(population_obj.Individuals) == 0:
        return None, 0, 0, 0
    aux_max = None
    aux_min = None
    aux_sum = 0
    aux_best_ind = None
    aux_best_ind_fitness = None
    for ind in population_obj.Individuals:
        fit = round(ind.fitness(), 2)
        aux_sum = aux_sum + fit
        if (aux_max is None) or (fit > aux_max):
            aux_max = fit
            if best_is_max:
                aux_best_ind = ind
                aux_best_ind_fitness = fit
        if (aux_min is None) or (fit < aux_min):
            aux_min = fit
            if not best_is_max:
                aux_best_ind = ind
                aux_best_ind_fitness = fit
    aux_avg = round(aux_sum / len(population_obj.Individuals), 2)
    if (best_individual_of_run is None) or (
        best_individual_of_run.fitness() > aux_best_ind.fitness()
    ):
        best_individual_of_run = aux_best_ind
        best_individual_of_run_cycle = cycle
    cyclesMaxFitness.append(aux_max)
    cyclesAvgFitness.append(aux_avg)
    cyclesMinFitness.append(aux_min)
    if show:
        print('      Max: ', aux_max, ' / Average: ', aux_avg, ' / Min: ', aux_min)
    return aux_best_ind, aux_max, aux_avg, aux_min


def show_run_params_detail():
    print('RUN EXECUTION: ')
    print('\t Stop criterion: ' + str(NUM_ITERATIONS) + ' cycles')
    print('\t Population size: ' + str(POPULATION_SIZE) + ' individuals.')
    print('\t Selection method: ranking')
    print(f"\t Mutation method: Simple with {MUTATION_PROBABILITY}% probability.")


def validations(best_individual_obj: Individual):
    # set width of bar
    barWidth = 0.25

    # dynamic labels based on team count
    num_teams = len(best_individual_obj.Teams)
    labels = [f'T{i+1}' for i in range(num_teams)]

    women_per_team = list(
        map(
            lambda team: sum(1 for player in team.Players if player.Gender == 'M'),
            best_individual_obj.Teams,
        )
    )

    men_per_team = list(
        map(
            lambda team: sum(1 for player in team.Players if player.Gender == 'H'),
            best_individual_obj.Teams,
        )
    )

    # Set position of bar on X axis
    br1 = np.arange(len(women_per_team))
    br2 = [x + barWidth for x in br1]

    # Plot 1: Men per team (black)
    plt.bar(br2, men_per_team, color='black', width=barWidth, edgecolor='grey')

    # Adding Xticks = teams
    plt.xlabel('Teams', fontweight='bold', fontsize=15)
    plt.ylabel('Number of Men', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(women_per_team))], labels)
    plt.legend()
    plt.show()

    # Plot 2: Women per team (purple)
    br1 = np.arange(len(women_per_team))
    br2 = [x + barWidth for x in br1]
    plt.bar(br1, women_per_team, color='purple', width=barWidth, edgecolor='grey')
    plt.ylabel('Number of Women', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(women_per_team))], labels)
    plt.legend()
    plt.show()

    # Plot 3: Overall Attack
    overall_attack_per_team = list(
        map(lambda team: team.overall_attack(), best_individual_obj.Teams)
    )
    overall_defense_per_team = list(
        map(lambda team: team.overall_defense(), best_individual_obj.Teams)
    )
    overall_block_per_team = list(
        map(lambda team: team.overall_block(), best_individual_obj.Teams)
    )
    overall_serve_per_team = list(
        map(lambda team: team.overall_serve(), best_individual_obj.Teams)
    )

    br1 = np.arange(len(overall_attack_per_team))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    plt.bar(br1, overall_attack_per_team, color='r', width=barWidth, edgecolor='grey')
    plt.ylabel('Overall Attack', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(overall_attack_per_team))], labels)
    plt.legend()
    plt.show()

    # Plot 4: Overall Defense
    br1 = np.arange(len(overall_attack_per_team))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    plt.bar(
        br2, overall_defense_per_team, color='b', width=barWidth, edgecolor='grey'
    )
    plt.ylabel('Overall Defense', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(overall_attack_per_team))], labels)
    plt.legend()
    plt.show()

    # Plot 5: Overall Block
    br1 = np.arange(len(overall_attack_per_team))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    plt.bar(br3, overall_block_per_team, color='g', width=barWidth, edgecolor='grey')
    plt.ylabel('Overall Block', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(overall_attack_per_team))], labels)
    plt.legend()
    plt.show()

    # Plot 6: Overall Serve
    br1 = np.arange(len(overall_attack_per_team))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    plt.bar(br4, overall_serve_per_team, color='y', width=barWidth, edgecolor='grey')
    plt.ylabel('Overall Serve', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(overall_attack_per_team))], labels)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    best_individual = None

    best_individual_of_run = None
    best_individual_of_run_cycle = 0
    cyclesIndividuals = []
    cyclesMaxFitness = []
    cyclesAvgFitness = []
    cyclesMinFitness = []

    individual = main()
    print('------- RESULT -------', individual.fitness())
    print(individual)
    individual.print_results()

    # SHOW RUN CHART
    plt.figure(figsize=(15, 8))
    plt.plot(cyclesAvgFitness)
    plt.plot(cyclesMinFitness)
    plt.plot(cyclesMaxFitness)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(-1000000, 1000000))
    plt.ticklabel_format(useOffset=False)
    plt.title('Run Results')
    plt.xlabel('Cycles')
    plt.ylabel('Fitness')
    plt.legend(['Average', 'Minimum', 'Maximum'], loc='lower right')
    plt.grid(True)
    plt.show()

    validations(best_individual)



import random


# --- 1. Define the 'Individual' (following the lesson example it is a cookie recipe) ---

# For simplicity, a recipe will be a list of 3 ingredient quantities (integers)
# e.g., [flour, sugar, chocolate_chips]

# --- 2. Define the 'Fitness Function' ---
# Let's say the "perfect" cookie has ingredients [5, 3, 7] (again it is simple example for the purposes of easier understanding)
# Our fitness will be how close a recipe is to this perfect ratio.
# Lower 'difference' means higher fitness (closer to perfect).
def calculate_fitness(recipe):
    perfect_recipe = [5, 3, 7]  # Our ideal cookie!
    difference = sum(abs(recipe[i] - perfect_recipe[i]) for i in range(len(recipe)))
    # We want to MINIMIZE this difference, so we will treat a smaller difference as higher fitness
    # For a simple fitness score, we can do 1 / (1 + difference) so higher is better
    return 1 / (1 + difference)


# --- 3. Initialization: Create the 'Population' ---
def create_initial_population(population_size, num_ingredients, max_ingredient_value):
    population = []
    for _ in range(population_size):
        # Each recipe (individual) has 'num_ingredients' values, randomly generated
        recipe = [random.randint(1, max_ingredient_value) for _ in range(num_ingredients)]
        population.append(recipe)
    return population


# --- 4. Selection: Choose 'Parents' ---
# We'll use a simple "tournament selection" for this example
def select_parents(population, num_parents_to_select, tournament_size=3):
    selected_parents = []
    for _ in range(num_parents_to_select):
        # Randomly pick individuals for the tournament
        tournament_competitors = random.sample(population, tournament_size)

        # Find the fittest competitor in the tournament
        fittest_competitor = None
        highest_fitness = -1
        for competitor in tournament_competitors:
            fitness = calculate_fitness(competitor)
            if fitness > highest_fitness:
                highest_fitness = fitness
                fittest_competitor = competitor
        selected_parents.append(fittest_competitor)
    return selected_parents


# --- 5. Reproduction: 'Crossover' ---
def crossover(parent1, parent2):
    # One-point crossover: pick a random point and swap parts
    crossover_point = random.randint(1, len(parent1) - 1)

    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


# --- 6. Reproduction: 'Mutation' ---
def mutate(recipe, mutation_rate, max_ingredient_value):
    mutated_recipe = list(recipe)  # Create a copy to avoid modifying original
    for i in range(len(mutated_recipe)):
        if random.random() < mutation_rate:  # Check if mutation occurs for this ingredient
            # Mutate by slightly changing the ingredient value
            mutated_recipe[i] = random.randint(1, max_ingredient_value)
    return mutated_recipe


# --- Putting it all together: The Genetic Algorithm Loop ---

def run_genetic_algorithm(
        population_size=20,
        num_ingredients=3,
        max_ingredient_value=10,
        generations=100,
        mutation_rate=0.1,  # 10% chance for each ingredient to mutate
        num_parents_to_select=10  # Number of parents to select for reproduction
):
    population = create_initial_population(population_size, num_ingredients, max_ingredient_value)

    print(f"Initial Population (first 5): {population[:5]}")
    print(f"Perfect recipe: {[5, 3, 7]}")

    for generation in range(generations):
        # Evaluate fitness of current population
        fitness_scores = [(recipe, calculate_fitness(recipe)) for recipe in population]

        # Sort by fitness to easily find the best (highest fitness first)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        # Keep track of the best recipe in this generation
        best_recipe_this_gen = fitness_scores[0][0]
        best_fitness_this_gen = fitness_scores[0][1]

        # Early stopping if we found the perfect recipe (fitness is 1.0)
        if best_fitness_this_gen == 1.0:
            print(f"\nFound perfect recipe at Generation {generation + 1}!")
            print(f"Best Recipe: {best_recipe_this_gen}, Fitness: {best_fitness_this_gen}")
            break

        # Selection: Choose parents for the next generation
        parents = select_parents(population, num_parents_to_select)

        next_generation = []
        # Elitism: Keep the very best individual(s) from the current generation
        # This prevents the best solution from being lost due to random operations
        num_elites = 1
        for i in range(num_elites):
            next_generation.append(fitness_scores[i][0])

        # Reproduction: Create offspring using crossover and mutation
        # We need to create enough offspring to fill the rest of the population
        while len(next_generation) < population_size:
            # Randomly pick two parents from the selected parents
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            # if possible ensure parents are different for meaningful crossover
            if parent1 == parent2 and len(parents) > 1:
                parent2 = random.choice(parents)

            child1, child2 = crossover(parent1, parent2)

            mutated_child1 = mutate(child1, mutation_rate, max_ingredient_value)
            mutated_child2 = mutate(child2, mutation_rate, max_ingredient_value)

            next_generation.append(mutated_child1)
            # Add child2 only if population size isn't exceeded
            if len(next_generation) < population_size:
                next_generation.append(mutated_child2)

        population = next_generation  # The new population for the next generation

        if (generation + 1) % 10 == 0 or generation == 0:
            print(f"\nGeneration {generation + 1}:")
            print(f"  Best Recipe: {best_recipe_this_gen}, Fitness: {best_fitness_this_gen:.4f}")
            print(f"  Population (first 5): {population[:5]}") # Uncomment to see population evolve

    final_best_recipe = fitness_scores[0][0]
    final_best_fitness = fitness_scores[0][1]
    print(f"\n--- Simulation End ---")
    print(f"Final Best Recipe Found: {final_best_recipe}")
    print(f"Final Best Fitness: {final_best_fitness:.4f}")
    if final_best_fitness == 1.0:
        print("Congratulations! The perfect cookie recipe was found!")
    else:
        print("The algorithm converged to a good (but not perfect) cookie recipe.")


# --- Run the Genetic Algorithm ---
run_genetic_algorithm()
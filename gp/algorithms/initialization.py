def syntactically_distinct(n, individual, retries):
    population = []
    expression_set = set()

    for _ in range(n):
        ind = None
        for retry in range(retries):
            ind = individual()
            expr = str(ind)
            if expr not in expression_set:
                expression_set.add(expr)
                break
        population.append(ind)

    return population


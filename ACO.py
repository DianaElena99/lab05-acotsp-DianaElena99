import numpy
import warnings
from src.Repo import Repo
from numpy import array, power, newaxis, multiply
from numpy import inf
from numpy import ones
from numpy import zeros
from numpy import sum


def run():
    warnings.filterwarnings("ignore")

    repo = Repo("easy.txt")

    dist = array(repo.container['mat'])
    iterations = 200
    citiesNr = repo.container['nrNoduri']
    antsNr = citiesNr

    evaporation_rate = 0.5
    alpha = 1 #pheromone factor
    beta = 2 #visibility factor

    #calculating the visibility of the next city
    # visibility[i][j] = 1/dist[i][j]

    visibility = 1/dist

    visibility[visibility == inf] = 0

    #initializing pheromone

    pheromone = .1*ones((antsNr, citiesNr))

    #initializing rute with len = citiesNr +1
    #bcus we want to return to source city

    rute = ones((antsNr, citiesNr+1))

    for ite in range(iterations):
        rute[:,0] = 1
        for i in range(antsNr):
            temp_vis = array(visibility)

            for j in range(citiesNr-1):
                combine_feat = zeros(citiesNr)
                cum_prob = zeros(citiesNr)

                cur_loc = int(rute[i,j]-1)

                temp_vis[:,cur_loc] = 0

                p_feat = power(pheromone[cur_loc,:], beta)
                v_feat = power(temp_vis[cur_loc,:], alpha)

                p_feat = p_feat[:,newaxis]
                v_feat = v_feat[:,newaxis]

                combine_feat = multiply(p_feat, v_feat)

                total = numpy.sum(combine_feat)

                probs = combine_feat/total

                cum_prob = numpy.cumsum(probs)

                r = numpy.random.random_sample()

                city = numpy.nonzero(cum_prob > r)[0][0] + 1

                rute[i,j+1] = city

            left = list(set([i for i in range(1, citiesNr+1)]) - set(rute[i,:-2]))[0]

            rute[i,-2] = left

        rute_opt = numpy.array(rute)

        dist_cost = numpy.zeros((antsNr, 1))

        for i in range(antsNr):

            s = 0
            for j in range(citiesNr-1):
                s = s + dist[int(rute_opt[i,j])-1, int(rute_opt[i,j+1])-1]

            dist_cost[i] = s

        dist_min_loc = numpy.argmin(dist_cost)
        dist_min_cost = dist_cost[dist_min_loc]

        best_route = rute[dist_min_loc,:]
        pheromone = (1-evaporation_rate)*pheromone

        for i in range(antsNr):
            for j in range(citiesNr-1):
                dt = 1/dist_cost[i]
                pheromone[int(rute_opt[i,j])-1, int(rute_opt[i,j+1])-1] += dt

    print('route of all the ants at the end :')
    print(rute_opt)
    print()
    print('best path :', best_route)
    print('cost of the best path', int(dist_min_cost[0]) + dist[int(best_route[-2]) - 1, 0])

if __name__ == '__main__':
    run()
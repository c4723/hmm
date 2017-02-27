"""
Exhaustive search of an HMM model
"""
import argparse
import itertools
import logging
import logging.handlers
import textwrap

class ProbMatrix(object):
    def get(self, observation, state):
        if self.probabilities.has_key(observation) and self.probabilities[observation].has_key(state):
            return self.probabilities[observation][state]
        else:
            return None

    def __init__(self, fname):
        self.states = []
        self.probabilities = {}
        self.observations = []

        with open(fname) as f:
            lines = f.readlines()
            self.observations = [x.lower() for x in lines.pop(0).split()]
            self.states = []
            self.probabilities = {}

            for line in lines:
                values = line.split()
                state = values.pop(0).lower()

                self.states.append(state)
                for obs in self.observations:
                    if obs not in self.probabilities:
                        self.probabilities[obs] = {}

                    self.probabilities[obs][state] = float(values.pop(0))


class HMM(object):
    def _calcProbability(self, prevProbabilities, observation, state):
        emissionProbability = self.emissionMatrix.get(observation, state)

        if prevProbabilities is None:
            return self.initialProbabilities[state] * emissionProbability
        else:
            probabilities = []
            for prevState in prevProbabilities:
                probabilities.append(prevProbabilities[prevState] * self.transitionMatrix.get(state, prevState) * emissionProbability)
            return sum(probabilities)

    def _getObservations(self, observationFile):
        observations = []
        with open(observationFile) as f:
            observations = [x.lower() for x in f.readline().split()]
        
        for state in self.states:
            for obs in observations:
                if self.emissionMatrix.get(obs, state) is None:
                    self._fatal("Could not find probability of {} given {}".format(obs, state))
        
        return observations
        
    def exhaustive(self, observationFile):
        observations = self._getObservations(observationFile)

        # generate all the possible states given the number of observations
        allStates = list(itertools.product(self.states, repeat=len(observations)))
        
        # calculate the probabilty of observation given each of the possible state sequences
        allProbabilities = []
        for stateSeq in allStates:
            probs = []
            prev = None

            # for each of the states in the sequence, get the transition and emission probability
            for idx, state in enumerate(stateSeq):
                if prev is None:
                    probs.append(self.initialProbabilities[state])
                else:
                    probs.append(self.transitionMatrix.get(state, prev))
                probs.append(self.emissionMatrix.get(observations[idx], state))
                prev = state

            # multiply all the probabilites to get the probability of the observation give the state sequence
            allProbabilities.append(reduce(lambda x, y: x * y, probs))

            self.logger.debug(stateSeq)
            self.logger.debug(probs)
            self.logger.debug(allProbabilities[-1])

        self.logger.debug(allProbabilities)

        # a sum of all the calculated probabilities
        return sum(allProbabilities)

    def forward(self, observationFile):
        observations = self._getObservations(observationFile)

        prevProbabilities = None
        probabilities = None
        for obs in observations:
            if probabilities is None:
                probabilities = {}
            else:
                prevProbabilities = probabilities
                probabilities = {}

            # calculate the probability of each state given the previous probabilities, and the current observation
            for state in self.states:
                probabilities[state] = self._calcProbability(prevProbabilities, obs, state)

        # retun a sum of the probability of all the possible states in the last observation
        return sum(probabilities.values())

    def _fatal(self, msg):
        self.logger.error(msg)
        exit(1)

    def __init__(self, emissionFile, transitionFile, initialFile):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        sysHandler = logging.StreamHandler()
        fmt = logging.Formatter('[%(levelname)s]: %(message)s')
        sysHandler.setFormatter(fmt)
        
        if _args.debug:
            sysHandler.setLevel(logging.DEBUG)
        else:
            sysHandler.setLevel(logging.INFO)
        
        self.logger.addHandler(sysHandler)

        self.emissionMatrix = ProbMatrix(emissionFile)
        self.states = self.emissionMatrix.states

        self.transitionMatrix = ProbMatrix(transitionFile)
        
        for state in self.states:
            for state2 in self.states:
                if self.transitionMatrix.get(state, state2) is None:
                    self._fatal("Could not find probability of observation {} given {}".format(state, state2))

        self.initialProbabilities = {}
        with open(initialFile) as f:
            for line in f:
                values = line.split()
                self.initialProbabilities[values[0].lower()] = float(values[1])

        keys = self.initialProbabilities.keys()
        for state in self.states:
            if state not in keys:
                self._fatal("Initial probability not found for " + state)

        

if __name__ == '__main__':
    _parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            Calculates probabability of an observation using exhaustive search and forward search
            All input files are split by white space in a case-insitive fashion. Examples follow.

            Example of an emission matrix files:
                   Happy  Grumpy
            Rain    0.6    0.4
            Cloudy  0.9    0.1

            Given the manner in which the files are read, the example above can be re-written as:
            happy grumpy
            rain 0.6 0.4
            cloudy 0.9 0.1

            Initial probability file:
            Rain 0.5
            Cloudy 0.5

            Observations:
            Happy Grumpy Grumpy
            '''
        )
    )

    _parser.add_argument('-e', '--emission', help='File containing the emission matrix', required=True)
    _parser.add_argument('-t', '--transition', help='File containing the transition matrix', required=True)
    _parser.add_argument('-i', '--initial', help='File containing the initial probabilities', required=True)
    _parser.add_argument('-o', '--observations', help='File with observation sequence', required=True)
    _parser.add_argument('-d', '--debug', help='Debug', default=False, required=False, action='store_true')
    _args = _parser.parse_args()

    _hmm = HMM(_args.emission, _args.transition, _args.initial)
    print "Probability of observation calculated using exhaustive search algorithm: {}".format(_hmm.exhaustive(_args.observations))
    print "Probability of observation calculated using forward algorithm: {}".format(_hmm.forward(_args.observations))

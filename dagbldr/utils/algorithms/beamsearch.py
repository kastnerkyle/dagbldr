# Author: Kyle Kastner
# License: BSD 3-Clause
# See core implementations here http://geekyisawesome.blogspot.ca/2016/10/using-beam-search-to-generate-most.html
import numpy as np
import heapq
import collections


class Beam(object):
    """
    From http://geekyisawesome.blogspot.ca/2016/10/using-beam-search-to-generate-most.html
    For comparison of prefixes, the tuple (prefix_probability, complete_sentence) is used.
    This is so that if two prefixes have equal probabilities then a complete sentence is preferred
    over an incomplete one since (0.5, False, whatever_prefix) < (0.5, True, some_other_prefix)
    """
    def __init__(self, beam_width, init_beam=None, use_log=True,
                 stochastic=False, temperature=1.0, random_state=None):
        if init_beam is None:
            self.heap = list()
        else:
            self.heap = init_beam
            heapq.heapify(self.heap)
        self.stochastic = stochastic
        self.random_state = random_state
        self.temperature = temperature
        # use_log currently unused...
        self.use_log = use_log
        self.beam_width = beam_width

    def add(self, score, complete, prob, prefix):
        heapq.heappush(self.heap, (score, complete, prob, prefix))
        while len(self.heap) > self.beam_width:
            if self.stochastic:
                # same whether logspace or no?
                probs = np.array([h[2] for h in self.heap])
                probs = probs / self.temperature
                e_x = np.exp(probs - np.max(probs))
                s_x = e_x / e_x.sum()
                is_x = 1. - s_x
                is_x = is_x / is_x.sum()
                to_remove = self.random_state.multinomial(1, is_x).argmax()
                completed = [n for n, h in enumerate(self.heap) if h[2] == True]
                # Don't remove completed sentences by randomness
                if to_remove not in completed:
                    # there must be a faster way...
                    self.heap.pop(to_remove)
                    heapq.heapify(self.heap)
                else:
                    heapq.heappop(self.heap)
            else:
                # remove lowest score from heap
                heapq.heappop(self.heap)

    def __iter__(self):
        return iter(self.heap)


def _beamsearch(probabilities_function, beam_width=10, clip_len=-1,
                start_token="<START>", end_token="<EOS>", use_log=True,
                renormalize=True, length_score=True,
                stochastic=False, temperature=1.0,
                random_state=None, eps=1E-9):
    """
    THIS IS THE CORE ALGORITHM - WRAPPED SO THAT IT RETURNS ON SCORE, RATHER THAN LENGTH (AS IN YIELD/GENERATOR)
    From http://geekyisawesome.blogspot.ca/2017/04/getting-top-n-most-probable-sentences.html

    returns a generator, which will yield beamsearched sequences in order of their probability

    "probabilities_function" returns a list of (next_prob, next_word) pairs given a prefix.

    "beam_width" is the number of prefixes to keep (so that instead of keeping the top 10 prefixes you can keep the top 100 for example).
    By making the beam search bigger you can get closer to the actual most probable sentence but it would also take longer to process.

    "clip_len" is a maximum length to tolerate, beyond which the most probable prefix is returned as an incomplete sentence.
    Without a maximum length, a faulty probabilities function which does not return a highly probable end token
    will lead to an infinite loop or excessively long garbage sentences.

    "start_token" can be a single string (token), or a sequence of tokens

    "end_token" is a single string (token), or a sequence of tokens that signifies end of the sequence

    "use_log, renormalize, length_score" are all related to calculation of beams to keep
    and should improve results when True

    "stochastic" uses a different sampling algorithm for reducing/aggregating beams
    it should result in more diverse and interesting outputs

    "temperature" is the softmax temperature for the underlying stochastic
    beamsearch - the default of 1.0 is usually fine

    "random_state" is a np.random.RandomState() object, passed when using the
    stochastic beamsearch in order to control randomness

    "eps" minimum probability for log-space calculations, to avoid numerical issues
    """
    if stochastic:
        if random_state is None:
            raise ValueError("Must pass np.random.RandomState() object if stochastic=True")

    completed_beams = 0
    prev_beam = Beam(beam_width - completed_beams, None, use_log, stochastic,
                     temperature, random_state)
    try:
        basestring
    except NameError:
        basestring = str

    if isinstance(start_token, collections.Sequence) and not isinstance(start_token, basestring):
        start_token = start_token
    elif isinstance(start_token, basestring) and len(start_token) > 1:
        start_token = list(start_token)
    else:
        # make it a list with 1 entry
        start_token = [start_token]

    if isinstance(end_token, collections.Sequence) and not isinstance(end_token, basestring):
        end_token = end_token
        end_token_is_seq = True
    elif isinstance(end_token, basestring) and len(end_token) > 1:
        end_token = list(end_token)
        end_token_is_seq = True
    else:
        # make it a list with 1 entry
        end_token = [end_token]
        end_token_is_seq = False

    if use_log:
        prev_beam.add(.0, False, .0, start_token)
    else:
        prev_beam.add(1.0, False, 1.0, start_token)


    while True:
        curr_beam = Beam(beam_width - completed_beams, None, use_log, stochastic,
                         temperature, random_state)
        if renormalize:
            sorted_prev_beam = sorted(prev_beam)
            # renormalize by the previous minimum value in the beam
            min_prob = sorted_prev_beam[0][0]
        else:
            if use_log:
                min_prob = 0.
            else:
                min_prob = 1.

        # Add complete sentences that do not yet have the best probability to the current beam, the rest prepare to add more words to them.
        for (prefix_score, complete, prefix_prob, prefix) in prev_beam:
            if complete == True:
                curr_beam.add(prefix_score, True, prefix_prob, prefix)
            else:
                # Get probability of each possible next word for the incomplete prefix
                for (next_prob, next_word) in probabilities_function(prefix):
                    # use eps tolerance to avoid log(0.) issues
                    if next_prob > eps:
                        n = next_prob
                    else:
                        n = eps

                    # score is renormalized prob
                    if use_log:
                        if length_score:
                            score = prefix_prob + np.log(n) + np.log(len(prefix)) - min_prob
                        else:
                            score = prefix_prob + np.log(n) - min_prob
                        prob = prefix_prob + np.log(n)
                    else:
                        if length_score:
                            score = (prefix_prob * n * len(prefix)) / min_prob
                        else:
                            score = (prefix_prob * n) / min_prob
                        prob = prefix_prob * n

                    if end_token_is_seq:
                        left_cmp = prefix[-len(end_token) + 1:] + [next_word]
                        right_cmp = end_token
                    else:
                        left_cmp = next_word
                        right_cmp = end_token[0]

                    if left_cmp == right_cmp:
                        # If next word is the end token then mark prefix as complete
                        curr_beam.add(score, True, prob, prefix + [next_word])
                    else:
                        curr_beam.add(score, False, prob, prefix + [next_word])

        # Get all prefixes in beam sorted by probability
        sorted_beam = sorted(curr_beam)

        any_removals = False
        while True:
            # Get highest probability prefix - heapq is sorted in ascending order
            (best_score, best_complete, best_prob, best_prefix) = sorted_beam[-1]
            if best_complete == True or len(best_prefix) - 1 == clip_len:
                # If most probable prefix is a complete sentence or has a length that
                # exceeds the clip length (ignoring the start token) then return it
                # yield best without start token, along with probability
                yield (best_prefix, best_score, best_prob)
                sorted_beam.pop()
                completed_beams += 1
                any_removals = True
                # If there are no more sentences in the beam then stop checking
                if len(sorted_beam) == 0:
                    break
            else:
                break

        if any_removals == True:
            if len(sorted_beam) == 0:
                break
            else:
                prev_beam = Beam(beam_width - completed_beams, sorted_beam, use_log,
                                 stochastic, temperature, random_state)
        else:
            prev_beam = curr_beam


def beamsearch(probabilities_function, beam_width=10, clip_len=-1,
               start_token="<START>", end_token="<EOS>", use_log=True,
               renormalize=True, length_score=True,
               stochastic=False, temperature=1.0,
               random_state=None, eps=1E-9):
    """
    From http://geekyisawesome.blogspot.ca/2017/04/getting-top-n-most-probable-sentences.html

    returns a generator, which will yield beamsearched sequences in order of their probability

    "probabilities_function" returns a list of (next_prob, next_word) pairs given a prefix.

    "beam_width" is the number of prefixes to keep (so that instead of keeping the top 10 prefixes you can keep the top 100 for example).
    By making the beam search bigger you can get closer to the actual most probable sentence but it would also take longer to process.

    "clip_len" is a maximum length to tolerate, beyond which the most probable prefix is returned as an incomplete sentence.
    Without a maximum length, a faulty probabilities function which does not return a highly probable end token
    will lead to an infinite loop or excessively long garbage sentences.

    "start_token" can be a single string (token), or a sequence of tokens

    "end_token" is a single string (token), or a sequence of tokens that signifies end of the sequence

    "use_log, renormalize, length_score" are all related to calculation of beams to keep
    and should improve results when True

    "stochastic" uses a different sampling algorithm for reducing/aggregating beams
    it should result in more diverse and interesting outputs

    "temperature" is the softmax temperature for the underlying stochastic
    beamsearch - the default of 1.0 is usually fine

    "random_state" is a np.random.RandomState() object, passed when using the
    stochastic beamsearch in order to control randomness

    "eps" minimum probability for log-space calculations, to avoid numerical issues
    """
    b = _beamsearch(probabilities_function, beam_width=beam_width,
                    clip_len=clip_len, start_token=start_token,
                    end_token=end_token, use_log=use_log,
                    renormalize=renormalize, length_score=length_score,
                    stochastic=stochastic, temperature=temperature,
                    random_state=random_state, eps=eps)
    all_results = []
    # get all beams
    for result in b:
        all_results.append(result)
    # sort by score
    all_results  = sorted(all_results, key=lambda x: x[1])
    return all_results

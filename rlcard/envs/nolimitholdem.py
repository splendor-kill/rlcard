import json
import os
import numpy as np
from collections import OrderedDict

import rlcard
from rlcard.envs import Env
from rlcard.games.nolimitholdem import Game
from rlcard.games.nolimitholdem.round import Action

DEFAULT_GAME_CONFIG = {
        'game_num_players': 2,
        'chips_for_each': 100,
        'dealer_id': None,
        'reverse_blind': False,
        'small_blind': 1,
        }

class NolimitholdemEnv(Env):
    ''' Limitholdem Environment
    '''

    def __init__(self, config):
        ''' Initialize the Limitholdem environment
        '''
        self.name = 'no-limit-holdem'
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        super().__init__(config)
        self.actions = Action
        self.reward_ver = config.get('reward_version', 0)
        assert 0 <= self.reward_ver <= 1
        self.state_ver = config.get('state_version', 0)
        assert 0 <= self.state_ver <= 1
        shape0 = 997 if self.state_ver == 1 else 54
        self.state_shape = [[shape0] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]
        # for raise_amount in range(1, self.game.init_chips+1):
        #     self.actions.append(raise_amount)
        self.action_recorder_extra = []

        with open(os.path.join(rlcard.__path__[0], 'games/limitholdem/card2index.json'), 'r') as file:
            self.card2index = json.load(file)

    def _get_legal_actions(self):
        ''' Get all leagal actions

        Returns:
            encoded_action_list (list): return encoded legal action list (from str to int)
        '''
        return self.game.get_legal_actions()

    def reset(self):
        self.action_recorder_extra = []
        return super().reset()

    def step(self, action, raw_action=False):
        stg = self.game.stage
        next_obs, player_id = super().step(action, raw_action)
        assert len(self.action_recorder) > 0
        p, a = self.action_recorder[-1]
        self.action_recorder_extra.append((p, stg.value, a.value))
        return next_obs, player_id

    def _extract_state(self, state):
        ''' Extract the state representation from state dictionary for agent

        Note: Currently the use the hand cards and the public cards. TODO: encode the states

        Args:
            state (dict): Original state from the game

        Returns:
            observation (list): combine the player's score and dealer's observable score for observation
        '''
        extracted_state = {}

        legal_actions = OrderedDict({action.value: None for action in state['legal_actions']})
        extracted_state['legal_actions'] = legal_actions

        if self.state_ver == 0:
            public_cards = state['public_cards']
            hand = state['hand']
            my_chips = state['my_chips']
            all_chips = state['all_chips']
            cards = public_cards + hand
            idx = [self.card2index[card] for card in cards]
            obs = np.zeros(54)
            obs[idx] = 1
            obs[52] = float(my_chips)
            obs[53] = float(max(all_chips))
        elif self.state_ver == 1:
            obs = self._build_obs(state, self.action_recorder_extra)
        extracted_state['obs'] = obs

        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
        extracted_state['action_record'] = self.action_recorder

        return extracted_state

    def get_payoffs(self):
        ''' Get the payoff of a game

        Returns:
           payoffs (list): list of payoffs
        '''
        payoffs = np.array(self.game.get_payoffs())
        if self.reward_ver == 1:
            payoffs = np.sign(payoffs) + payoffs / self.game.init_chips
        return payoffs

    def _decode_action(self, action_id):
        ''' Decode the action for applying to the game

        Args:
            action id (int): action id

        Returns:
            action (str): action for the game
        '''
        legal_actions = self.game.get_legal_actions()
        if self.actions(action_id) not in legal_actions:
            if Action.CHECK in legal_actions:
                return Action.CHECK
            else:
                print("Tried non legal action", action_id, self.actions(action_id), legal_actions)
                return Action.FOLD
        return self.actions(action_id)

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        state = {}
        state['chips'] = [self.game.players[i].in_chips for i in range(self.num_players)]
        state['public_card'] = [c.get_index() for c in self.game.public_cards] if self.game.public_cards else None
        state['hand_cards'] = [[c.get_index() for c in self.game.players[i].hand] for i in range(self.num_players)]
        state['current_player'] = self.game.game_pointer
        state['legal_actions'] = self.game.get_legal_actions()
        return state

    def _build_obs(self, state, action_rec):
        '''Build the observation for an agent

        Args:
            state (dict): The original state
            action_recorder (list): The list of actions played before

        Returns:
            ndarray: The observation for an agent
        '''
        MAX_NUM_PLAYERS = 9
        assert 2 <= self.num_players <= MAX_NUM_PLAYERS
        public_cards = state['public_cards']
        hand = state['hand']
        player_id = state['current_player']
        all_chips0 = np.array(state['all_chips']) / self.game.big_blind
        all_chips = np.zeros(MAX_NUM_PLAYERS)
        all_chips[: self.num_players] = all_chips0
        stakes0 = np.array(state['stakes']) / self.game.big_blind
        stakes = np.zeros(MAX_NUM_PLAYERS)
        stakes[: self.num_players] = stakes0
        cards = public_cards + hand
        idx = [self.card2index[card] for card in cards]
        cards = np.zeros(52)
        cards[idx] = 1

        num_players_mask = np.zeros(MAX_NUM_PLAYERS)
        num_players_mask[: self.num_players] = 1  # 例如6人局，mask=[1,1,1,1,1,1,0,0,0]

        current_player_mask = np.zeros(MAX_NUM_PLAYERS)
        current_player_mask[player_id] = 1

        n_actions = len(Action)
        legal_actions_mask = np.zeros(n_actions)
        legal_actions_mask[[a.value for a in state['legal_actions']]] = 1

        n_stages = 4  # len(self.game.stages)
        stage = np.zeros(n_stages)
        stage[state['stage'].value] = 1

        MAX_NUN_BETS = 5
        action_seq = np.zeros((n_stages, MAX_NUN_BETS, MAX_NUM_PLAYERS, n_actions))
        stage_count = {(i, p): 0 for i in range(n_stages) for p in range(self.num_players)}
        for p, stg, a in action_rec[::-1]:
            stage_count[(stg, p)] += 1
            if stage_count[(stg, p)] > MAX_NUN_BETS:
                continue
            action_seq[stg, stage_count[(stg, p)] - 1, p, a] = 1
        action_seq = action_seq.flatten()

        obs = np.concatenate(
            [cards, num_players_mask, current_player_mask, all_chips, stakes, legal_actions_mask, stage, action_seq]
        )
        return obs

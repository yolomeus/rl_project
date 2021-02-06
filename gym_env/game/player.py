from gym_env.game.buildings import Building


class Player:
    def __init__(self, player_id, board):
        self.player_id = player_id
        self.buildings = []
        self.board = board

    def act(self, action):
        pass

    def register_building(self, building: Building):
        self.buildings.append(building)

    def get_observation(self):
        return self.board.to_binary()

    @property
    def current_reward(self):
        building_reward = sum([building.compute_reward() for building in self.buildings])
        return building_reward

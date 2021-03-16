import os

import hydra
from omegaconf import DictConfig
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.dqn import MlpPolicy

from gym_env.env import Expando


def get_env(op_policies, conf):
    env = make_vec_env(Expando,
                       env_kwargs=dict(**conf,
                                       policies_other=op_policies),
                       n_envs=1)
    env.reset()
    return env


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        return True

    def _on_rollout_end(self) -> None:
        self.init_callback(self.model)
        game = self.training_env.get_attr('game', indices=0)[0]
        player = game.players[0]
        self.logger.record('rollout/happiness', player.happiness_penalty)
        self.logger.record('rollout/room', player.room)
        self.logger.record('rollout/population', player.population)
        self.logger.record('rollout/total_reward', player.total_reward)


class SelfPlay(BaseCallback):
    def __init__(self, checkpoint_path, env_conf):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.env_conf = env_conf

    def _on_step(self) -> bool:
        ckpt_path = os.path.join(self.checkpoint_path, f'timestep_{self.num_timesteps}')
        self.model.save(ckpt_path)
        saved_policy = self.model.__class__.load(ckpt_path)
        env = get_env([saved_policy], self.env_conf)
        self.model.set_env(env)
        self.init_callback(self.model)
        return True


@hydra.main(config_path='config/', config_name='config')
def main(cfg: DictConfig):
    env = get_env(None, cfg.env)
    model = DQN(MlpPolicy,
                env,
                **cfg.model,
                tensorboard_log='logs/',
                verbose=1)

    callbacks = [TensorboardCallback()]
    if cfg.self_play:
        self_play = EveryNTimesteps(cfg.n_update_selfplay, callback=SelfPlay('ckpts/', cfg.env))
        callbacks.append(self_play)
    if cfg.ckpt_freq:
        ckpt_cb = CheckpointCallback(save_freq=cfg.ckpt_freq, save_path='ckpts/')
        callbacks.append(ckpt_cb)

    model.learn(total_timesteps=cfg.n_total_steps, callback=callbacks, tb_log_name=cfg.log_name)


if __name__ == '__main__':
    main()

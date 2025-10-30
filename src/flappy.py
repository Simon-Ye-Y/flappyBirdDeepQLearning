import asyncio
import sys
from collections import deque

import pygame
import torch
from pygame.locals import K_ESCAPE, K_SPACE, K_UP, KEYDOWN, QUIT

from .entities import (
    Background,
    Floor,
    GameOver,
    Pipes,
    Player,
    PlayerMode,
    Score,
    WelcomeMessage,
)
from .utils import GameConfig, Images, Sounds, Window
from FlapPyBird.src.DQN import Agent


class Flappy:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Flappy Bird")
        window = Window(288, 512)
        screen = pygame.display.set_mode((window.width, window.height))
        images = Images()
        self.num_episodes = 50
        self.config = GameConfig(
            screen=screen,
            clock=pygame.time.Clock(),
            fps=30,
            window=window,
            images=images,
            sounds=Sounds(),
        )
        self.agent = Agent(
            gamma=0.99,
            epsilson=0.5,
            lr=0.001,
            input_dims=[3],
            batch_size=32,
            n_actions=2,
            max_mem_size=100000,
            eps_end=0.05,
            eps_dec=1e-4
        )

    async def start(
        self,
        *,
        mode="train",
        episodes=None,
        save_best_model_path=None,
        best_average_window=10,
        model_path=None,
        evaluation_episodes=None,
    ):
        if mode == "train":
            total_episodes = episodes or self.num_episodes
            await self._train_loop(
                episodes=total_episodes,
                save_best_model_path=save_best_model_path,
                best_average_window=best_average_window,
            )
        elif mode == "evaluate":
            total_episodes = evaluation_episodes or episodes or 1
            await self._evaluate_loop(episodes=total_episodes, model_path=model_path)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def reset_scene(self):
        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.player = Player(self.config)
        self.welcome_message = WelcomeMessage(self.config)
        self.game_over_message = GameOver(self.config)
        self.pipes = Pipes(self.config)
        self.score = Score(self.config)

    async def _train_loop(self, episodes, save_best_model_path=None, best_average_window=10):
        reward_window = deque(maxlen=best_average_window) if best_average_window else None
        best_metric = float("-inf")

        for episode in range(1, episodes + 1):
            episode_reward = await self._run_episode(training=True)
            if reward_window is not None:
                reward_window.append(episode_reward)
                metric = sum(reward_window) / len(reward_window)
            else:
                metric = episode_reward

            if save_best_model_path and metric > best_metric:
                self.agent.save_model(save_best_model_path)
                best_metric = metric

            print(f"Episode {episode}: reward={episode_reward:.2f}, metric={metric:.2f}")

    async def _evaluate_loop(self, episodes, model_path=None):
        if model_path:
            self.agent.load_model(model_path, load_optimizer=False)
        previous_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0

        for episode in range(1, episodes + 1):
            self.reset_scene()
            print(f"Evaluation episode {episode}")
            await self.play(ai_controlled=True)

        self.agent.epsilon = previous_epsilon

    async def _run_episode(self, training=True):
        self.reset_scene()
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)
        done = False
        observation = self.closest_entity()
        total_reward = 0.0
        flap_cooldown = 15
        flap_counter = 0

        while not done:
            action = 0
            pipe_distance = observation[2] if len(observation) > 2 else 1
            if flap_counter == 0:
                action = self.select_action(observation, exploit=not training)
                if action:
                    self.player.flap()
                    flap_counter = flap_cooldown
            else:
                flap_counter -= 1

            crossed = False
            for pipe in self.pipes.upper:
                if self.player.crossed(pipe):
                    crossed = True
                    self.score.add()
                    break

            next_state = self.closest_entity()
            done = self.player.collided(self.pipes, self.floor)

            distance = max(abs(pipe_distance), 1)
            step_reward = 0.0
            if crossed:
                step_reward += 100

            if self.pipes.upper and self.pipes.lower:
                upper_pipe = self.pipes.upper[0]
                lower_pipe = self.pipes.lower[0]
                if self.player.y > upper_pipe.rect.bottom or self.player.y < lower_pipe.rect.top:
                    step_reward -= 50 / distance
                else:
                    step_reward += 50 / distance

            if done:
                step_reward -= 100

            total_reward += step_reward

            if training:
                self.agent.store_transition(observation, action, step_reward, next_state, done)
                self.agent.learn()

            observation = next_state

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

        return total_reward

    async def splash(self):
        """Shows welcome splash screen animation of flappy bird"""

        self.player.set_mode(PlayerMode.SHM)

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    return

            self.background.tick()
            self.floor.tick()
            self.player.tick()
            self.welcome_message.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    def check_quit_event(self, event):
        if event.type == QUIT or (
                event.type == KEYDOWN and event.key == K_ESCAPE
        ):
            pygame.quit()
            sys.exit()

    def is_tap_event(self, event):
        m_left, _, _ = pygame.mouse.get_pressed()
        space_or_up = event.type == KEYDOWN and (
                event.key == K_SPACE or event.key == K_UP
        )
        screen_tap = event.type == pygame.FINGERDOWN
        return m_left or space_or_up or screen_tap

    async def play(self, ai_controlled=False):
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)

        while True:
            if self.player.collided(self.pipes, self.floor):
                return

            for pipe in self.pipes.upper:
                if self.player.crossed(pipe):
                    self.score.add()
                    break

            for event in pygame.event.get():
                self.check_quit_event(event)
                if not ai_controlled and self.is_tap_event(event):
                    self.player.flap()

            if ai_controlled:
                observation = self.closest_entity()
                if self.select_action(observation, exploit=True):
                    self.player.flap()

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    async def game_over(self):
        """crashes the player down and shows gameover image"""

        self.player.set_mode(PlayerMode.CRASH)
        self.pipes.stop()
        self.floor.stop()

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    if self.player.y + self.player.h >= self.floor.y - 1:
                        return

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()
            self.game_over_message.tick()

            self.config.tick()
            pygame.display.update()

    def closest_entity(self):
        # Assuming screen width as screen_width
        screen_height = self.config.window.height
        player = self.player

        # Set default values
        default_top = screen_height
        default_bot = 0

        # Get the position of the nearest upper and lower pipes
        nearest_pipe_x = 0
        nearest_entity_y_upper_bottom = default_bot
        nearest_entity_y_lower_top = default_top

        if len(self.pipes.upper) > 0 and len(self.pipes.lower) > 0:
            nearest_upper_pipe = self.pipes.upper[0]
            nearest_pipe_x = nearest_upper_pipe.x
            nearest_entity_y_upper_bottom = nearest_upper_pipe.rect.bottom
            nearest_lower_pipe = self.pipes.lower[0]
            nearest_entity_y_lower_top = nearest_lower_pipe.rect.top

        state = [
            player.rect.y - nearest_entity_y_upper_bottom,
            player.rect.y - nearest_entity_y_lower_top,
            nearest_pipe_x - player.rect.x,
        ]
        return state

    def select_action(self, observation, exploit=False):
        state_tensor = torch.tensor(observation, dtype=torch.float32).to(self.agent.Q_eval.device)
        if exploit:
            with torch.no_grad():
                actions = self.agent.Q_eval.forward(state_tensor)
            return torch.argmax(actions).item()
        action = self.agent.choose_action(state_tensor)
        print(state_tensor)
        return action

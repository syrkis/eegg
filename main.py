# %% main.py
#   eegg
# by: Noah Syrkis
# from stable_baselines3 import PPO
# from stable_baselines3.common import logger
from ollama import chat, ChatResponse, Message
from stable_baselines3 import PPO
from stable_baselines3.common import logger
import numpy as np
import base64
import os
from PIL import Image
from craftium.wrappers import DiscreteActionWrapper
import io

# import gymnasium as gym
import craftium  # noqa
from craftium import CraftiumEnv


# %% Utils
def image_to_base64(image):
    image = Image.fromarray(image.astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image_base64


def god_fn(
    messages, observation, reward, terminated, truncated
) -> Message:  # GOD of parrent that tells what is going on
    message = {
        "role": "user",
        "content": f"Reward: {reward}",
        "images": [image_to_base64(observation)],
    }
    messages += [message]
    response: ChatResponse = chat(model="god", messages=messages + [message])
    messages += [response.message]
    return messages


def ova_fn(
    observation, reward, terminated, truncated
) -> int:  # three leter word for agent inside an egg (also name for collection of cells in an egg)
    return 0  # Action and words to god


def egg_fn(env_name):  # the egg in which the agent is in parented by god
    path = os.path.join(os.path.dirname(craftium.__file__), "craftium-envs", env_name)
    env = CraftiumEnv(
        env_dir=path,  # type: ignore
        render_mode="rgb_array",
        obs_width=64,
        obs_height=64,
    )

    # Wrap the environment to simplify the action space
    env = DiscreteActionWrapper(
        env,
        actions=[
            "forward",
            "mouse x+",
            "mouse x-",
        ],  # Specify the actions you want to use
        mouse_mov=0.5,
    )

    model = PPO("CnnPolicy", env, verbose=1)
    new_logger = logger.configure("logs-ppo-agent", ["stdout", "csv"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=100)

    env.close()


egg_fn("chop-tree")


#
# observation, info = env.reset()
# messages = []
# reward, terminated, truncated = 0, False, False
# action = env.action_space.sample()
# print(action)
# exit()
#
# for t in range(10):
# action = ova_fn(messages, observation, reward, terminated, truncated)
# observation, reward, terminated, truncated, _info = env.step(action)
# messages = god_fn(messages, observation, reward, terminated, truncated)
#
# if terminated or truncated:
# observation, info = env.reset()
#
# env.close()

import os
import torch
import gymnasium as gym
import numpy as np
import argparse

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from sac_rgbd_base import Actor, ReplayBuffer, Args


def collect_data(args, checkpoint_path, eval_env, replay_buffer, output_buffer_file):


    # Load model checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Initialize actor and load weights
    obs, _ = eval_env.reset()
    actor = Actor(eval_env, sample_obs=obs).to(device)

    try:
        # Load model strictly, will raise an error if mismatched
        actor.load_state_dict(checkpoint['actor'])
    except RuntimeError as e:
        print(f"Strict load failed with error: {e}")
        print("Attempting to load with non-strict mode to ignore mismatched layers.")
        actor.load_state_dict(checkpoint['actor'], strict=False)

    actor.eval()

    obs, _ = eval_env.reset()

    # Evaluation loop
    step_count = 0
    while step_count < args.total_timesteps:
    # for step_count in range(args.total_timesteps):
        with torch.no_grad():
            action = actor.get_eval_action(obs)
            
        next_obs, reward, done, trunc, infos = eval_env.step(action)
        step_count += 1

        real_next_obs = {k: v.clone() for k, v in next_obs.items()}

        replay_buffer.add(obs, real_next_obs, action, reward, done)

        obs = next_obs
    
    # print(f"Total expert samples added to buffer: {replay_buffer.rewards.shape[0]}")
    
    # #save buffer as pt
    # # output_buffer_file = "expert_buffer_2.pt"
    # torch.save(replay_buffer, output_buffer_file)
    # print("Replay buffer saved.")
    print(replay_buffer.rewards.shape)
    print(replay_buffer.rewards)
    
    eval_env.close()

# def inspect_buffer(output_buffer_file):

    

    # print(f"\n\nLoading expert buffer from {output_buffer_file}...")
    # expert_rb = torch.load(output_buffer_file, weights_only=False)

    # print(f"Length of buffer {len(expert_rb.rewards)}")
    # print(expert_rb.rewards[-10:])

if __name__ == "__main__":
    args = Args(exp_name="sac_rgbd", env_id="PickCube-v1", num_envs = 2, obs_mode="rgb", cuda=True, total_timesteps=200, buffer_size = 200,
                control_mode="pd_ee_delta_pos"  # Ensure this matches your training config
                )

    #init env
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment setup
    env_kwargs = dict(
        obs_mode=args.obs_mode, 
        render_mode=args.render_mode, 
        sim_backend="gpu",
        control_mode=args.control_mode  # Ensuring control mode matches training
    )
    eval_env = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
    eval_env = FlattenRGBDObservationWrapper(eval_env, rgb=True, depth=False, state=args.include_state)
    eval_env = ManiSkillVectorEnv(eval_env, args.num_envs, ignore_terminations=True, auto_reset=True ,record_metrics=True) #set 'auto_reset = False' if you don't want the env to reset after reaching goal

    # Print action space to verify dimensions
    print(f"Action space during evaluation: {eval_env.single_action_space.shape}")

    #init replay buffer
    rb = ReplayBuffer(env=eval_env, num_envs=args.num_envs, buffer_size=args.buffer_size, 
                        storage_device=device, sample_device=device    
                        )


    # Specify your checkpoint path here
    checkpoint_path = "runs/PickCube-v1__sac_rgbd__1__1738602197/ckpt_250048.pt"
    output_buffer_file = "expert_buffer_2.pt"

    collect_data(args=args, checkpoint_path=checkpoint_path, eval_env=eval_env, replay_buffer=rb, output_buffer_file=output_buffer_file)

    # inspect_buffer(output_buffer_file)















"""


def collect_expert_data(
    env_id="PickCube-v1",
    obs_mode="rgb",
    num_envs=1,
    control_mode="pd_ee_delta_pos",
    expert_ckpt="runs/PickCube-v1__sac_rgbd__1__1738602197/ckpt_250048.pt",
    output_buffer_file="expert_buffer.pt",
    total_steps=50_000,
    buffer_size=300_000,
):
    
    print(f"Collecting data from expert={expert_ckpt}, saving to={output_buffer_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ############################################################################
    # 1) Create the environment on CPU, forcing camera=64×64
    ############################################################################
    env_kwargs = dict(
        obs_mode=obs_mode,
        render_mode="none",
        sim_backend="cpu",
        sensor_configs=dict(width=64, height=64),
    )
    if control_mode is not None:
        env_kwargs["control_mode"] = control_mode

    # Make environment
    envs = gym.make(env_id, num_envs=num_envs, **env_kwargs)
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=True)
    envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=True, auto_reset=True ,record_metrics=False)

    ############################################################################
    # 2) Initialize a ReplayBuffer (on CPU)
    ############################################################################
    rb = ReplayBuffer(
        env=envs,
        num_envs=num_envs,
        buffer_size=buffer_size,
        storage_device=torch.device("cpu"),  # CPU for storage
        sample_device=torch.device("cpu")    # CPU for sampling
    )

    ############################################################################
    # 3) Load expert policy (Actor) on CPU
    ############################################################################
    obs, _ = envs.reset(seed=0)
    checkpoint = torch.load(expert_ckpt)

    expert_actor = Actor(envs, sample_obs=obs)
    try:
        expert_actor.load_state_dict(checkpoint["actor"])
    except RuntimeError as e:
        print(f"Strict load failed with error: {e}")
        expert_actor.load_state_dict(checkpoint["actor"], strict=False)
    
    expert_actor.eval()

    ############################################################################
    # 4) Roll out with the expert and fill the buffer
    ############################################################################
    collected_steps = 0
    next_obs_buf = []
    rewards_buf = []
    dones_buf = []

    while collected_steps < total_steps:
        with torch.no_grad():
            actions = expert_actor.get_eval_action(obs)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        info_keys = list(infos.keys())
        elapsed_steps = infos['elapsed_steps'].item()
        success = infos['success'].item()

        print(f"elapsed steps: {elapsed_steps}   |   success: {success}")

        # if success:
        #     if elapsed_steps == 50:
        #         print(f" Success trial:   {elapsed_steps}  |  {success}")
        #         print("Add the data to buffer \n")

        # print(f"Rewards: {rewards.items()}    | termin")
        # print(f"termintations: {terminations.item()}")

        # store data
        real_next_obs = {k: v.clone() for k, v in next_obs.items()}

        # We do not attempt to fetch final_observation or anything similar
        rb.add(obs, real_next_obs, actions, rewards, terminations)

        obs = next_obs
        collected_steps += num_envs

        if collected_steps % 5000 == 0:
            print(f"Collected {collected_steps} steps so far...")

    print(f"Finished collecting {collected_steps} transitions from expert.")

    ############################################################################
    # 5) Save the buffer to file
    ############################################################################
    print(f"Saving replay buffer to={output_buffer_file} ...")
    torch.save(rb, output_buffer_file)
    print("Replay buffer saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="PickCube-v1")
    parser.add_argument("--obs_mode", type=str, default="rgb")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--control_mode", type=str, default="pd_ee_delta_pos")
    parser.add_argument("--expert_ckpt", type=str, default="runs/PickCube-v1__sac_rgbd__1__1738602197/ckpt_250048.pt")
    parser.add_argument("--output_buffer_file", type=str, default="expert_buffer_2.pt")
    parser.add_argument("--total_steps", type=int, default=500)
    parser.add_argument("--buffer_size", type=int, default=300000)
    args = parser.parse_args()

    collect_expert_data(
        env_id=args.env_id,
        obs_mode=args.obs_mode,
        num_envs=args.num_envs,
        control_mode=args.control_mode,
        expert_ckpt=args.expert_ckpt,
        output_buffer_file=args.output_buffer_file,
        total_steps=args.total_steps,
        buffer_size=args.buffer_size,
    )


"""

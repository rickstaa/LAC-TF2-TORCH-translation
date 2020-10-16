"""Simple script used to test the performance of a trained model."""

import os
import sys
from itertools import cycle
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse

from lac import LAC

from utils import get_env_from_name
from variant import EVAL_PARAMS, ENVS_PARAMS, ENV_NAME, ENV_SEED, REL_PATH


###################################################
# Main inference eval script ######################
###################################################
if __name__ == "__main__":

    # Parse Arguments
    parser = argparse.ArgumentParser(
        description="Evaluate the LAC agent in a given environment."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=EVAL_PARAMS["eval_list"],
        help="The name of the model you want to evaluate.",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default=ENV_NAME,
        help="The name of the env you want to evaluate.",
    )
    parser.add_argument(
        "--plot-r",
        type=bool,
        default=EVAL_PARAMS["plot_ref"],
        help="Whether want to plot the states of reference.",
    )
    parser.add_argument(
        "--plot-o",
        type=bool,
        default=EVAL_PARAMS["plot_obs"],
        help="Whether you want to plot the observations.",
    )
    parser.add_argument(
        "--plot-c",
        type=bool,
        default=EVAL_PARAMS["plot_cost"],
        help="Whether want to plot the cost.",
    )
    parser.add_argument(
        "--save-figs",
        type=bool,
        default=EVAL_PARAMS["save_figs"],
        help="Whether you want to save the figures to pdf.",
    )
    parser.add_argument(
        "--fig-file-type",
        type=str,
        default=EVAL_PARAMS["fig_file_type"],
        help="The file type you want to use for saving the figures.",
    )
    args = parser.parse_args()

    # Validate file types
    sup_file_types = ["pdf", "svg", "png", "jpg"]
    if args.fig_file_type not in sup_file_types:
        print(
            f"The requested figure save file type {args.fig_file_type} is not "
            "supported file types are {sup_file_types}."
        )
        sys.exit(0)

    # Create model path
    eval_agents = (
        [args.model_name] if not isinstance(args.model_name, list) else args.model_name
    )

    ###############################################
    # Perform robustness eval for agents ##########
    ###############################################
    print("\n=========Performing inference evaluation=========")
    for name in eval_agents:
        if REL_PATH:
            MODEL_PATH = "/".join(["./log", args.env_name, name])
            LOG_PATH = "/".join([MODEL_PATH, "figure"])
            os.makedirs(LOG_PATH, exist_ok=True)
        else:
            dirname = os.path.dirname(__file__)
            MODEL_PATH = os.path.abspath(
                os.path.join(dirname, "./log/" + args.env_name + "/" + name)
            )  # TODO: Make log paths env name lowercase
            LOG_PATH = os.path.abspath(os.path.join(MODEL_PATH, "figure"))
            os.makedirs(LOG_PATH, exist_ok=True)

        print("evaluating " + name)
        print(f"Using model folder: {MODEL_PATH}")
        print(f"In environment: {args.env_name}")

        ###########################################
        # Create environment and setup policy #####
        ###########################################

        # Create environment
        env = get_env_from_name(args.env_name, ENV_SEED)

        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            print(
                f"Shutting down robustness eval since model `{args.model_name}` was "
                f"not found for the `{args.env_name}` environment."
            )
            sys.exit(0)

        # Get environment action and observation space dimensions
        a_lowerbound = env.action_space.low
        a_upperbound = env.action_space.high
        s_dim = env.observation_space.shape[0]
        a_dim = env.action_space.shape[0]

        # Create policy
        policy = LAC(a_dim, s_dim)

        # Retrieve agents
        print("Looking for policies (rollouts)...")
        rollout_list = os.listdir(MODEL_PATH)
        rollout_list = [
            rollout_name
            for rollout_name in rollout_list
            if os.path.exists(
                os.path.abspath(MODEL_PATH + "/" + rollout_name + "/policy/checkpoint")
            )
        ]
        rollout_list = [int(item) for item in rollout_list if item.isnumeric()]
        rollout_list.sort()  # Sort rollouts_list

        # Check if model exists
        if not rollout_list:
            print(
                f"Shutting down robustness eval since no rollouts were found for model "
                f"`{args.model_name}` in the `{args.env_name}` environment."
            )
            sys.exit(0)

        ###########################################
        # Run inference ###########################
        ###########################################

        # Retrieve rollouts from variant file
        rollouts_input = EVAL_PARAMS["which_policy_for_inference"]

        # Use all rollouts if roolouts_inpuy is emptry []
        rollouts_input = rollout_list if not rollouts_input else rollouts_input

        # Validate given policies
        if any([not isinstance(x, (int, float)) for x in rollouts_input]):
            print(
                "Please provide a valid list of rollouts in the "
                "`which_policy_for_inference` variable of the variant file "
                " (example: [1, 2, 3])."
            )
            sys.exit(0)
        rollouts_input = [
            int(item) for item in rollouts_input
        ]  # convert to nr for comparison
        invalid_input_rollouts = [
            x for i, x in enumerate(rollouts_input) if not (x in rollout_list)
        ]
        if len(invalid_input_rollouts) != 0:
            rollout_str = "Rollout" if sum(invalid_input_rollouts) <= 1 else "Rollouts"
            print(
                f"Please re-check the list you supplied in the "
                "`which_policy_for_inference` variable of the variant file. "
                f"{rollout_str} {invalid_input_rollouts} do not exist."
            )
            sys.exit(0)
        rollouts_input = [str(item) for item in rollouts_input]  # convert back to str

        # TODO: Old input prompt we can remove later
        # # Get user input on which rollouts to use
        # policy_str = "policy was" if len(rollout_list) == 1 else "policies were"
        # print(
        #     f"{len(rollout_list)} {(policy_str)} found in the model folder "
        #     f"{rollout_list}."
        # )
        # while 1:

        #     # Ask user for rollouts until input is correct
        #     rollouts_input = input(
        #         "Which policies (rollouts) do you want to use (Uses all if empty): "
        #     )
        #     if rollouts_input == "":
        #         rollouts_input = [str(item) for item in rollout_list]
        #         break
        #     rollouts_input = rollouts_input.replace("[", "").replace("]", "").split(",")

        #     # Check user input
        #     invalid_input = [
        #         not item.isnumeric() for item in rollouts_input
        #     ]  # Check if numbers
        #     if any(invalid_input):
        #         print("Please provide a valid input (example: [1,2,3]).")
        #         continue
        #     rollouts_input = [
        #         int(item) for item in rollouts_input
        #     ]  # convert to nr for comparison
        #     invalid_input_rollouts = [
        #         x for i, x in enumerate(rollouts_input) if not (x in rollout_list)
        #     ]
        #     if len(invalid_input_rollouts) != 0:
        #         rollout_str = (
        #             "rollout" if sum(invalid_input_rollouts) <= 1 else "rollouts"
        #         )
        #         print(
        #             f"Please re-check your input {rollout_str} "
        #             f"{invalid_input_rollouts} do not exist."
        #         )
        #         continue
        #     rollouts_input = [
        #         str(item) for item in rollouts_input
        #     ]  # convert back to str
        #     break

        # Print used roll outs
        print(f"Using rollouts: {rollouts_input}")

        # Perform a number of paths in each rollout and store them
        roll_outs_paths = {}
        for rollout in rollouts_input:

            # Rollouts paths storage bucket
            roll_out_paths = {
                "s": [],
                "r": [],
                "s_": [],
                "state_of_interest": [],
                "reference": [],
                "episode_length": [],
                "return": [],
                "death_rate": 0.0,
            }

            # Load current rollout agent
            LAC = policy.restore(
                os.path.abspath(MODEL_PATH + "/" + rollout + "/policy")
            )
            if not LAC:
                print(
                    f"Agent {rollout} could not be loaded. Continuing to the next "
                    "agent."
                )
                continue

            # Perform a number of paths in the environment
            for i in range(
                math.ceil(EVAL_PARAMS["num_of_paths"] / len(rollouts_input))
            ):

                # Path storage buckets
                episode_path = {
                    "s": [],
                    "r": [],
                    "s_": [],
                    "state_of_interest": [],
                    "reference": [],
                }

                # env.reset() # MAke sure this is not seeded when reset
                if env.__class__.__name__.lower() == "ex3_ekf_gyro":
                    s = env.reset(eval=True)
                else:
                    s = env.reset()

                # Perfrom trail
                for j in range(ENVS_PARAMS[args.env_name]["max_ep_steps"]):

                    # Perform action in the environment
                    a = policy.choose_action(s, True)
                    action = (
                        a_lowerbound + (a + 1.0) * (a_upperbound - a_lowerbound) / 2
                    )
                    s_, r, done, info = env.step(action)

                    # Store observations
                    episode_path["s"].append(s)
                    episode_path["r"].append(r)
                    episode_path["s_"].append(s_)
                    if "state_of_interest" in info.keys():
                        episode_path["state_of_interest"].append(
                            np.array([info["state_of_interest"]])
                        )
                    if "reference" in info.keys():
                        episode_path["reference"].append(np.array(info["reference"]))

                    # Terminate if max step has been reached
                    done = False  # Ignore done signal from env because inference
                    if j == (ENVS_PARAMS[args.env_name]["max_ep_steps"] - 1):
                        done = True

                    # Update current state
                    s = s_

                    # Check if episode is done and break loop
                    if done:
                        break

                # Append paths to paths list
                roll_out_paths["s"].append(episode_path["s"])
                roll_out_paths["r"].append(episode_path["r"])
                roll_out_paths["s_"].append(episode_path["s_"])
                roll_out_paths["state_of_interest"].append(
                    episode_path["state_of_interest"]
                )
                roll_out_paths["reference"].append(episode_path["reference"])
                roll_out_paths["episode_length"].append(len(episode_path["s"]))
                roll_out_paths["return"].append(np.sum(episode_path["r"]))

            # Calculate roll_out death rate
            roll_out_paths["death_rate"] = sum(
                [
                    episode <= (ENVS_PARAMS[args.env_name]["max_ep_steps"] - 1)
                    for episode in roll_out_paths["episode_length"]
                ]
            ) / len(roll_out_paths["episode_length"])

            # Store rollout results in rollouts dictionary
            roll_outs_paths["roll_out_" + rollout] = roll_out_paths

        # Loop through rollouts display statistics and add append paths to eval dict
        eval_paths = {}
        roll_outs_diag = {}
        for roll_out, roll_out_val in roll_outs_paths.items():

            # Calculate rollouts statistics
            roll_outs_diag[roll_out] = {}
            roll_outs_diag[roll_out]["mean_return"] = np.mean(roll_out_val["return"])
            roll_outs_diag[roll_out]["mean_episode_length"] = np.mean(
                roll_out_val["episode_length"]
            )
            roll_outs_diag[roll_out]["death_rate"] = roll_out_val.pop("death_rate")

            # concatenate rollout to eval dictionary
            for key, val in roll_out_val.items():
                if key not in eval_paths.keys():
                    eval_paths[key] = val
                else:
                    eval_paths[key].extend(val)

        ###########################################
        # Display path diagnostics ################
        ###########################################

        # Display rollouts diagnostics
        print("\n==Rollouts Diagnostics==")
        eval_diagnostics = {}
        for roll_out, roll_out_diagnostics_val in roll_outs_diag.items():
            print(f"{roll_out}:")
            for key, val in roll_out_diagnostics_val.items():
                print(f"- {key}: {val}")
                if key not in eval_diagnostics:
                    eval_diagnostics[key] = [val]
                else:
                    eval_diagnostics[key].append(val)
            # print("")
        print("all_roll_outs:")
        for key, val in eval_diagnostics.items():
            print(f"- {key}: {np.mean(val)}")
            print(f"- {key}_std: {np.std(val)}")

        ###########################################
        # Plot mean paths #########################
        ###########################################

        # Plot mean path of reference and state_of_interrest
        if args.plot_r:
            print("Plotting states of reference...")
            print("Plotting mean path and standard deviation...")

            # Retrieve requested sates list
            req_ref = EVAL_PARAMS["ref"]

            # Calculate mean path of reference and state_of_interest
            soi_trimmed = [
                path
                for path in eval_paths["state_of_interest"]
                if len(path) == max(eval_paths["episode_length"])
            ]  # Needed because unequal paths # FIXME: CLEANUP
            ref_trimmed = [
                path
                for path in eval_paths["reference"]
                if len(path) == max(eval_paths["episode_length"])
            ]  # Needed because unequal paths # FIXME: CLEANUP
            soi_mean_path = np.transpose(
                np.squeeze(np.mean(np.array(soi_trimmed), axis=0))
            )
            soi_std_path = np.transpose(
                np.squeeze(np.std(np.array(soi_trimmed), axis=0))
            )
            ref_mean_path = np.transpose(
                np.squeeze(np.mean(np.array(ref_trimmed), axis=0))
            )
            ref_std_path = np.transpose(
                np.squeeze(np.std(np.array(ref_trimmed), axis=0))
            )

            # Make sure arrays are right dimension
            soi_mean_path = (
                np.expand_dims(soi_mean_path, axis=0)
                if len(soi_mean_path.shape) == 1
                else soi_mean_path
            )
            soi_std_path = (
                np.expand_dims(soi_std_path, axis=0)
                if len(soi_std_path.shape) == 1
                else soi_std_path
            )
            ref_mean_path = (
                np.expand_dims(ref_mean_path, axis=0)
                if len(ref_mean_path.shape) == 1
                else ref_mean_path
            )
            ref_std_path = (
                np.expand_dims(ref_std_path, axis=0)
                if len(ref_std_path.shape) == 1
                else ref_std_path
            )

            # Check if requested state_of interest exists
            ref_str = (
                req_ref if req_ref else list(range(1, (soi_mean_path.shape[0] + 1)))
            )
            print(f"Plotting results for states of reference {ref_str}.")
            invalid_refs = [
                ref for ref in req_ref if (ref > soi_mean_path.shape[0] or ref < 0)
            ]
            if invalid_refs:
                for ref in invalid_refs:
                    print(
                        f":WARNING: Sate of intrest and/or reference {ref} could not "
                        "be ploted as it does not exist."
                    )

            # Plot mean path of reference and state_of_interrest
            if EVAL_PARAMS["merged"]:
                fig_1 = plt.figure(figsize=(9, 6), num=f"LAC_TF115_1")
                ax = fig_1.add_subplot(111)
                colors = "bgrcmk"
                cycol = cycle(colors)
            for i in range(0, max(soi_mean_path.shape[0], ref_mean_path.shape[0])):
                if (i + 1) in req_ref or not req_ref:
                    if not EVAL_PARAMS["merged"]:
                        fig_1 = plt.figure(figsize=(9, 6), num=f"LAC_TF115_{i+1}",)
                        ax = fig_1.add_subplot(111)
                        color1 = "red"
                        color2 = "blue"
                    else:
                        color1 = next(cycol)
                        color2 = color1
                    t = range(max(eval_paths["episode_length"]))
                    if i <= (len(soi_mean_path) - 1):
                        ax.plot(
                            t,
                            soi_mean_path[i],
                            color=color1,
                            linestyle="dashed",
                            label=f"state_of_interest_{i+1}_mean",
                        )
                        if not EVAL_PARAMS["merged"]:
                            ax.set_title(f"States of interest and reference {i+1}")
                        ax.fill_between(
                            t,
                            soi_mean_path[i] - soi_std_path[i],
                            soi_mean_path[i] + soi_std_path[i],
                            color=color1,
                            alpha=0.3,
                            label=f"state_of_interest_{i+1}_std",
                        )
                    if i <= (len(ref_mean_path) - 1):
                        ax.plot(
                            t,
                            ref_mean_path[i],
                            color=color2,
                            # label=f"reference_{i+1}",
                        )
                        # ax.fill_between(
                        #     t,
                        #     ref_mean_path[i] - ref_std_path[i],
                        #     ref_mean_path[i] + ref_std_path[i],
                        #     color=color2,
                        #     alpha=0.3,
                        #     label=f"reference_{i+1}_std",
                        # )  # FIXME: remove
                    if not EVAL_PARAMS["merged"]:
                        handles, labels = ax.get_legend_handles_labels()
                        ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
                if EVAL_PARAMS["merged"]:
                    ax.set_title("True and Estimated Quatonian")
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)

        # Also plot mean and std of the observations
        if args.plot_o:
            print("Plotting observations...")
            print("Plotting mean path and standard deviation...")

            # Retrieve requested obs list
            req_obs = EVAL_PARAMS["obs"]

            # Create figure
            # BUG: Doesn't work with non merged option
            fig_2 = plt.figure(figsize=(9, 6), num="LAC_TF115_2")
            colors = "bgrcmk"
            cycol = cycle(colors)
            ax2 = fig_2.add_subplot(111)

            # Calculate mean observation path and std
            obs_trimmed = [
                path
                for path in eval_paths["s"]
                if len(path) == max(eval_paths["episode_length"])
            ]
            obs_mean_path = np.transpose(
                np.squeeze(np.mean(np.array(obs_trimmed), axis=0))
            )
            obs_std_path = np.transpose(
                np.squeeze(np.std(np.array(obs_trimmed), axis=0))
            )
            t = range(max(eval_paths["episode_length"]))

            # Check if requested observation exists
            obs_str = (
                req_ref if req_ref else list(range(1, (obs_mean_path.shape[0] + 1)))
            )
            print(f"Plotting results for obs {obs_str}.")
            invalid_obs = [
                obs for obs in req_obs if (obs > obs_mean_path.shape[0] or obs < 0)
            ]
            if invalid_obs:
                for obs in invalid_obs:
                    print(
                        f":WARNING: Observation {obs} could not be ploted as it does "
                        "not exist."
                    )

            # Plot state paths and std
            for i in range(0, obs_mean_path.shape[0]):
                if (i + 1) in req_obs or not req_obs:
                    color = next(cycol)
                    ax2.plot(
                        t,
                        obs_mean_path[i],
                        color=color,
                        linestyle="dashed",
                        label=(f"s_{i+1}"),
                    )
                    ax2.fill_between(
                        t,
                        obs_mean_path[i] - obs_std_path[i],
                        obs_mean_path[i] + obs_std_path[i],
                        color=color,
                        alpha=0.3,
                        label=(f"s_{i+1}_std"),
                    )
            ax2.set_title("Observations")
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(handles2, labels2, loc=2, fancybox=False, shadow=False)

        # Plot mean cost and std
        if args.plot_c:
            print("Plotting cost...")
            print("Plotting mean path and standard deviation...")

            # Create figure
            fig_3 = plt.figure(figsize=(9, 6), num="LAC_TF115_3")
            ax3 = fig_3.add_subplot(111)

            # Calculate mean observation path and std
            cost_trimmed = [
                path
                for path in eval_paths["r"]
                if len(path) == max(eval_paths["episode_length"])
            ]
            cost_mean_path = np.transpose(
                np.squeeze(np.mean(np.array(cost_trimmed), axis=0))
            )
            cost_std_path = np.transpose(
                np.squeeze(np.std(np.array(cost_trimmed), axis=0))
            )
            t = range(max(eval_paths["episode_length"]))

            # Plot state paths and std
            ax3.plot(
                t, cost_mean_path, color="g", linestyle="dashed", label=("mean cost"),
            )
            ax3.fill_between(
                t,
                cost_mean_path - cost_std_path,
                cost_mean_path + cost_std_path,
                color="g",
                alpha=0.3,
                label=("mean cost std"),
            )
            ax3.set_title("Mean cost")
            handles3, labels3 = ax3.get_legend_handles_labels()
            ax3.legend(handles3, labels3, loc=2, fancybox=False, shadow=False)

        # Show figures
        plt.show()

        # Save figures to pdf if requested
        if args.save_figs:
            fig_1.savefig(
                os.path.join(LOG_PATH, "Quatonian." + args.fig_file_type),
                bbox_inches="tight",
            )
            fig_2.savefig(
                os.path.join(LOG_PATH, "State." + args.fig_file_type),
                bbox_inches="tight",
            )
            fig_3.savefig(
                os.path.join(LOG_PATH, "Cost." + args.fig_file_type),
                bbox_inches="tight",
            )

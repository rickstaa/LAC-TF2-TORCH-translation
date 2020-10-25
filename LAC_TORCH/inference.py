"""Script that can be used to display the performance and robustness of a trained
agent."""

import os
import sys
from itertools import cycle
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
import itertools

from lac import LAC

from utils import get_env_from_name, colorize
from variant import EVAL_PARAMS, ENVS_PARAMS, ENV_NAME, ENV_SEED, REL_PATH

# IMPROVEMENT: Add render evaluation option -> inference_type=["render", "plot"]


def validate_req_sio(req_sio, soi_mean_path, ref_mean_path):
    """Validates whether the requested states of interest exists. Throws a warning
    message if a requested state is not found.

    Args:
        req_sio (list): The requested states of interest.
        soi_mean_path (numpy.ndarray): The state of interest mean paths.
        ref_mean_path (numpy.ndarray): The reference mean paths.

    Returns:
        tuple: Lists whith the requested states of interest and reference which are
        valid.
    """

    # Check if the requested states of interest are present
    req_sio = (
        req_sio if req_sio else list(range(1, (soi_mean_path.shape[0] + 1)))
    )  # Get req sio (Use all if empty)
    N_sio = soi_mean_path.squeeze().shape[0] if soi_mean_path.squeeze().ndim > 1 else 1
    N_refs = ref_mean_path.squeeze().shape[0] if ref_mean_path.squeeze().ndim > 1 else 1
    invalid_sio = [sio for sio in req_sio if (sio > N_sio or sio < 1)]
    invalid_refs = [ref for ref in req_sio if (ref > N_refs or ref < 1)]

    # Display warning if not found
    if invalid_sio and invalid_refs:
        warning_str = (
            "{} {} and {} {}".format(
                "states of interest" if len(invalid_sio) > 1 else "state of interest",
                invalid_sio,
                "references" if len(invalid_refs) > 1 else "reference",
                invalid_refs,
            )
            + " could not be plotted as they did not exist."
        )
        print(colorize("WARN: " + warning_str.capitalize(), "yellow"))
    elif invalid_sio:
        warning_str = "WARN: {} {}".format(
            "States of interest" if len(invalid_sio) > 1 else "State of interest",
            invalid_sio,
        ) + " could not be plotted as {} does not exist.".format(
            "they" if len(invalid_sio) > 1 else "it",
        )
        print(colorize(warning_str, "yellow"))
    elif invalid_refs:
        warning_str = "WARN: {} {}".format(
            "References" if len(invalid_refs) > 1 else "Reference", invalid_refs,
        ) + " {} not be plotted as {} does not exist.".format(
            "could" if len(invalid_refs) > 1 else "can",
            "they" if len(invalid_refs) > 1 else "it",
        )
        print(colorize(warning_str, "yellow"))

    # Retrieve valid states of interest and references
    valid_sio = list(set(invalid_sio) ^ set(req_sio))
    valid_refs = list(set(invalid_refs) ^ set(req_sio))

    # Return valid states of interest and references
    return valid_sio, valid_refs


def validate_req_obs(req_obs, obs_mean_path):
    """Validates whether the requested observations exists. Throws a warning
    message if a requested state is not found.

    Args:
        req_obs (list): The requested observations.
        obs_mean_path (numpy.ndarray): The mean paths of the observations.

    Returns:
        list: The requested observations which are valid.
    """

    # Check if the requested observations are present
    req_obs = (
        req_obs if req_obs else list(range(1, (obs_mean_path.shape[0] + 1)))
    )  # Get req sio (Use all if empty)
    invalid_obs = [obs for obs in req_obs if (obs > obs_mean_path.shape[0] or obs < 1)]
    if invalid_obs:
        warning_str = (
            "WARN: {} {}".format(
                "Observations" if len(invalid_obs) > 1 else "Observations", invalid_obs,
            )
            + " could not be plotted as they does not exist."
        )
        print(colorize(warning_str, "yellow"))
    valid_obs = list(set(invalid_obs) ^ set(range(1, len(obs_mean_path))))

    # Return valid observations
    return valid_obs


if __name__ == "__main__":

    # Parse cmdline arguments
    parser = argparse.ArgumentParser(
        description="Evaluate trained the LAC agents in a given environment."
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
        "--plot-s",
        type=bool,
        default=EVAL_PARAMS["plot_soi"],
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
    args = parser.parse_args()

    # Validate specified figure output file type
    sup_file_types = ["pdf", "svg", "png", "jpg"]
    if EVAL_PARAMS["fig_file_type"] not in sup_file_types:
        file_Type = EVAL_PARAMS["fig_file_type"]
        print(
            f"The requested figure save file type {file_Type} "
            "is not supported file types are {sup_file_types}."
        )
        sys.exit(0)

    # Retrieve available policies
    eval_agents = (
        [args.model_name] if not isinstance(args.model_name, list) else args.model_name
    )

    ####################################################
    # Perform Inference for all agents #################
    ####################################################
    print("\n=========Performing inference evaluation=========")
    print(f"Evaluationing agents: {eval_agents}")

    # Loop though USER defined agents list
    for name in eval_agents:

        # Create agent policy and log paths
        if REL_PATH:
            MODEL_PATH = "/".join(["./log", args.env_name.lower(), name])
            LOG_PATH = "/".join([MODEL_PATH, "figure"])
            os.makedirs(LOG_PATH, exist_ok=True)
        else:
            dirname = os.path.dirname(__file__)
            MODEL_PATH = os.path.abspath(
                os.path.join(dirname, "./log/" + args.env_name.lower() + "/" + name)
            )
            LOG_PATH = os.path.abspath(os.path.join(MODEL_PATH, "figure"))
            os.makedirs(LOG_PATH, exist_ok=True)
        print("\n====Evaluation agent " + name + "====")
        print(f"Using model folder: {MODEL_PATH}")

        # Create environment
        print(f"In environment: {args.env_name}")
        env = get_env_from_name(args.env_name, ENV_SEED)

        # Check if specified agent exists
        if not os.path.exists(MODEL_PATH):
            warning_str = (
                f"WARN: Inference could not be run for agent `{name}` as it was not "
                f"found for the `{args.env_name}` environment."
            )
            print(colorize(warning_str, "yellow"))
            continue

        # Get environment action and observation space dimensions
        a_lowerbound = env.action_space.low
        a_upperbound = env.action_space.high
        s_dim = env.observation_space.shape[0]
        a_dim = env.action_space.shape[0]

        # Initiate the LAC policy
        policy = LAC(a_dim, s_dim)

        # Retrieve all trained policies (rollouts) for a given agent
        print("Looking for policies (rollouts)...")
        rollout_list = os.listdir(MODEL_PATH)
        rollout_list = [
            rollout_name
            for rollout_name in rollout_list
            if os.path.exists(
                os.path.abspath(MODEL_PATH + "/" + rollout_name + "/policy/model.pth")
            )
        ]
        rollout_list = [int(item) for item in rollout_list if item.isnumeric()]
        rollout_list.sort()  # Sort rollouts_list

        # Check if a given policy (rollout) exists
        if not rollout_list:
            warning_str = (
                f"Shutting down robustness eval since no rollouts were found for model "
                f"`{args.model_name}` in the `{args.env_name}` environment."
            )
            warning_str = (
                f"WARN: Inference could not be run for agent `{name}` as no rollouts "
                " were found."
            )
            print(colorize(warning_str, "yellow"))
            continue

        # Retrieve USER defined rollouts list
        rollouts_input = EVAL_PARAMS["which_policy_for_inference"]
        rollouts_input = rollout_list if not rollouts_input else rollouts_input

        # Process requested rollouts input
        # Here we check if they exist and also convert them to the
        # right format.
        if any([not isinstance(x, (int, float)) for x in rollouts_input]):
            print(
                "Please provide a valid list of rollouts in the "
                "`which_policy_for_inference` variable of the variant file "
                " (example: [1, 2, 3])."
            )
            sys.exit(0)
        rollouts_input = [int(item) for item in rollouts_input]
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
        rollouts_input = [str(item) for item in rollouts_input]
        print(f"Using rollouts: {rollouts_input}")

        ############################################
        # Run policy inference #####################
        ############################################
        # Perform a number of paths in each rollout and store them.
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

            # Load current rollout policy
            retval = policy.restore(
                os.path.abspath(MODEL_PATH + "/" + rollout + "/policy")
            )
            if not retval:
                print(
                    f"Policy {rollout} could not be loaded. Continuing to the next "
                    "policy."
                )
                continue

            # Perform a number of paths in the environment
            for i in range(
                math.ceil(EVAL_PARAMS["num_of_paths"] / len(rollouts_input))
            ):

                # Path storage bucket
                episode_path = {
                    "s": [],
                    "r": [],
                    "s_": [],
                    "state_of_interest": [],
                    "reference": [],
                }

                # Reset environment
                # NOTE (rickstaa): This check was added since some of the supported
                # environments have a different reset when running the inference.
                # TODO: Add these environments in a config file!
                if env.__class__.__name__.lower() == "ex3_ekf_gyro":
                    s = env.reset(eval=True)
                else:
                    s = env.reset()

                # Retrieve path
                for j in range(ENVS_PARAMS[args.env_name]["max_ep_steps"]):

                    # Perform action in the environment
                    a = policy.choose_action(s, True)
                    action = (
                        a_lowerbound + (a + 1.0) * (a_upperbound - a_lowerbound) / 2
                    )
                    s_, r, done, info = env.step(action)

                    # Store observations in path
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

                # Append path to rollout paths list
                roll_out_paths["s"].append(episode_path["s"])
                roll_out_paths["r"].append(episode_path["r"])
                roll_out_paths["s_"].append(episode_path["s_"])
                roll_out_paths["state_of_interest"].append(
                    episode_path["state_of_interest"]
                )
                roll_out_paths["reference"].append(episode_path["reference"])
                roll_out_paths["episode_length"].append(len(episode_path["s"]))
                roll_out_paths["return"].append(np.sum(episode_path["r"]))

            # Calculate rollout death rate
            roll_out_paths["death_rate"] = sum(
                [
                    episode <= (ENVS_PARAMS[args.env_name]["max_ep_steps"] - 1)
                    for episode in roll_out_paths["episode_length"]
                ]
            ) / len(roll_out_paths["episode_length"])

            # Store rollout results in rollout dictionary
            roll_outs_paths["roll_out_" + rollout] = roll_out_paths

        ############################################
        # Calculate rollout statistics #############
        ############################################
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

        ############################################
        # Display rollout diagnostics ##############
        ############################################

        # Display rollouts diagnostics
        print("Printing rollouts diagnostics...")
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
            print("")
        print("all_roll_outs:")
        for key, val in eval_diagnostics.items():
            print(f" - {key}: {np.mean(val)}")
            print(f" - {key}_std: {np.std(val)}")

        ############################################
        # Display performance figures ##############
        ############################################

        ####################################
        # Plot mean path and std for #######
        # states of interest and ###########
        # references. ######################
        ####################################
        print("\n==Rollouts inference plots==")
        figs = {
            "states_of_interest": [],
            "states": [],
            "costs": [],
        }  # Store all figers (Needed for save)
        if args.plot_s:
            print("Plotting states of interest mean path and standard deviation...")

            # Retrieve USER defined sates of reference list
            req_sio = EVAL_PARAMS["soi"]

            # Calculate mean path of the state of interest
            soi_trimmed = [
                path
                for path in eval_paths["state_of_interest"]
                if len(path) == max(eval_paths["episode_length"])
            ]  # Trim unfinished paths
            soi_mean_path = np.transpose(
                np.squeeze(np.mean(np.array(soi_trimmed), axis=0))
            )
            soi_std_path = np.transpose(
                np.squeeze(np.std(np.array(soi_trimmed), axis=0))
            )

            # Calculate mean path of the of the reference
            ref_trimmed = [
                path
                for path in eval_paths["reference"]
                if len(path) == max(eval_paths["episode_length"])
            ]  # Trim unfinished paths
            ref_mean_path = np.transpose(
                np.squeeze(np.mean(np.array(ref_trimmed), axis=0))
            )
            ref_std_path = np.transpose(
                np.squeeze(np.std(np.array(ref_trimmed), axis=0))
            )

            # Make sure mean path and std arrays are the right shape
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

            # Check if requested state_of_interest exists
            valid_sio, valid_refs = validate_req_sio(
                req_sio, soi_mean_path, ref_mean_path
            )

            # Plot mean path and std for states of interest and references
            if valid_sio or valid_refs:  # Check if any sio or refs were found
                print(
                    "Using: {}".format(
                        (
                            "states of interest "
                            if len(valid_sio) > 1
                            else "state of interest "
                        )
                        + str(valid_sio)
                        + (" and " if valid_sio and valid_refs else "")
                        + ("references " if len(valid_refs) > 1 else "reference ")
                        + str(valid_refs)
                    )
                )

                # Plot sio/ref mean path and std
                if EVAL_PARAMS["sio_merged"]:  # Add all soi in one figure
                    fig = plt.figure(
                        figsize=(9, 6),
                        num=(
                            "LAC_TORCH_"
                            + str(len(list(itertools.chain(*figs.values()))) + 1)
                        ),
                    )
                    ax = fig.add_subplot(111)
                    colors = "bgrcmk"
                    cycol = cycle(colors)
                    figs["states_of_interest"].append(fig)  # Store figure reference
                for i in range(0, max(soi_mean_path.shape[0], ref_mean_path.shape[0])):
                    if (i + 1) in req_sio or not req_sio:
                        if not EVAL_PARAMS[
                            "sio_merged"
                        ]:  # Create separate figs for each sio
                            fig = plt.figure(
                                figsize=(9, 6),
                                num=(
                                    "LAC_TORCH_"
                                    + str(
                                        len(list(itertools.chain(*figs.values()))) + 1
                                    )
                                ),
                            )
                            ax = fig.add_subplot(111)
                            color1 = "red"
                            color2 = "blue"
                            figs["states_of_interest"].append(
                                fig
                            )  # Store figure reference
                        else:
                            color1 = color2 = next(cycol)
                        t = range(max(eval_paths["episode_length"]))

                        # Plot states of interest
                        if i <= (len(soi_mean_path) - 1):
                            ax.plot(
                                t,
                                soi_mean_path[i],
                                color=color1,
                                linestyle="dashed",
                                label=f"state_of_interest_{i+1}_mean",
                            )
                            if not EVAL_PARAMS["sio_merged"]:
                                ax.set_title(f"States of interest and reference {i+1}")
                            ax.fill_between(
                                t,
                                soi_mean_path[i] - soi_std_path[i],
                                soi_mean_path[i] + soi_std_path[i],
                                color=color1,
                                alpha=0.3,
                                label=f"state_of_interest_{i+1}_std",
                            )

                        # Plot references
                        if i <= (len(ref_mean_path) - 1):
                            ax.plot(
                                t,
                                ref_mean_path[i],
                                color=color2,
                                label=f"reference_{i+1}",
                            )
                            # ax.fill_between(
                            #     t,
                            #     ref_mean_path[i] - ref_std_path[i],
                            #     ref_mean_path[i] + ref_std_path[i],
                            #     color=color2,
                            #     alpha=0.3,
                            #     label=f"reference_{i+1}_std",
                            # )  # Should be zero

                        # Add figure legend (Separate figures)
                        if not EVAL_PARAMS["sio_merged"]:
                            handles, labels = ax.get_legend_handles_labels()
                            ax.legend(
                                handles, labels, loc=2, fancybox=False, shadow=False
                            )

                    # Add figure legend and title (merged figure)
                    if EVAL_PARAMS["sio_merged"]:
                        ax.set_title("True and Estimated Quatonian")
                        handles, labels = ax.get_legend_handles_labels()
                        ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
            else:
                print(
                    colorize(
                        "WARN: No states of interest or references were found.",
                        "yellow",
                    )
                )

        ####################################
        # Plot mean path and std for #######
        # the observations #################
        ####################################
        if args.plot_o:
            print("Plotting observations mean path and standard deviation...")

            # Retrieve USER defined observations list
            req_obs = EVAL_PARAMS["obs"]

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

            # Check if USER requested observation exists
            valid_obs = validate_req_obs(req_obs, obs_mean_path)

            # Plot mean observations path and std
            print(f"Using observations {valid_obs}...")
            if valid_obs:  # Check if any sio or refs were found
                if EVAL_PARAMS["obs_merged"]:
                    fig = plt.figure(
                        figsize=(9, 6),
                        num=(
                            "LAC_TORCH_"
                            + str(len(list(itertools.chain(*figs.values()))) + 1)
                        ),
                    )
                    colors = "bgrcmk"
                    cycol = cycle(colors)
                    ax2 = fig.add_subplot(111)
                    figs["states"].append(fig)  # Store figure reference
                for i in range(0, obs_mean_path.shape[0]):
                    if (i + 1) in req_obs or not req_obs:
                        if not EVAL_PARAMS[
                            "obs_merged"
                        ]:  # Create separate figs for each sio
                            fig = plt.figure(
                                figsize=(9, 6),
                                num=(
                                    "LAC_TORCH_"
                                    + str(
                                        len(list(itertools.chain(*figs.values()))) + 1
                                    )
                                ),
                            )
                            ax2 = fig.add_subplot(111)
                            color = "blue"
                            figs["states"].append(fig)  # Store figure reference
                        else:
                            color = next(cycol)
                        ax2.plot(
                            t,
                            obs_mean_path[i],
                            color=color,
                            linestyle="dashed",
                            label=(f"s_{i+1}"),
                        )
                        if not EVAL_PARAMS["obs_merged"]:
                            ax2.set_title(f"Observation {i+1}")
                        ax2.fill_between(
                            t,
                            obs_mean_path[i] - obs_std_path[i],
                            obs_mean_path[i] + obs_std_path[i],
                            color=color,
                            alpha=0.3,
                            label=(f"s_{i+1}_std"),
                        )

                        # Add figure legend (Separate figures)
                        if not EVAL_PARAMS["obs_merged"]:
                            handles2, labels2 = ax2.get_legend_handles_labels()
                            ax2.legend(
                                handles2, labels2, loc=2, fancybox=False, shadow=False
                            )

                    # Add figure legend and title (merged figure)
                    if EVAL_PARAMS["obs_merged"]:
                        ax2.set_title("Observations")
                        handles2, labels2 = ax2.get_legend_handles_labels()
                        ax2.legend(
                            handles2, labels2, loc=2, fancybox=False, shadow=False
                        )
            else:
                print(colorize("WARN: No observations were found.", "yellow"))

        ####################################
        # Plot mean cost and std for #######
        # the observations #################
        ####################################
        if args.plot_c:
            print("Plotting mean cost and standard deviation...")

            # Create figure
            fig = plt.figure(
                figsize=(9, 6),
                num=(
                    "LAC_TORCH_" + str(len(list(itertools.chain(*figs.values()))) + 1)
                ),
            )
            ax3 = fig.add_subplot(111)
            figs["costs"].append(fig)  # Store figure reference

            # Calculate mean cost and std
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

            # Plot mean cost and std
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

        # Save figures if requested
        print("Saving plots...")
        print(f"Save path: {LOG_PATH}")
        if args.save_figs:
            for index, fig in enumerate(figs["states_of_interest"]):
                save_path = (
                    os.path.join(
                        LOG_PATH,
                        "Quatonian_"
                        + str(index + 1)
                        + "."
                        + EVAL_PARAMS["fig_file_type"],
                    )
                    if not EVAL_PARAMS["sio_merged"]
                    else os.path.join(
                        LOG_PATH, "Quatonians" + "." + EVAL_PARAMS["fig_file_type"],
                    )
                )
                fig.savefig(
                    save_path, bbox_inches="tight",
                )
            for index, fig in enumerate(figs["states"]):
                save_path = (
                    os.path.join(
                        LOG_PATH,
                        "State_" + str(index + 1) + "." + EVAL_PARAMS["fig_file_type"],
                    )
                    if not EVAL_PARAMS["obs_merged"]
                    else os.path.join(
                        LOG_PATH, "States" + "." + EVAL_PARAMS["fig_file_type"],
                    )
                )
                fig.savefig(
                    save_path, bbox_inches="tight",
                )
            for index, fig in enumerate(figs["costs"]):
                fig.savefig(
                    os.path.join(
                        LOG_PATH, "Cost" + "." + EVAL_PARAMS["fig_file_type"],
                    ),
                    bbox_inches="tight",
                )

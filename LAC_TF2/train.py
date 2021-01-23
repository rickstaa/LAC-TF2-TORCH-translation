"""Start the LAC agent training."""

from variant import TRAIN_PARAMS
from lac import train
from utils import get_log_path

if __name__ == "__main__":

    # Get log path
    log_path = get_log_path(new_log_path=True)

    # Train several agents in the environment and save the results
    for i in range(
        TRAIN_PARAMS["start_of_trial"],
        TRAIN_PARAMS["start_of_trial"] + TRAIN_PARAMS["num_of_policies"],
    ):
        roll_out_log_path = log_path + "/" + str(i)
        train(roll_out_log_path)

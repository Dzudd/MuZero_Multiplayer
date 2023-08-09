from abc import ABC, abstractmethod
from typing import Literal
import os


class AbstractConfig(ABC):

    @abstractmethod
    def __init__(self):
        # More information is available here:
        # https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        # Seed for numpy, torch and the game
        self.seed = int
        # Fix the maximum number of GPUs to use.
        # It's usually faster to use a single GPU (set it to 1) if it has enough memory.
        # None will use every GPUs available
        self.max_num_gpus = int | None
        # Dimensions of the game observation, must be 3D (channel, height, width).
        # For a 1D array, please reshape it to (1, 1, length of array) e.g (3,3,3)
        self.observation_shape = tuple[int]
        # Fixed list of all possible actions e.g. list(range(9))
        self.action_space = list[int]
        # List of players. You should only edit the length e.g. list(range(2))
        self.players = list[int]
        # Number of previous observations and previous actions
        # to add to the current observation
        self.stacked_observations = int
        # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.muzero_player = int
        # Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        # It doesn't influence training.
        # None, "random" or "expert" if implemented in the Game class
        self.opponent = None | str
        # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.num_workers = int
        # gpu or cpu
        self.selfplay_on_gpu = bool
        # Maximum number of moves if game is not finished before
        self.max_moves = int
        # Number of future moves self-simulated
        self.num_simulations = int
        # Chronological discount of the reward
        self.discount = int

        # Number of moves before dropping the temperature given by
        # visit_softmax_temperature_fn to 0 (ie selecting the best action).
        # If None, visit_softmax_temperature_fn is used every time
        self.temperature_threshold = None | int

        # Root prior exploration noise
        self.root_dirichlet_alpha = float
        self.root_exploration_fraction = float

        # UCB formula
        self.pb_c_base = int
        self.pb_c_init = float

        # tbd
        self.network = Literal["resnet", "fullyconnected"]  # "resnet" / "fullyconnected"

        # Value and reward are scaled (with almost sqrt) and encoded on a vector with
        # a range of -support_size to support_size.
        # Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        self.support_size = int
        # Downsample observations before representation network,
        # False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.downsample = bool
        # Number of blocks in the ResNet
        self.blocks = int
        # Number of channels in the ResNet
        self.channels = int

        # Number of channels in reward head
        self.reduced_channels_reward = int
        # Number of channels in value head
        self.reduced_channels_value = int
        # Number of channels in policy head
        self.reduced_channels_policy = int
        # Define the hidden layers in the reward head of the dynamic network e.g [8]
        self.resnet_fc_reward_layers = list[int]
        # Define the hidden layers in the value head of the prediction network e.g [8]
        self.resnet_fc_value_layers = list[int]
        # Define the hidden layers in the policy head of the prediction network e.g [8]
        self.resnet_fc_policy_layers = list[int]

        # FULLY CONNECTED

        self.encoding_size = int
        # Define the hidden layers in the representation network e.g []
        self.fc_representation_layers = list[int]
        # Define the hidden layers in the dynamics network e.g [16]
        self.fc_dynamics_layers = list[int]
        # Define the hidden layers in the reward network e.g [16]
        self.fc_reward_layers = list[int]
        # Define the hidden layers in the value network e.g [2]
        self.fc_value_layers = list[int]
        # Define the hidden layers in the policy network e.g []
        self.fc_policy_layers = list[int]

        # Path to store the model weights and TensorBoard logs
        self.results_path = os.PathLike
        # Save the checkpoint in results_path as model.checkpoint
        self.save_model = bool
        # Total number of training steps (ie weights update according to a batch)
        self.training_steps = int
        # Number of parts of games to train on at each training step
        self.batch_size = int
        # Number of training steps before using the model for self-playing
        self.checkpoint_interval = int
        # Scale the value loss to avoid over fitting of the value function,
        # paper recommends 0.25 (See paper appendix Reanalyze)
        self.value_loss_weight = float
        # Train on GPU if available. best is to
        # use torch.cuda.is_available()
        self.train_on_gpu = bool

        # "Adam" or "SGD". Paper uses SGD
        self.optimizer = Literal["Adam", "SGD"]
        # L2 weights regularization
        # e.g. 1e-4
        self.weight_decay = float
        # Used only if optimizer is SGD
        # e.g 0.9
        self.momentum = float

        # Exponential learning rate schedule
        # Initial learning rate
        # e.g 0.003
        self.lr_init = float
        # Set it to 1 to use a constant learning rate
        self.lr_decay_rate = int
        # tbd
        # e.g. 10000
        self.lr_decay_steps = int

        # Number of self-play games to keep in the replay buffer
        # e.g. 3000
        self.replay_buffer_size = int
        # Number of game moves to keep for every batch element
        # e.g. 20
        self.num_unroll_steps = int
        # Number of steps in the future to take into account
        # for calculating the target value
        # e.g. 20
        self.td_steps = int
        # Prioritized Replay (See paper appendix Training), select in priority
        # the elements in the replay buffer which are unexpected for the network
        self.PER = bool
        # How much prioritization is used,
        # 0 corresponding to the uniform case, paper suggests 1
        self.PER_alpha = bool

        # Reanalyze (See paper appendix Reanalyse)
        # Use the last model to provide a fresher, stable n-step value
        self.use_last_model_value = bool
        self.reanalyse_on_gpu = bool

        # Adjust the self play / training ratio to avoid over/underfitting
        # Number of seconds to wait after each played game
        self.self_play_delay = int
        # Number of seconds to wait after each training step
        self.training_delay = int
        # Desired training steps per self played step ratio.
        # Equivalent to a synchronous version, training can take much longer.
        self.ratio = None | float

    @abstractmethod
    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that
        the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action
        (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class AbstractGame(ABC):
    """
    Inherit this class for muzero to play
    """

    @abstractmethod
    def __init__(self, seed=None):
        pass

    @abstractmethod
    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        pass

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return 0

    @abstractmethod
    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available,
        it can return the whole action space. At each turn, the game
        have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long,
        the idea is to define the legal actions equal to the action space
        but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        pass

    def close(self):
        """
        Properly close the game.
        """
        pass

    @abstractmethod
    def render(self):
        """
        Display the game observation.
        """
        pass

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        choice = input(f"Enter the action to play for the player {self.to_play()}: ")
        while int(choice) not in self.legal_actions():
            choice = input("Illegal action. Enter another action : ")
        return int(choice)

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        raise NotImplementedError

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return str(action_number)

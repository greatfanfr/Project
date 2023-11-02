import torch
import os
import h5py
from flearn.users.user_scaffold import UserSCAFFOLD
from flearn.servers.server_base import Server
from utils.model_utils import read_data, read_user_data, read_data_cross_validation
from scipy.stats import rayleigh
import numpy as np

# Implementation for FedVAR Server
class FedVRA(Server):
    def __init__(self, dataset, algorithm, model, nb_users, nb_samples, user_ratio, sample_ratio, L,
                 local_learning_rate, max_norm, num_glob_iters, local_updates, users_per_round, similarity, noise,
                 times, dp, sigma_gaussian, alpha, beta, number, dim_pca, use_cuda, warm_start, k_fold=None,
                 nb_fold=None):

        if similarity is None:
            similarity = (alpha, beta)

        if alpha < 0 and beta < 0:
            similarity = "iid"


    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")
            # loss_ = 0

            # Each user gets the global parameters
            self.send_parameters()

            # Evaluate model at each iteration
            self.evaluate()

            # Users are selected
            if self.noise:
                self.selected_users = self.select_transmitting_users()
                print(f"Transmitting {len(self.selected_users)} users")
            else:
                self.selected_users = self.select_users(glob_iter, self.users_per_round)

            # Local updates
            for user in self.selected_users:
                if self.dp == "None":
                    user.train_no_dp(glob_iter)
                else:
                    user.train_dp(self.sigma_g, glob_iter, self.max_norm)
                user.drop_lr()

            # Aggregation

            self.aggregate_parameters()
            self.get_max_norm()

            if self.noise:
                self.apply_channel_effect()

        self.save_results()
        self.save_norms()
        self.save_model()

    def aggregate_parameters(self):
        """Aggregation update of the server model."""
        assert (self.users is not None and len(self.users) > 0)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)

    def add_parameters(self, user, ratio):
        """Adding to the server model the contribution term from user."""
        # num_of_selected_users = len(self.selected_users)
        for server_param, del_model in zip(self.model.parameters(), user.delta_model):
            server_param.data = server_param.data + self.global_learning_rate * del_model.data * ratio

            # below : same sample size for all users
            # server_param.data = server_param.data + self.global_learning_rate * del_model.data / num_of_selected_users

    def get_max_norm(self):
        """Getting the maximum ||x_user^t+1 -x_server^t|| over the users"""
        param_norms = []
        for user in self.selected_users:
            param_norms.append(user.get_params_norm())
        self.param_norms.append(max(param_norms))

    def apply_channel_effect(self, sigma=1, power_control=2500):
        num_of_selected_users = len(self.selected_users)
        users_norms = []
        for user in self.selected_users:
            users_norms.append(user.get_params_norm())
        alpha_t = power_control / max(users_norms) ** 2
        for param in self.model.parameters():
            param.data = param.data + sigma / (alpha_t ** 0.5 * num_of_selected_users * self.communication_thresh) \
                         * torch.randn(param.data.size())
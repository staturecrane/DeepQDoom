from model import DoomConv
from PIL import Image
from vizdoom import *
from collections import deque
from torch.autograd import Variable

import random
import skimage.transform
import numpy as np
import torch
import torch.nn as nn
import gc
epochs = 10000
learning_rate = 0.0002
discount = 0.99
observe = 50
max_queue = 20
EXPLORE = 3000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
eps = INITIAL_EPSILON

game = DoomGame()

"""
setup all configuration
in default virtualenv environment, vizdoom scenarios will be in 
<path_to_site_packages>/vizdoom/scenarios
"""

game.set_doom_scenario_path("../../environments/gym/lib/python3.5/site-packages/vizdoom/scenarios/basic.wad")
game.set_doom_map("map01")
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_mode(Mode.PLAYER)

game.add_available_button(Button.MOVE_LEFT)
game.add_available_button(Button.MOVE_RIGHT)
game.add_available_button(Button.ATTACK)

game.add_available_game_variable(GameVariable.AMMO2)
# game.set_living_reward(-1)

game.set_episode_timeout(200)
game.set_episode_start_time(10)
game.set_window_visible(True)
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.init()

shoot = [0, 0, 1]
left = [1, 0, 0]
right = [0, 1, 0]
actions = [shoot, left, right]

model = DoomConv(3, 3).cuda()

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

softmax = nn.LogSoftmax().cuda()


def exploration_rate(epoch):
    """# Define exploration rate change over time"""
    start_eps = 1.0
    end_eps = 0.1
    const_eps_epochs = 0.1 * epochs  # 10% of learning time
    eps_decay_epochs = 0.6 * epochs  # 60% of learning time

    if epoch < const_eps_epochs:
        return start_eps
    elif epoch < eps_decay_epochs:
        # Linear decay
        return start_eps - (epoch - const_eps_epochs) / \
                           (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
    else:
        return end_eps


def runExploration(epoch):

    global eps

    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game.new_episode()

    experiences = []

    model.zero_grad()

    while not game.is_episode_finished():

        hidden, output_state = model.initHidden()
        # Gets the state
        state = game.get_state()

        # Which consists of:
        n           = state.number
        vars        = state.game_variables
        screen_buf  = state.screen_buffer
        depth_buf   = state.depth_buffer
        labels_buf  = state.labels_buffer
        automap_buf = state.automap_buffer
        labels      = state.labels

        img = skimage.transform.resize(screen_buf, (3, 128, 128))
        img = img.astype(np.float32)

        tensor = torch.FloatTensor(img).cuda()
        tensor = tensor.view(1, 3, 128, 128)

        output, hidden, output_state = model(Variable(tensor), hidden, output_state)

        # eps = exploration_rate(epoch)

        if random.random() <= eps:
            action = random.randint(0, len(actions) - 1)
        else:
            topv, topi = output.data.topk(1)
            action = topi[0][0]

        if eps > FINAL_EPSILON and epoch > observe:
            eps -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # Makes a random action and get remember reward.
        reward = game.make_action(actions[action])
        terminal = game.is_episode_finished()

        """
        # Prints state's game variables and reward.
        print("State #" + str(n))
        print("Game variables:", vars)
        print("Reward:", reward)
        print("=====================")
        """

        experiences.append((output, tensor, hidden, output_state, action, terminal, reward))

    return experiences


def test(epoch):

    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game.new_episode()

    experiences = []

    model.zero_grad()

    while not game.is_episode_finished():

        hidden, output_state = model.initHidden()
        # Gets the state
        state = game.get_state()

        # Which consists of:
        n = state.number
        vars = state.game_variables
        screen_buf = state.screen_buffer
        depth_buf = state.depth_buffer
        labels_buf = state.labels_buffer
        automap_buf = state.automap_buffer
        labels = state.labels

        img = skimage.transform.resize(screen_buf, (3, 128, 128))
        img = img.astype(np.float32)

        tensor = torch.FloatTensor(img).cuda()
        tensor = tensor.view(1, 3, 128, 128)

        output, hidden, output_state = model(Variable(tensor), hidden, output_state)

        topv, topi = output.data.topk(1)
        action = topi[0][0]

        # Makes a random action and get remember reward.
        reward = game.make_action(actions[action])
        terminal = game.is_episode_finished()

        """
        # Prints state's game variables and reward.
        print("State #" + str(n))
        print("Game variables:", vars)
        print("Reward:", reward)
        print("=====================")
        """

        # Check how the episode went.
    print("Episode {} finished".format(epoch))
    print("Total reward:", game.get_total_reward())
    print("************************")


def unpack_experience(output, frame, hidden, output_state, action, terminal, reward):
    return output, frame, hidden, output_state, action, terminal, reward


def learn(experience):

    episode_loss = 0

    model.zero_grad()

    for i in range(len(experience)):
        """
        To perform our Q-learning algorithm, we need the action and reward from the current experience state,
        plus the frame from i + 1 experience. We calculate the max-possible reward for the next frame and then
        run backprop on the mean-squared loss between our experienced output and our reward-enhanced output.
        
        In a nutshell, this trains our network to minimize the loss between what we did and what the Q-algorithm says
        we should have done based upon the reward (and the total future-reward) of the experience
        """
        output, frame, hidden, output_state, action, terminal, reward = unpack_experience(*experience[i])

        # for our target values, we have to create an entirely new tensor and fill it with the values of our output
        # before modifying with reward
        target_tensor = torch.FloatTensor(1, 3).cuda()
        target_tensor[0][0] = output.data[0][0]
        target_tensor[0][1] = output.data[0][1]
        target_tensor[0][2] = output.data[0][2]

        if terminal:
            target_tensor[0][action] += reward
        else:
            _, next_frame, _, _, _, _, _ = unpack_experience(*experience[i + 1])
            target_t, _, _ = model(Variable(next_frame), hidden, output_state)
            topv_q, _ = target_t.data.topk(1)
            target_tensor[0][action] += reward + discount * topv_q[0][0]

        _, target_topi = target_tensor.topk(1)
        target_idx = target_topi[0][0]

        target = Variable(torch.LongTensor([target_idx]).cuda())
        # target = Variable(target_tensor)
        episode_loss += criterion.forward(output, target)

    episode_loss.backward()

    torch.nn.utils.clip_grad_norm(model.parameters(), 0.2)

    optimizer.step()

    return episode_loss[0] / len(experience)

queue = []

for epoch in range(epochs):
    queue.append(
        runExploration(epoch)
    )

    if epoch > observe:
        random_idx = random.randint(0, len(queue) - 1)
        loss = learn(queue[random_idx])

        del queue[random_idx]

        print("Epoch {}: {}".format(epoch, loss.data[0]))

    if len(queue) > max_queue:
        queue.pop()

    # if epoch % 50 == 0:
    #    test(epoch)

    gc.collect()

    if epoch % 100 == 0:
        with open('checkpoints/model.pt', 'wb') as f:
            torch.save(model, f)

game.close()
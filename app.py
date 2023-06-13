import os.path

from flask import Flask, send_file, render_template_string, render_template, request
from PIL import Image
import io
import keyboard
from utils import *

app = Flask(__name__)
env = gym.make('CarRacing-v2', render_mode='rgb_array', continuous=False, domain_randomize=False)
observation, info = env.reset()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

stacked_size = 4
state_dim = (stacked_size, 84, 84)
action_dim = env.action_space.n
agent = DQN(state_dim, action_dim, buffer_size=0)
agent.network.load_state_dict(torch.load(os.path.join('Model', 'dqn_agent.pt'), map_location=device))
# agent.target_network.load_state_dict(torch.load(os.path.join('Model', 'dqn_agent.pt'), map_location=device))

observation = preprocess(observation)
stacked_frames = np.tile(observation, (stacked_size, 1, 1))

# Default to AI control
control = 'user'

# Default action
# action = 0 # Do nothing
user_input = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/switch-control', methods=['POST'])
def switch_control():
    global control
    control = 'user' if control == 'ai' else 'ai'
    return control


@app.route('/reset-game', methods=['POST'])
def reset_game():
    global stacked_frames
    observation, info = env.reset()
    observation = preprocess(observation)
    stacked_frames = np.tile(observation, (stacked_size, 1, 1))
    return 'Game reset'


@app.route('/keyboard-input', methods=['POST'])
def keyboard_input():
    global user_input
    user_input = request.get_json()['key']
    return ''


@app.route('/step')
def step():
    global stacked_frames, control, user_input
    if control == 'ai':
        action = agent.act(stacked_frames, training=False)
    else:
        action = get_action_user(user_input)
        user_input = None  # Reset user input

    for _ in range(stacked_size):  # Skip 4 frames
        observation, reward, done, truncated, info = env.step(action)
        if done:
            observation, info = env.reset()

    observation = preprocess(observation)
    stacked_frames = np.concatenate((stacked_frames[1:], observation[np.newaxis]), axis=0)

    last_rgb_array = env.render()
    img = Image.fromarray(last_rgb_array)
    file_object = io.BytesIO()
    img.save(file_object, 'PNG')
    file_object.seek(0)
    return send_file(file_object, mimetype='image/PNG')


def get_action_user(key):
    if key == 'ArrowLeft':
        return 2
    elif key == 'ArrowRight':
        return 1
    elif key == 'ArrowUp':
        return 3
    elif key == 'ArrowDown':
        return 4
    else:
        return 0


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000, debug=False)
    app.run()
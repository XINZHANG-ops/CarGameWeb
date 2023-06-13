from flask import Flask, send_file, render_template_string, render_template, request
from PIL import Image
import io
import keyboard
from utils import *

app = Flask(__name__)
env = gym.make('CarRacing-v2', render_mode='rgb_array', continuous=False, domain_randomize=False)
observation, info = env.reset()
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

stacked_size = 4
state_dim = (stacked_size, 84, 84)
action_dim = env.action_space.n
agent = DQN(state_dim, action_dim, buffer_size=0)
agent.network.load_state_dict(torch.load('dqn_agent.pt', map_location=device))
agent.target_network.load_state_dict(torch.load('dqn_agent.pt', map_location=device))

observation = preprocess(observation)
stacked_frames = np.tile(observation, (stacked_size, 1, 1))

# Default to AI control
control = 'user'

# Default action
# action = 0 # Do nothing
user_input = None


index_html = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            background-color: black; /* 设置背景颜色为黑色 */
            margin: 20px; /* 设置页面边距为20像素 */
        }

        #game {
            display: block;
            margin-bottom: 20px; /* 设置游戏窗口下方的外边距为20像素 */
        }
    </style>
</head>
<body>
    <img id="game" src="/step" width="600" height="400">
    <button id="switch">Switch to AI Control</button>
    <button id="reset">Reset Game</button>
    <script>
        setInterval(function() {
            document.getElementById('game').src = '/step?' + new Date().getTime();
        }, 100);  // 每100毫秒刷新一次图片

        document.getElementById('switch').addEventListener('click', function() {
            fetch('/switch-control', {
                method: 'POST'
            }).then(function(response) {
                return response.text();
            }).then(function(control) {
                document.getElementById('switch').textContent = 'Switch to ' + (control === 'ai' ? 'User' : 'AI') + ' Control';
            });
        });

        document.getElementById('reset').addEventListener('click', function() {
            fetch('/reset-game', {
                method: 'POST'
            });
        });
    </script>
    <script>
    window.addEventListener('keydown', function(event) {
        fetch('/keyboard-input', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                key: event.key
            }),
        });
    });
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    return render_template_string(index_html)


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
    app.run(host='192.168.1.164', port=5000, debug=False)
    #app.run(debug=False)
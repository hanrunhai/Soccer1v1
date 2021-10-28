import cv2
import numpy as np
from module import SoccerAgentFactory, SoccerAction, args_soccer, colors, width


class SoccerEnv:
    def __init__(self, agent_mode=None, opponent_mode=None, args=None):
        if args is None:
            args = args_soccer
            print("Soccer1V1 with its default args")
        self.args = args

        self.height = args["height"]
        self.width = args["width"]
        self.raw_occupancy = np.zeros(shape=(self.height, self.width))
        self.raw_occupancy[:, 0] = 1
        self.raw_occupancy[0, :] = 1
        self.raw_occupancy[:, -1] = 1
        self.raw_occupancy[-1, :] = 1
        self.raw_occupancy[1, 1] = 1
        self.raw_occupancy[2, 1] = 1
        self.raw_occupancy[6, 1] = 1
        self.raw_occupancy[7, 1] = 1
        self.raw_occupancy[1, 9] = 1
        self.raw_occupancy[2, 9] = 1
        self.raw_occupancy[6, 9] = 1
        self.raw_occupancy[7, 9] = 1

        if agent_mode is None:
            agent_mode = self.args["agent_mode"]
        if opponent_mode is None:
            opponent_mode = self.args["opponent_mode"]
        self.name = "_".join([self.args["env_name"], agent_mode, opponent_mode])

        self.agent_mode = agent_mode
        self.opponent_mode = opponent_mode
        self.agent = SoccerAgentFactory.get_agent("agent", self.agent_mode, args)
        self.opponent = SoccerAgentFactory.get_agent("opponent", self.opponent_mode, args)

        self.init_mode = self.args["init_mode"]
        self.step_mode = self.args["step_mode"]
        self.step_func = {"syn": self._step_syn, "asyn": self._step_asyn}

        self.step_counter = None
        self.ball_pos = None
        self.ball_owner = None
        self.occupancy = None
        self.initiative = None
        self.is_render = None
        self._init()
        self.plot_func = {
            "position": self.plot_trace_
        }

    def __del__(self):
        if self.is_render:
            cv2.destroyWindow("soccer")
        if getattr(self, "agent"):
            del self.agent
        if getattr(self, "opponent"):
            del self.opponent

    def _init(self):
        self.step_counter = 0
        if self.init_mode == "fixed_point":
            self.agent.pos = tuple(self.args["agent_start_position"])
            self.opponent.pos = tuple(self.args["opponent_start_position"])
            self.ball_pos = tuple(self.args["ball_start_position"])
        elif self.init_mode == "random":
            # self.opponent.pos = tuple(self.args["opponent_start_position"])  # only random init agent
            self.opponent.pos = self._get_random_pos(self.opponent.side)
            while True:
                self.agent.pos = self._get_random_pos(self.agent.side)
                if self.agent.pos != self.opponent.pos:
                    break
        else:
            raise NotImplemented
        self.agent.pos_clear()
        self.opponent.pos_clear()
        self.ball_owner = self.args["ball_owner"]
        if self.ball_owner == "random":
            self.ball_owner = np.random.choice(["agent", "opponent"])
        self.initiative = "opponent" if self.ball_owner == "agent" else "agent"
        self.is_render = False

        self._update_ball_pos()
        self.occupancy = self.raw_occupancy.copy()
        self.occupancy[self.agent.pos] = 0.5  # json和yaml中没有tuple的概念
        self.occupancy[self.opponent.pos] = 0.5

    def reset(self):
        self.agent.reset()
        self.opponent.reset()
        self._init()
        state_agent, state_opponent = self.cal_state()
        # self.render()
        return state_agent, state_opponent

    def print_info(self):
        print("The env is ready with {} situation and {} step".format(self.init_mode, self.step_mode))
        print("The ball holder is {} and initiative is {}".format(self.ball_owner, self.initiative))
        self.agent.print_info()
        self.opponent.print_info()

    def cal_state(self):
        agent_code = self._encode_pos(self.agent.pos)
        opponent_code = self._encode_pos(self.opponent.pos)
        state_anent = np.hstack((agent_code, opponent_code, int(self.ball_owner == self.agent.side)))
        state_oppo = np.hstack((opponent_code, agent_code, int(self.ball_owner == self.opponent.side)))
        return state_anent, state_oppo

    def step(self, action_agent=None, action_oppo=None, op_goal_id=1):
        if (action_agent is None) or (action_oppo is None):  # 配置默认动作
            state_agent, state_oppo = self.cal_state()
            if action_agent is None:
                action_agent = self.agent.choose_action(state_agent)
            if action_oppo is None:
                action_oppo = self.opponent.choose_action(state_oppo)
        self.step_counter += 1
        self.step_func[self.step_mode](action_agent, action_oppo)

        if not self.agent.check_pos() or not self.opponent.check_pos():  # 边界安全性检查
            raise ValueError("agent exceed bounder")

        next_state_agent, next_state_oppo = self.cal_state()
        (reward_agent, reward_oppo), done, info = self._check_done()
        return (action_agent, action_oppo), (next_state_agent, next_state_oppo), reward_agent, done, info

    def _step_syn(self, action_agent, action_oppo):
        next_pos_agent = self.agent.cal_next_pos(action_agent)
        next_pos_oppo = self.opponent.cal_next_pos(action_oppo)
        if next_pos_agent == next_pos_oppo:  # 发生碰撞
            self._switch_ball()
            if self.agent.pos != next_pos_agent and self.opponent.pos != next_pos_oppo:  # 这里处理不相邻争夺位置的情形
                initiative = np.random.choice(["agent", "opponent"], 1)[0]
                if initiative == "agent":  # 先手若为agent, opponent 位置不变
                    next_pos_oppo = self.opponent.pos
                else:
                    next_pos_agent = self.agent.pos
            elif self.agent.pos == next_pos_agent or self.opponent.pos == next_pos_oppo:  # 这里处理相邻相互追赶的情形
                if self.agent.pos == next_pos_agent:  # 双方位置均不变化
                    next_pos_oppo = self.opponent.pos
                else:
                    next_pos_agent = self.agent.pos
        elif self.agent.pos == next_pos_oppo and self.opponent.pos == next_pos_agent:  # 这里处理相邻交换位置的情形
            self._switch_ball()
            next_pos_agent = self.agent.pos
            next_pos_oppo = self.opponent.pos
        self.occupancy = self.agent.move_syn(next_pos_agent, self.occupancy)
        self.occupancy = self.opponent.move_syn(next_pos_oppo, self.occupancy)
        self._update_ball_pos()

    def _step_asyn(self, action_agent, action_oppo):
        if self.initiative == "agent":
            self.occupancy, collide_agent = self.agent.move_asyn(action_agent, self.occupancy)
            self.occupancy, collide_oppo = self.opponent.move_asyn(action_oppo, self.occupancy)
        else:
            self.occupancy, collide_oppo = self.opponent.move_asyn(action_oppo, self.occupancy)
            self.occupancy, collide_agent = self.agent.move_asyn(action_agent, self.occupancy)
        if (collide_agent + collide_oppo) == 1:  # 发生碰撞
            self._switch_ball()
        self._update_ball_pos()

    def _check_done(self):
        reward_agent, reward_oppo = 0, 0
        reward_agent += self.args["step_reward"]
        reward_oppo += self.args["step_reward"]

        done, info = False, ("draw", "IN VIOLENCE BATTLE")
        if self.agent.reach_goal() or self.opponent.reach_goal():
            goal_id = self.ball_pos[0] - 2
            # if goal_id in [op_goal_id]:  # 可以对不同的目标点设置不同的奖励
            if goal_id in [1, 2, 3]:
                done = True
                reward_goal = self.args["G{}_reward".format(goal_id)]
                if self.agent.reach_goal():
                    reward_agent += reward_goal
                    reward_oppo -= reward_goal
                    info = "red", "AGENT REACH GOAL_{}".format(goal_id)
                elif self.opponent.reach_goal():
                    reward_agent -= reward_goal
                    reward_oppo += reward_goal
                    info = "blue", "OPPONENT REACH GOAL_{}".format(goal_id)
                else:
                    raise ValueError("ball without owner")
        if not done and self.step_counter == self.args["episode_step"]:
            done, info = True, ("draw", "NO AGENT REACH GOALS")
        return (reward_agent, reward_oppo), done, info

    def _encode_pos(self, pos):
        row = np.zeros(self.height)
        column = np.zeros(self.width)
        row[pos[0]] = 1
        column[pos[1]] = 1
        code = np.hstack((row, column))
        return code

    def _decode_pos(self, code):
        row = np.argmax(code[0:self.height])
        column = np.argmax(code[self.height:(self.height + self.width)])
        return row, column

    def _update_ball_pos(self):
        if self.ball_owner == self.agent.side:
            self.agent.hold_ball = True
            self.opponent.hold_ball = False
            self.ball_pos = self.agent.pos
        elif self.ball_owner == self.opponent.side:
            self.agent.hold_ball = False
            self.opponent.hold_ball = True
            self.ball_pos = self.opponent.pos
        else:
            raise ValueError("ball without owner")

    def _switch_ball(self):
        if self.ball_owner == self.agent.side:
            self.ball_owner = self.opponent.side
        elif self.ball_owner == self.opponent.side:
            self.ball_owner = self.agent.side
        else:
            raise ValueError("ball without owner")

    def _get_random_pos(self, side):
        row_min = 1
        row_max = self.height - 1
        if side == "agent":
            column_min = 2
            column_max = 5
        elif side == "opponent":
            column_min = 6
            column_max = self.width - 2
        else:
            column_min = 2
            column_max = self.width - 2
        row = np.random.randint(row_min, row_max)
        column = np.random.randint(column_min, column_max)
        return row, column

    def _get_random_state(self):
        while True:
            agent_pos = self._get_random_pos("agent")
            opponent_pos = self._get_random_pos("opponent")
            if agent_pos != opponent_pos:
                break
        ag_code = self._encode_pos(agent_pos)
        opponent_code = self._encode_pos(opponent_pos)
        ball_code = np.random.randint(0, 2)
        state_agent = np.hstack((ag_code, opponent_code, ball_code))
        state_oppo = np.hstack((ag_code, opponent_code, 1 - ball_code))
        return state_agent, state_oppo

    def render(self, delay=0.1):
        self.is_render = True
        obs = np.ones(shape=(self.height * width, self.width * width, 3))
        for i in range(self.height):
            for j in range(self.width):
                if self.occupancy[i, j] == 1:
                    self._plot_rect(obs, (i, j), "blank")
                elif self.occupancy[i, j] == 0.5:
                    self._plot_rect(obs, (i, j), "green")
                else:
                    self._plot_rect(obs, (i, j), "white")
                    self._plot_line(obs, (i, j), "blank")
        for goal in self.agent.goals:
            self._plot_rect(obs, goal, "yellow")
            self._plot_line(obs, goal, "blank")
        for goal in self.opponent.goals:
            self._plot_rect(obs, goal, "yellow")
            self._plot_line(obs, goal, "blank")

        self._plot_rect(obs, self.agent.pos, "red")
        self._plot_rect(obs, self.opponent.pos, "blue")
        self._plot_circle(obs, self.ball_pos, "green")
        # cv2.imwrite("./soccer_.png", obs)  # save the pic of the env

        cv2.imshow("soccer", obs)
        cv2.waitKey(int(delay * 1000))

    def plot_trace(self, delay=0.1):
        """
        :param delay: time to display the trace
        :return: None
        """
        obs = np.ones(shape=(self.height * width, self.width * width, 3))
        for i in range(self.height):
            for j in range(self.width):
                if self.occupancy[i, j] == 1:
                    self._plot_rect(obs, (i, j), "blank")
                elif self.occupancy[i, j] == 0.5:
                    self._plot_rect(obs, (i, j), "green")
                else:
                    self._plot_rect(obs, (i, j), "white")
                    self._plot_line(obs, (i, j), "blank")
        for goal in self.agent.goals:
            self._plot_rect(obs, goal, "yellow_1")
            self._plot_line(obs, goal, "blank")
        for goal in self.opponent.goals:
            self._plot_rect(obs, goal, "yellow_2")
            self._plot_line(obs, goal, "blank")

        for count, pos in enumerate(self.agent.pos_list):
            self._plot_circle(obs, pos, "red", 0.5 * count / self.step_counter)
            self._plot_txt(obs, pos, str(count), "blank")
        for count, pos in enumerate(self.opponent.pos_list):
            self._plot_circle(obs, pos, "blue", 0.5 * count / self.step_counter)
            self._plot_txt(obs, pos, str(count), "blank")
        self._plot_rect(obs, self.agent.pos, "red")
        self._plot_rect(obs, self.opponent.pos, "blue")
        self._plot_circle(obs, self.ball_pos, "green")

        cv2.imshow("soccer_trace", obs)
        cv2.waitKey(int(delay * 1000))
        cv2.destroyWindow("soccer_trace")

    def plot_trace_(self, render_info=None):
        """
        show and save trace
        :param render_info: dict with save and render info
        :return: None
        """
        if render_info is None:
            return
        obs = np.ones(shape=(self.height * width, self.width * width, 3))
        for i in range(self.height):
            for j in range(self.width):
                if self.occupancy[i, j] == 1:
                    self._plot_rect(obs, (i, j), "blank")
                elif self.occupancy[i, j] == 0.5:
                    self._plot_rect(obs, (i, j), "green")
                else:
                    self._plot_rect(obs, (i, j), "white")
                    self._plot_line(obs, (i, j), "blank")
        for goal in self.agent.goals:
            self._plot_rect(obs, goal, "yellow_1")
            self._plot_line(obs, goal, "blank")
        for goal in self.opponent.goals:
            self._plot_rect(obs, goal, "yellow_2")
            self._plot_line(obs, goal, "blank")

        for count, pos in enumerate(self.agent.pos_list):
            self._plot_circle(obs, pos, "red", 0.5 * count / self.step_counter)
            self._plot_txt(obs, pos, str(count), "red")
        for count, pos in enumerate(self.opponent.pos_list):
            self._plot_circle(obs, pos, "blue", 0.5 * count / self.step_counter)
            self._plot_txt(obs, pos, str(count), "blue")
        self._plot_rect(obs, self.agent.pos, "red")
        self._plot_rect(obs, self.opponent.pos, "blue")
        self._plot_circle(obs, self.ball_pos, "green")

        if render_info.get("save", False):
            cv2.imwrite(render_info["save_path"], obs)
            print("soccer trace fig saved at", render_info["save_path"])
        if render_info["render"]:
            cv2.imshow("soccer_trace_", obs)
            cv2.waitKey(int(render_info["delay"] * 1000))
            cv2.destroyWindow("soccer_trace_")

    @staticmethod
    def _plot_rect(obs, pos, color):
        x, y = pos
        cv2.rectangle(obs, (y * width, x * width), ((y + 1) * width, (x + 1) * width), colors[color], -1)

    @staticmethod
    def _plot_line(obs, pos, color):
        x, y = pos
        cv2.line(obs, (y * width, x * width), ((y + 1) * width, x * width), colors[color], 1)
        cv2.line(obs, (y * width, (x + 1) * width), ((y + 1) * width, (x + 1) * width), colors[color], 1)
        cv2.line(obs, (y * width, x * width), (y * width, (x + 1) * width), colors[color], 1)
        cv2.line(obs, ((y + 1) * width, x * width), ((y + 1) * width, (x + 1) * width), colors[color], 1)

    @staticmethod
    def _plot_circle(obs, pos, color, radius=0.5):
        x, y = pos
        cv2.circle(obs, (int((y + 0.5) * width), int((x + 0.5) * width)), int(radius * width), colors[color], -1)

    @staticmethod
    def _plot_txt(obs, pos, txt, color):
        x, y = pos
        if txt is None:
            txt = color
        cv2.putText(obs, txt, (int((y - 1) * width), int((x + 0.5) * width)), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                    colors[color], 1)
        # cv2.putText(obs, "goal", (int(y * width), int((x + 0.5) * width)), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
        #             colors["blank"], 1)

    def run_episodes(self, episodes=1, agent=None, opponent=None, render=False, args=None):
        done_steps = []
        reward_episodes = []
        winner_count = {"red": 0, "blue": 0, "draw": 0}
        done_info_count = {}
        save_count = 0
        if agent is not None and agent.mode not in ["AI"]:  # for agent whose mode in ["rule_2", "normal", ""]
            del self.agent
            self.agent = agent
        if opponent is not None:
            del self.opponent
            self.opponent = opponent

        self.reset()
        self.print_info()

        for episode in range(1, episodes + 1):
            step = 0
            reward_episode = 0
            state_red, state_blue = self.reset()
            while True:
                action_red = agent.choose_action(state_red, opponent=self.opponent)
                # action_red = self.agent.choose_action(state_red, opponent=self.opponent)# todo without the AI PLAYER
                action_blue = self.opponent.choose_action(state_blue, opponent=self.agent)
                action, next_state, reward, done, info = self.step(action_red, action_blue)
                step += 1
                state_red, state_blue = next_state
                reward_episode += reward
                if render:
                    self.render(args["delay"])
                    print("step-{}\taction_agent:{}\taction_oppo:{}\treward:{}\tinfo:{}".format(
                        self.step_counter, SoccerAction(action_red).name, action_blue.name, reward, info))
                if done:
                    break

            render_info = {"render": False, "winner": info[0], "done_info": info[1], "save": False,
                           "delay": args.get("delay", 0.001)}
            if (render or args["save"]) and save_count < args["save_num"]:
                save_count += 1
                render_info["save"] = args["save"]
                render_info["render"] = render
                for plot_type in args.get("plot_type", []):
                    fig_name = "_".join(
                        [args.get("mode", "test"), str(args.get("current_episode", 1)), plot_type, str(save_count)])
                    save_path = args.get("results_path", "results/fig_temp/") + args.get("eval_path", "")
                    render_info["save_path"] = save_path + fig_name + ".png"
                    # self.plot_func[plot_type](render_info)

            if render_info["render"]:
                self.render()

            done_steps.append(step)
            reward_episodes.append(reward_episode)
            winner = info[0]
            winner_count[winner] = round(winner_count.get(winner, 0) + 1 / episodes, 4)
            done_info = "{}_{}".format(winner.upper(), info[1][-6:])
            done_info_count[done_info] = round(done_info_count.get(done_info, 0) + 1 / episodes, 4)
            print("eval-episode:{}\tstep:{}\twinner:{}\treward:{:.4f}\tmin:{:.4f}\tmean:{:.4f}\tmax:{:.4f}".format(
                episode, step, info, reward_episode, min(reward_episodes), np.mean(reward_episodes),
                max(reward_episodes)))
        return done_steps, reward_episodes, winner_count, done_info_count

    def train(self):
        pass

    def set_side(self, side, mode):
        if side in ["agent", "opponent"]:
            delattr(self, side)
            setattr(self, side, SoccerAgentFactory.get_agent(side, mode, self.args))
            print("{} switch {} as {}".format(self.name, side, mode))


def main_1():
    # for policy_id in range(1, args_soccer["opponent_policy_num"]):
    #     soccer = SoccerEnv("random", "normal_.{}".format(policy_id))
    soccer = SoccerEnv("AI", "AI")
    for policy_id in range(1, 100):
        state = soccer.reset()
        while True:
            action_agent = SoccerAction.UP
            action_oppo = SoccerAction.RIGHT

            action, next_state, reward, done, info = soccer.step()
            # action, state_next, reward, done, info = soccer.step(action_agent)
            # action, state_next, reward, done, info = soccer.step(action_agent, action_oppo)
            state = next_state
            action_agent, action_oppo = action
            print("step-{}\taction_agent:{}\taction_oppo:{}\treward:{}\tinfo:{}".format(
                soccer.step_counter, action_agent.name, action_oppo.name, reward, info))
            soccer.render(0.1)
            if done:
                break
        # soccer.plot_trace()
        s = input("The game is done!, waiting(y/n):\n")
        print(s)
        if "y" not in s:
            break


def main_2():
    env = SoccerEnv()

    red = SoccerAgentFactory.get_agent("agent", "rule_2")
    blue = SoccerAgentFactory.get_agent("opponent", "rule_2")

    done_steps, reward_episodes, winner_count, done_info_count = \
        env.run_episodes(episodes=100, agent=red, opponent=blue, render=False,
                         args={"save": False, "save_num": 100, "plot_type": ["position"], "delay": 0.1})
    # env.run_episodes(episodes=100, agent=red, opponent=blue)

    print(winner_count, done_info_count)
    print(reward_episodes)


if __name__ == "__main__":
    # main_1()
    main_2()

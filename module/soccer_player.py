import warnings
import numpy as np
from enum import IntEnum

from util.utils import config_args

args_soccer = config_args("env/soccer_1v1_conf")
colors = {
    "blank": (0, 0, 0), "white": (255, 255, 255),
    "red": (0, 0, 255), "blue": (255, 0, 0),
    "green": (0, 255, 0), "yellow": (0, 255, 255),
    "yellow_1": (0, 181, 255), "yellow_2": (0, 226, 176)
}
width = args_soccer["env_enlarge"]


class SoccerAction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STILL = 4


class SoccerPlayer:
    def __init__(self, side="agent", args=None):
        self.args = args
        self.side = side
        self.opponent_side = "opponent" if self.side == "agent" else "agent"
        self.action_space = ["stable", "left", "right", "up", "down"]
        if self.side == "agent":
            self.goals = [tuple(self.args["ag_G1"]), tuple(self.args["ag_G2"]), tuple(self.args["ag_G3"])]
            self.goals_opponent = [tuple(self.args["op_G1"]), tuple(self.args["op_G2"]), tuple(self.args["op_G3"])]
        else:
            self.goals = [tuple(self.args["op_G1"]), tuple(self.args["op_G2"]), tuple(self.args["op_G3"])]
            self.goals_opponent = [tuple(self.args["ag_G1"]), tuple(self.args["ag_G2"]), tuple(self.args["ag_G3"])]

        self.hold_ball = None  # decided by the env
        self.pos = None
        self.pos_list = []
        self.init()

    def reset(self):
        self.init()

    def print_info(self):
        print("{} player start with pos:{}".format(self.side.capitalize(), self.pos), end=" ")

    def init(self):
        self.pos_clear()

    def pos_clear(self):
        self.pos_list.clear()
        self.pos_list.append(self.pos)

    def cal_next_pos(self, action, pos=None):
        if pos is None:
            pos = self.pos
        if action == SoccerAction.LEFT:
            next_pos = pos[0], pos[1] - 1
        elif action == SoccerAction.RIGHT:
            next_pos = pos[0], pos[1] + 1
        elif action == SoccerAction.UP:
            next_pos = pos[0] - 1, pos[1]
        elif action == SoccerAction.DOWN:
            next_pos = pos[0] + 1, pos[1]
        else:
            next_pos = pos
        if not self.check_pos(next_pos):  # 处理碰到障碍物的情况，未考虑智能体障碍物
            next_pos = pos
        return next_pos

    def cal_next_poses(self, action_list, pos=None):  # 未考虑agent障碍物
        if pos is None:
            pos = self.pos
        pos_list = []
        for action in action_list:
            next_pos = self.cal_next_pos(action, pos)
            pos = next_pos
            pos_list.append(pos)
        return pos_list

    def move_syn(self, next_pos, occupancy):
        occupancy[self.pos] = 0
        self.pos = next_pos
        self.pos_list.append(self.pos)
        occupancy[self.pos] = 0.5
        return occupancy

    def move_asyn(self, action, occupancy):
        occupancy[self.pos] = 0
        next_pos = self.cal_next_pos(action)

        collide = False
        if occupancy[next_pos] == 0.5 and next_pos != self.pos:  # to detect whether collide happened
            collide = True

        if occupancy[next_pos] == 1 or collide:  # stay where they are, when encounter the wall or collide happened
            next_pos = self.pos

        occupancy[next_pos] = 0.5
        self.pos = next_pos
        self.pos_list.append(self.pos)
        return occupancy, collide

    def reach_goal(self):
        return self.hold_ball and self.pos in self.goals

    def check_pos(self, pos=None):
        if pos is None:
            pos = self.pos
        height, width = 9, 11
        if 1 <= pos[0] <= height - 2 and 2 <= pos[1] <= width - 3:
            return True
        elif pos in self.goals:
            return True
        elif pos in self.goals_opponent:
            return True
        else:
            return False

    @staticmethod
    def _decode_pos(code, side="self"):
        if side == "self":
            row = np.argmax(code[0:9])
            column = np.argmax(code[9:20])
        elif side == "oppo":
            row = np.argmax(code[20:29])
            column = np.argmax(code[29:40])
        else:
            raise NotImplementedError
        return row, column


class SoccerAgent(SoccerPlayer):
    def __init__(self, side, mode, args=None):
        super().__init__(side, args)
        self.mode = mode
        self.name = "_".join([side, mode])
        self.state_dim = self.args["state_dim"]
        self.action_dim = self.args["action_dim"]

    def reset(self):
        super().reset()

    def print_info(self):
        super().print_info()
        print("the {}-{} policy with {}-state and {}-action".format(
            self.side, self.mode, self.state_dim, self.action_dim))

    def choose_action(self, state, opponent=None, phase="eval"):
        """
            # random
            # manual
            # AI, still
            # rule_2 0, 2, 4
            # normal_1 1, 2, 3, 4, 5, 6
        """
        if phase == "no pos":  # for soccer env
            self.pos = self._decode_pos(state, "self")
        else:
            assert self.pos == self._decode_pos(state, "self")
        if self.mode in ["random", "AI"]:
            action = np.random.randint(0, self.action_dim)
        elif self.mode in ["still"]:
            action = SoccerAction.STILL
        elif self.mode in ["manual"] or "normal" in self.mode or "rule" in self.mode:
            action = SoccerAction.STILL
        else:
            raise NotImplementedError
        assert 0 <= action < self.action_dim
        return SoccerAction(action)


class FixedPolicy(SoccerAgent):
    def __init__(self, side, mode="normal_5", args=None):
        if args is None:
            args = args_soccer
            print("FixedPolicy with its default args")
        super().__init__(side, mode, args)

        self.policies = {
            1: {(3, 7): 0, (2, 7): 0, (1, 7): 3, (1, 6): 3, (1, 5): 2, (2, 5): 2, (3, 5): 3, (3, 4): 3, (3, 3): 0,
                (2, 3): 3, (2, 2): 2, (3, 2): 3, (3, 1): 4},
            2: {(3, 7): 2, (4, 7): 2, (5, 7): 2, (6, 7): 3, (6, 6): 3, (6, 5): 3, (6, 4): 0, (5, 4): 3, (5, 3): 0,
                (4, 3): 0, (3, 3): 3, (3, 2): 3, (3, 1): 4},
            3: {(3, 7): 3, (3, 6): 3, (3, 5): 0, (2, 5): 3, (2, 4): 2, (3, 4): 2, (4, 4): 3, (4, 3): 3, (4, 2): 3,
                (4, 1): 4},
            4: {(3, 7): 3, (3, 6): 2, (4, 6): 3, (4, 5): 3, (4, 4): 3, (4, 3): 3, (4, 2): 3, (4, 1): 4},
            # 6: {(3, 7): 3, (3, 6): 3, (3, 5): 2, (4, 5): 3, (4, 4): 0, (3, 4): 3, (3, 3): 2, (4, 3): 2, (5, 3): 3, (5, 2): 3, (5, 1): 4},
            6: {(3, 7): 3, (3, 6): 3, (3, 5): 2, (4, 5): 3, (4, 4): 2, (5, 4): 2, (6, 4): 3, (6, 3): 0, (5, 3): 3,
                (5, 2): 3, (5, 1): 4},
            5: {(3, 7): 2, (4, 7): 2, (5, 7): 3, (5, 6): 3, (5, 5): 2, (6, 5): 2, (7, 5): 3, (7, 4): 3, (7, 3): 3,
                (7, 2): 0, (6, 2): 0, (5, 2): 3, (5, 1): 4}
        }
        self.policy_id = None
        self.policy = None
        self._init()

    def _init(self):
        policy_id = int(self.mode[-1])
        assert policy_id in range(1, self.args["opponent_policy_num"] + 1)
        self.set_policy(policy_id)

    def set_policy(self, policy_id):
        self.policy_id = policy_id
        self.policy = self.policies[policy_id]
        # print("Opponent set policy {}".format(policy_id))

    def _check_policy_usable(self, policy_id, pos=None):
        if pos is None:
            pos = self.pos
        return pos in self.policies[policy_id]

    def choose_action(self, state, opponent=None, phase="eval"):
        super().choose_action(state, opponent, phase)
        if self._check_policy_usable(self.policy_id):
            action = self.policy[self.pos]
        else:
            action = np.random.randint(0, self.action_dim)
        return SoccerAction(action)


class RulePolicy(SoccerAgent):
    def __init__(self, side, mode="rule_2", args=None):
        if args is None:
            args = args_soccer
            print("RulePolicy with its default args")
        super().__init__(side, mode, args)
        self.level = int(mode[-1])

    def _cal_pos2goal(self, oppo_pos, goal_side):
        if goal_side is None:
            goal_side = self.side
        up, middle, down = self.goals[0][0], self.goals[1][0], self.goals[-1][0]
        left, right = 1, 9
        self_dis2goal = (up - self.pos[0]) if self.pos[0] < middle else (self.pos[0] - down) if self.pos[
                                                                                                    0] > middle else 0
        oppo_dis2goal = (up - oppo_pos[0]) if oppo_pos[0] < middle else (oppo_pos[0] - down) if oppo_pos[
                                                                                                    0] > middle else 0
        if goal_side == "agent":
            self_dis2goal += right - self.pos[1]
            oppo_dis2goal += right - oppo_pos[1]
        else:
            self_dis2goal += self.pos[1] - left
            oppo_dis2goal += oppo_pos[1] - left
        return oppo_dis2goal - self_dis2goal

    def _cal_pos2oppo(self, oppo_pos):
        if self.side == "agent":
            dis2oppo = self.pos[1] - oppo_pos[1]
        else:
            dis2oppo = oppo_pos[1] - self.pos[1]
        return dis2oppo

    def _check_at_home(self):
        return (self.pos[1] < 5) if self.side == "agent" else (self.pos[1] > 5)

    def choose_action(self, state, opponent=None, phase="eval"):
        super().choose_action(state, opponent, phase)
        oppo_pos = self._decode_pos(state, "oppo")
        action4offensive = SoccerAction.RIGHT if self.side == "agent" else SoccerAction.LEFT
        action4defensive = SoccerAction.LEFT if self.side == "agent" else SoccerAction.RIGHT
        if self.hold_ball:
            action_candidate = self.choose_candidate_action(oppo_pos)
            if action_candidate:
                suboptimal_action = np.random.choice(action_candidate, 1)[0]
            else:
                suboptimal_action = SoccerAction.STILL
                raise ValueError("No suitable action")

            if self.pos[0] < 3:  # self 位于上方
                action = SoccerAction.DOWN if SoccerAction.DOWN in action_candidate \
                    else action4offensive if action4offensive in action_candidate else suboptimal_action
            elif self.pos[0] > 5:  # self 位于下方
                action = SoccerAction.UP if SoccerAction.UP in action_candidate \
                    else action4offensive if action4offensive in action_candidate else suboptimal_action
            else:  # self 位于中部
                if self._cal_pos2goal(oppo_pos, self.side) >= 0:  # 离自己目标的距离，占优势
                    action = action4offensive
                # else:
                #     action = action4defensive  # 占劣势, 躺平
                else:  # 占劣势, 卷一卷
                    if self.pos[0] != oppo_pos[0]:  # self 和 oppo 不位于同一条线上
                        if self.pos[0] in [3, 4] and self.cal_next_pos(action4offensive, self.cal_next_pos(
                                SoccerAction.DOWN)) == oppo_pos:  # 考虑oppo处于右下侧
                            action = SoccerAction.UP if self.pos[0] == 4 else \
                                np.random.choice([action4offensive, SoccerAction.DOWN], 1)[0]
                        elif self.pos[0] in [4, 5] and self.cal_next_pos(action4offensive, self.cal_next_pos(
                                SoccerAction.UP)) == oppo_pos:  # 考虑oppo处于右上侧
                            action = SoccerAction.DOWN if self.pos[0] == 4 else \
                                np.random.choice([action4offensive, SoccerAction.UP], 1)[0]
                        else:
                            action = action4offensive
                    else:  # self 和 oppo 位于同一条线上,not very good, 只考虑了# todo
                        if self.pos[0] == 3:
                            action = np.random.choice([action4offensive, SoccerAction.DOWN], 1)[0]
                            # action = SoccerAction.DOWN
                        elif self.pos[0] == 5:
                            action = np.random.choice([action4offensive, SoccerAction.UP], 1)[0]
                            # action = SoccerAction.UP
                        else:
                            action = np.random.choice([action4offensive, SoccerAction.UP, SoccerAction.DOWN], 1)[0]
                            # action = suboptimal_action  # random
        else:
            dis_margin4defensive = self.level  # decided by (height-2-1)/2 # todo policy 的保守程度
            dis2goal = self._cal_pos2goal(oppo_pos, self.opponent_side)  # 离oppo目标的距离，占优势
            if dis2goal >= dis_margin4defensive:  # self 位于我方大后方
                # self 位于上方或下方
                action = SoccerAction.DOWN if self.pos[0] < 4 else SoccerAction.UP if self.pos[
                                                                                          0] > 4 else action4offensive
            elif dis2goal > 0:  # self 频近接敌
                # if dis2goal >= 0:  # self 频近接敌
                if self.pos[0] == oppo_pos[0]:
                    action = action4defensive if self._check_at_home() else action4offensive  # todo
                else:
                    action = SoccerAction.UP if self.pos[0] > oppo_pos[0] else SoccerAction.DOWN
            else:
                action = action4defensive
        # print("side:{}\thold ball:{}\tdis2goal:{}\tdis2oppo:{}".format(
        #     self.side, self.hold_ball, self._cal_pos2goal(oppo_pos), self._cal_pos2oppo(oppo_pos)))
        return SoccerAction(action)

    def choose_candidate_action(self, oppo_pos):
        action_list = []
        for action in SoccerAction:
            if self.cal_next_pos(action) != oppo_pos:
                action_list.append(action)
        return action_list


class ManualPolicy(SoccerAgent):
    key = None

    @staticmethod
    def on_press(key):
        if hasattr(key, "name") and key.name in ["up", "down", "left", "right", "space"]:
            ManualPolicy.key = key.name
            return False
        else:
            return True

    def __init__(self, side, mode="manual", args=None):
        if args is None:
            args = args_soccer
            print("ManualPolicy with its default args")
        super().__init__(side, mode, args)

        self.listener = None
        self.action_dict = {
            "up": SoccerAction.UP, "down": SoccerAction.DOWN,
            "left": SoccerAction.LEFT, "right": SoccerAction.RIGHT, "space": SoccerAction.STILL
        }

    def __del__(self):
        del self.listener

    def choose_action(self, state, opponent=None, phase="eval"):
        super().choose_action(state, opponent, phase)
        from pynput import keyboard
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        print("waiting for input ......")
        while not ManualPolicy.key:
            pass
        action = self.action_dict[ManualPolicy.key]
        del self.listener
        ManualPolicy.key = None
        self.listener = None
        return SoccerAction(action)


# # from alg import alg_registry_
# alg_registry_ = {"d3qn_pi", None}
#
# alg_name = "d3qn_pi"
# alg = alg_registry_[alg_name]
# class AIPolicy(SoccerAgent, alg):
#     def __init__(self, side, mode="AI", args=None):
#         print(AIPolicy.__mro__)
#         if args is None:
#             args = args_soccer
#             print("AIPolicy with its default args")
#         super(AIPolicy, self).__init__(side, mode, args)
#         if not args.get("gamma"):
#             print("AIPolicy with its default args")
#             args.update(config_args("{}_conf".format(alg_name)))
#         super(SoccerPlayer, self).__init__(args)
#
#     def reset(self):
#         super(AIPolicy, self).reset()
#
#     def choose_action(self, state, opponent=None, phase="eval"):
#         super(AIPolicy, self).choose_action(state, opponent, phase)
#         action = super(SoccerPlayer, self).choose_action(state, opponent, phase)
#         return SoccerAction(action)


class SoccerAgentFactory:
    @staticmethod
    def get_agent(side, mode, args=None):
        if args is None:
            args = args_soccer
            print("AgentFactory with its default args")
        if mode in ["random", "still"]:
            agent = SoccerAgent(side, mode, args)
        elif mode == "AI":
            # agent = AIPolicy(side, mode, args)  # todo
            agent = SoccerAgent(side, "AI", args)
        elif "normal" in mode:
            agent = FixedPolicy(side, mode, args)
        elif "rule" in mode:
            agent = RulePolicy(side, mode, args)
        elif mode == "manual":
            agent = ManualPolicy(side, mode, args)
        else:
            agent = None
            warnings.warn("agent is None")
        return agent

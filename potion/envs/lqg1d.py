"""classic Linear Quadratic Gaussian Regulator task"""
from numbers import Number

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

"""
Linear quadratic gaussian regulator task.

References
----------
  - Simone Parisi, Matteo Pirotta, Nicola Smacchia,
    Luca Bascetta, Marcello Restelli,
    Policy gradient approaches for multi-objective sequential decision making
    2014 International Joint Conference on Neural Networks (IJCNN)
  - Jan  Peters  and  Stefan  Schaal,
    Reinforcement  learning of motor  skills  with  policy  gradients,
    Neural  Networks, vol. 21, no. 4, pp. 682-697, 2008.

"""

class LQG1D(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, discrete_reward=False):
        self.horizon = 20
        self.gamma = 0.9
        self.sigma_controller = 1.
        self.discrete_reward = discrete_reward
        self.max_pos = 2.0
        self.max_action = 2.0
        self.sigma_noise = 0
        self.A = np.array([1]).reshape((1, 1))
        self.B = np.array([1]).reshape((1, 1))
        self.Q = np.array([0.5]).reshape((1, 1))
        self.R = np.array([0.5]).reshape((1, 1))

        # gym attributes
        self.viewer = None
        high = np.array([self.max_pos])
        self.action_space = spaces.Box(low=-self.max_action,
                                       high=self.max_action,
                                       shape=(1,), 
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high,
                                            dtype=np.float32)

        self.initial_states = np.array([[1, 2, 5, 7, 10]]).T

        # initialize state
        self.seed()
        self.reset()

    def step(self, action, render=False):
        u = np.clip(action, -self.max_action, self.max_action)
        noise = 0
        if self.sigma_noise > 0:
            noise = self.np_random.randn() * self.sigma_noise
        xn = np.clip(np.dot(self.A, self.state) + np.dot(self.B, u) + noise, -self.max_pos, self.max_pos)
        cost = np.dot(self.state,
                      np.dot(self.Q, self.state)) + \
            np.dot(u, np.dot(self.R, u))

        self.state = np.array(xn.ravel())
        if self.discrete_reward:
            if abs(self.state[0]) <= 2 and abs(u) <= 2:
                return self.get_state(), 0, False, {}
            return self.get_state(), -1, False, {}
        return self.get_state(), -np.asscalar(cost), False, False, {}

    def reset(self, *, seed = None, options = None):

        if options is not None:
            if 'state' in options:
                self.state = np.array(options['state'])
        else:
            self.state = np.array([self.np_random.uniform(low=-self.max_pos,
                                                          high=self.max_pos)])

        return self.get_state(), {}

    def get_state(self):
        return np.array(self.state)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = (self.max_pos * 2) * 2
        scale = screen_width / world_width
        bally = 100
        ballradius = 3

        if self.viewer is None:
            clearance = 0  # y-offset
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            mass = rendering.make_circle(ballradius * 2)
            mass.set_color(.8, .3, .3)
            mass.add_attr(rendering.Transform(translation=(0, clearance)))
            self.masstrans = rendering.Transform()
            mass.add_attr(self.masstrans)
            self.viewer.add_geom(mass)
            self.track = rendering.Line((0, bally), (screen_width, bally))
            self.track.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(self.track)
            zero_line = rendering.Line((screen_width / 2, 0),
                                       (screen_width / 2, screen_height))
            zero_line.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(zero_line)

        x = self.state[0]
        ballx = x * scale + screen_width / 2.0
        self.masstrans.set_translation(ballx, bally)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _computeP2(self, K):
        """
        This function computes the Riccati equation associated to the LQG
        problem.
        Args:
            K (matrix): the matrix associated to the linear controller K * x

        Returns:
            P (matrix): the Riccati Matrix

        """
        I = np.eye(self.Q.shape[0], self.Q.shape[1])
        if np.array_equal(self.A, I) and np.array_equal(self.B, I):
            P = (self.Q + np.dot(K.T, np.dot(self.R, K))) / (I - self.gamma *
                                                             (I + 2 * K + K **
                                                              2))
        else:
            tolerance = 0.0001
            converged = False
            P = np.eye(self.Q.shape[0], self.Q.shape[1])
            while not converged:
                Pnew = self.Q + self.gamma * np.dot(self.A.T,
                                                    np.dot(P, self.A)) + \
                       self.gamma * np.dot(K.T, np.dot(self.B.T,
                                                       np.dot(P, self.A))) + \
                       self.gamma * np.dot(self.A.T,
                                           np.dot(P, np.dot(self.B, K))) + \
                       self.gamma * np.dot(K.T,
                                           np.dot(self.B.T,
                                                  np.dot(P, np.dot(self.B,
                                                                   K)))) + \
                       np.dot(K.T, np.dot(self.R, K))
                converged = np.max(np.abs(P - Pnew)) < tolerance
                P = Pnew
        return P

    def computeOptimalK(self):
        """
        This function computes the optimal linear controller associated to the
        LQG problem (u = K * x).

        Returns:
            K (matrix): the optimal controller

        """
        P = np.eye(self.Q.shape[0], self.Q.shape[1])
        for i in range(100):
            K = -self.gamma * np.dot(np.linalg.inv(
                self.R + self.gamma * (np.dot(self.B.T, np.dot(P, self.B)))),
                                       np.dot(self.B.T, np.dot(P, self.A)))
            P = self._computeP2(K)
        K = -self.gamma * np.dot(np.linalg.inv(self.R + self.gamma *
                                               (np.dot(self.B.T,
                                                       np.dot(P, self.B)))),
                                 np.dot(self.B.T, np.dot(P, self.A)))
        return K

    def computeJ(self, K, Sigma, n_random_x0=100):
        """
        This function computes the discounted reward associated to the provided
        linear controller (u = Kx + \epsilon, \epsilon \sim N(0,\Sigma)).
        Args:
            K (matrix): the controller matrix
            Sigma (matrix): covariance matrix of the zero-mean noise added to
                            the controller action
            n_random_x0: the number of samples to draw in order to average over
                         the initial state

        Returns:
            J (float): The discounted reward

        """
        if isinstance(K, Number):
            K = np.array([K]).reshape(1, 1)
        if isinstance(Sigma, Number):
            Sigma = np.array([Sigma]).reshape(1, 1)

        P = self._computeP2(K)
        W =  (1 / (1 - self.gamma)) * \
                np.trace(np.dot(
                    Sigma, (self.R + self.gamma * np.dot(self.B.T,
                                             np.dot(P, self.B)))))

        if np.size(K)==1:
            return min(0,np.asscalar(-self.max_pos**2*P/3 - W))

        J = 0.0
        for i in range(n_random_x0):
            self.reset()
            x0 = self.get_state()
            J -= np.dot(x0.T, np.dot(P, x0)) \
                +  W
        J /= n_random_x0
        return min(0,J)

    def grad_K(self, K, Sigma):
        I = np.eye(self.Q.shape[0], self.Q.shape[1])
        if not np.array_equal(self.A, I) or not np.array_equal(self.B, I):
            raise NotImplementedError
        if not isinstance(K,Number) or not isinstance(Sigma, Number):
            raise NotImplementedError
        theta = np.asscalar(np.array(K))
        sigma = np.asscalar(np.array(Sigma))

        den = 1 - self.gamma*(1 + 2*theta + theta**2)
        dePdeK = 2*(theta*self.R/den + self.gamma*(self.Q + theta**2*self.R)*(1+theta)/den**2)
        return np.asscalar(- dePdeK*(self.max_pos**2/3 + self.gamma*sigma/(1 - self.gamma)))

    def grad_Sigma(self, K, Sigma=None):
        I = np.eye(self.Q.shape[0], self.Q.shape[1])
        if not np.array_equal(self.A, I) or not np.array_equal(self.B, I):
            raise NotImplementedError
        if not isinstance(K,Number) or not isinstance(Sigma, Number):
            raise NotImplementedError

        K = np.array(K)
        P = self._computeP2(K)
        return np.asscalar(-(self.R + self.gamma*P)/(1 - self.gamma))

    def grad_mixed(self, K, Sigma=None):
        I = np.eye(self.Q.shape[0], self.Q.shape[1])
        if not np.array_equal(self.A, I) or not np.array_equal(self.B, I):
            raise NotImplementedError
        if not isinstance(K,Number) or not isinstance(Sigma, Number):
            raise NotImplementedError
        theta = np.asscalar(np.array(K))

        den = 1 - self.gamma*(1 + 2*theta + theta**2)
        dePdeK = 2*(theta*self.R/den + self.gamma*(self.Q + theta**2*self.R)*(1+theta)/den**2)

        return np.asscalar(-dePdeK*self.gamma/(1 - self.gamma))

    def computeQFunction(self, x, u, K, Sigma, n_random_xn=100):
        """
        This function computes the Q-value of a pair (x,u) given the linear
        controller Kx + epsilon where epsilon \sim N(0, Sigma).
        Args:
            x (int, array): the state
            u (int, array): the action
            K (matrix): the controller matrix
            Sigma (matrix): covariance matrix of the zero-mean noise added to
            the controller action
            n_random_xn: the number of samples to draw in order to average over
            the next state

        Returns:
            Qfun (float): The Q-value in the given pair (x,u) under the given
            controller

        """
        if isinstance(x, Number):
            x = np.array([x])
        if isinstance(u, Number):
            u = np.array([u])
        if isinstance(K, Number):
            K = np.array([K]).reshape(1, 1)
        if isinstance(Sigma, Number):
            Sigma = np.array([Sigma]).reshape(1, 1)

        P = self._computeP2(K)
        Qfun = 0
        for i in range(n_random_xn):
            noise = self.np_random.randn() * self.sigma_noise
            action_noise = self.np_random.multivariate_normal(
                np.zeros(Sigma.shape[0]), Sigma, 1)
            nextstate = np.dot(self.A, x) + np.dot(self.B,
                                                   u + action_noise) + noise
            Qfun -= np.dot(x.T, np.dot(self.Q, x)) + \
                np.dot(u.T, np.dot(self.R, u)) + \
                self.gamma * np.dot(nextstate.T, np.dot(P, nextstate)) + \
                (self.gamma / (1 - self.gamma)) * \
                np.trace(np.dot(Sigma,
                                self.R + self.gamma *
                                np.dot(self.B.T, np.dot(P, self.B))))
        Qfun = np.asscalar(Qfun) / n_random_xn
        return Qfun


if __name__ == '__main__':

    env = LQG1D()
    theta_star = env.computeOptimalK()
    print('theta^* = ', theta_star)
    print('J^* = ', env.computeJ(theta_star,env.sigma_controller))

import numpy as np
import torch

METADATA_v0={
        'version' : 0,
        'sensor_r': 10.0,
        'fov' : 120,
        'sensor_r_sd': 0.2, # sensor range noise.
        'sensor_b_sd': 0.01, # sensor bearing noise.
        'target_init_cov': 30.0, # initial target diagonal Covariance.
        'target_init_vel': [0.0, 0.0], # target's initial velocity.
        'target_speed_limit': 2.0, # velocity limit of targets.
        'lin_dist_range_a2b':(0.0, 8.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 5.0),
        'ang_dist_range_b2t':(-np.pi, np.pi),
        'margin': 1.0, # a marginal distance btw targets and the agent.
        'margin2wall': 0.5, # a marginal distance from a wall.
        'action_v': [3, 2, 1, 0], # action primitives - linear velocities.
        'action_w': [np.pi/2, np.pi, 0, -np.pi/2,-np.pi], # action primitives - angular velocities.
        'const_q': 0.01, # target noise constant in beliefs.
        'const_q_true': 0.01, # target noise constant of actual targets.
    }

METADATA_v1={
        'version' : 1,
        'sensor_r': 10.0,
        'fov' : 120,
        'sensor_r_sd': 0.2, # sensor range noise.
        'sensor_b_sd': 0.01, # sensor bearing noise.
        'target_init_cov': 30.0, # initial target diagonal Covariance.
        'target_init_vel': [0.0, 0.0], # target's initial velocity.
        'target_speed_limit': .1, # velocity limit of targets.
        # 'lin_dist_range_a2b':(5.0, 20.0),
        # 'ang_dist_range_a2b':(-np.pi, np.pi),
        # 'lin_dist_range_b2t':(0.0, 20.0),
        # 'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'lin_dist_range_a2b':(5.0, 6.0),
        'ang_dist_range_a2b':(-0.1, 0.0),
        'lin_dist_range_b2t':(0.0, 1.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'margin': 1.0, # a marginal distance btw targets and the agent.
        'margin2wall': 1.0, # a marginal distance from a wall.
        'action_v': [3, 2, 1, 0], # action primitives - linear velocities.
        'action_w': [np.pi/2, 0, -np.pi/2], # action primitives - angular velocities.
        'const_q': 0.5, # target noise constant in beliefs.
        'const_q_true': 0.01, # target noise constant of actual targets.
    }

METADATA_multi_v1={
        'version' : 'm1',
        'sensor_r': 10.0,
        'fov' : 120,
        'sensor_r_sd': 0.2, # sensor range noise.
        'sensor_b_sd': 0.01, # sensor bearing noise.
        'target_init_cov': 30.0, # initial target diagonal Covariance.
        'target_init_vel': [0.0, 0.0], # target's initial velocity.
        'target_speed_limit': 1.0, # velocity limit of targets.
        'lin_dist_range_a2b':(5.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 10.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'margin': 1.0, # a marginal distance btw targets and the agent.
        'margin2wall': 1.0, # a marginal distance from a wall.
        'action_v': [3, 2, 1, 0], # action primitives - linear velocities.
        'action_w': [np.pi/2, 0, -np.pi/2], # action primitives - angular velocities.
        'const_q': 0.2, # target noise constant in beliefs.
        'const_q_true': 0.2, # target noise constant of actual targets.
    }

# Designate a metadata version to be used throughout the target tracking env.
METADATA = METADATA_v1

TTENV_EVAL_SET = [
        { # Tracking
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 3.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':False,
        'target_speed_limit': 3.2,
        'const_q':0.5,
        },
        { # Discovery
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(10.0, 15.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':None,
        'target_speed_limit': 2.0,
        'const_q': 0.2,
        },
        { # Navigation
        'lin_dist_range_a2b':(10.0, 20.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 3.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':True,
        'target_speed_limit': 2.0,
        'const_q': 0.2,
        },
]

TTENV_EVAL_MULTI_SET = [
        {
        'lin_dist_range_a2b':(3.0, 10.0),
        'ang_dist_range_a2b':(-np.pi, np.pi),
        'lin_dist_range_b2t':(0.0, 3.0),
        'ang_dist_range_b2t':(-np.pi/2, np.pi/2),
        'blocked':None,
        'target_speed_limit':1.0,
        'const_q':0.02,
        }
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

